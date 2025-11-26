# groknifty_production.py
# Option C - Hybrid (ML + statistical fallback) dynamic Nifty/BankNifty bot
# Put this at ~/grok-nifty/groknifty_production.py
# Requires python packages: python-telegram-bot==13.15, yfinance, scikit-learn, pandas, joblib, apscheduler, nltk, beautifulsoup4, requests, sqlite3 (stdlib), etc.

import os
import sys
import time
import logging
import sqlite3
import traceback
from datetime import datetime, timedelta
from functools import partial
from typing import Tuple, Dict, Any, List

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from telegram import Update, Bot, ParseMode, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, CallbackQueryHandler, CallbackContext
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Path to uploaded public key (developer note: transform to URL if needed)
SSH_PUB_KEY_PATH = "/mnt/data/ssh-key-2025-11-23.key (1).pub"

# -------------------------
# Configuration & Env load
# -------------------------
# Try environment variables, else try /etc/groknifty/groknifty.env
ENV_FILE = "/etc/groknifty/groknifty.env"

def load_env_file(path=ENV_FILE):
    if os.path.exists(path):
        try:
            # load lines like KEY=VALUE, ignore comments
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        v = v.strip().strip('"').strip("'")
                        os.environ.setdefault(k.strip(), v)
        except Exception as e:
            print("Failed to load env file:", e)

load_env_file()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
ADMIN_IDS = set()
if os.environ.get("ADMIN_IDS"):
    try:
        ADMIN_IDS = set(int(i.strip()) for i in os.environ.get("ADMIN_IDS", "").split(",") if i.strip())
    except:
        ADMIN_IDS = set()

if not TELEGRAM_TOKEN:
    print("ERROR: TELEGRAM_TOKEN not set in environment.")
    sys.exit(1)

# Prediction model files
MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "grok_model.joblib")
MODEL_META = os.path.join(MODEL_DIR, "grok_model_meta.joblib")

# DB
DB_PATH = os.path.join(os.getcwd(), "grok.db")

# URLs / scrapers
NEWS_URL = "https://www.moneycontrol.com/news/business/markets/"
FII_DII_URL = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.html"

# Logging
LOG_FILE = os.path.join(os.getcwd(), "groknifty.log")
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ],
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("grok-nifty")

# NLTK setup
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# Scheduler
scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
# We'll start scheduler after Updater initialization.

# -------------------------
# Utilities: DB
# -------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        tg_id INTEGER UNIQUE,
        tier TEXT DEFAULT 'free',
        created_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY,
        user_tg_id INTEGER,
        action TEXT,
        index_symbol TEXT,
        month TEXT,
        strike INTEGER,
        opt_type TEXT,
        price REAL,
        qty INTEGER,
        raw_text TEXT,
        created_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY,
        path TEXT,
        trained_at TEXT,
        score REAL
    )
    """)
    conn.commit()
    return conn

DB_CONN = init_db()

# -------------------------
# Helper: index detection
# -------------------------
def detect_index_from_text(text: str) -> str:
    t = text.upper()
    if "BANKNIFTY" in t or "BANK NIFTY" in t or ("BN" in t and "BANK" in t):
        return "BANKNIFTY"
    if "NIFTY" in t or "NSEI" in t:
        return "NIFTY"
    # fallback default
    return "BANKNIFTY"

def get_index_ticker(index_name: str) -> str:
    if index_name == "BANKNIFTY":
        return "^NSEBANK"
    if index_name == "NIFTY":
        return "^NSEI"
    return "^NSEBANK"

# -------------------------
# Data fetchers
# -------------------------
def fetch_spot_and_sentiment(index_name: str) -> Tuple[float, float]:
    """
    Returns (spot_price, sentiment_score)
    sentiment_score is average compound across top headlines
    """
    ticker = get_index_ticker(index_name)
    try:
        df = yf.download(ticker, period="2d", interval="1d", progress=False)
        spot = float(df["Close"].iloc[-1]) if not df.empty else None
    except Exception as e:
        logger.exception("yfinance failed: %s", e)
        spot = None

    # simple sentiment
    try:
        resp = requests.get(NEWS_URL, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Extract headlines broadly
        headlines = [h.text.strip() for h in soup.find_all(["h2", "h3"])][:10]
        if headlines:
            sentiment = np.mean([sia.polarity_scores(h)["compound"] for h in headlines])
        else:
            sentiment = 0.0
    except Exception as e:
        logger.warning("News fetch failed: %s", e)
        sentiment = 0.0

    if spot is None:
        # fallback: try ticker quick fetch
        try:
            info = yf.Ticker(ticker).history(period="1d")
            spot = float(info["Close"].iloc[-1]) if not info.empty else 0.0
        except:
            spot = 0.0

    return float(spot), float(sentiment)

def fetch_historical_df(index_name: str, days: int = 90) -> pd.DataFrame:
    ticker = get_index_ticker(index_name)
    df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
    if df is None or df.empty:
        raise RuntimeError("No historical data fetched for " + ticker)
    df = df.dropna()
    return df

# -------------------------
# Feature engineering & model
# -------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # basic features: returns, moving averages, vol
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_3"] = df["Close"].pct_change(3)
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["vol_5"] = df["Close"].rolling(5).std()
    df["vol_10"] = df["Close"].rolling(10).std()
    df["target_next"] = df["Close"].shift(-1)  # predict next day close
    df = df.dropna()
    return df

def train_model_for_index(index_name: str) -> Dict[str, Any]:
    """
    Train RandomForest and save model. Returns metadata dict.
    """
    try:
        df = fetch_historical_df(index_name, days=120)
        df = compute_features(df)
        features = ["Close", "ret_1", "ret_3", "ma_5", "ma_10", "vol_5", "vol_10"]
        X = df[features].values
        y = df["target_next"].values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        # Save model
        meta = {
            "index": index_name,
            "trained_at": datetime.now().isoformat(),
            "mae": float(mae)
        }
        file_name = f"grok_model_{index_name}.joblib"
        path = os.path.join(MODEL_DIR, file_name)
        joblib.dump({"model": model, "features": features}, path)
        # store meta
        joblib.dump(meta, MODEL_META.replace(".joblib", f"_{index_name}.joblib"))
        # Save DB record
        cur = DB_CONN.cursor()
        cur.execute("INSERT INTO models (path, trained_at, score) VALUES (?, ?, ?)", (path, meta["trained_at"], meta["mae"]))
        DB_CONN.commit()
        logger.info("Trained model %s saved to %s (mae=%s)", index_name, path, mae)
        return {"path": path, **meta}
    except Exception as e:
        logger.exception("Model training failed for %s: %s", index_name, e)
        return {}

def load_model_for_index(index_name: str):
    file_name = f"grok_model_{index_name}.joblib"
    path = os.path.join(MODEL_DIR, file_name)
    if os.path.exists(path):
        try:
            data = joblib.load(path)
            return data.get("model"), data.get("features")
        except Exception as e:
            logger.warning("Failed to load model at %s: %s", path, e)
            return None, None
    return None, None

# -------------------------
# Prediction wrapper (Hybrid)
# -------------------------
def predict_next(df: pd.DataFrame, index_name: str) -> Dict[str, Any]:
    """
    Returns dict with:
      spot, pred_price, pred_return, low, high, confidence
    """
    spot, sentiment = fetch_spot_and_sentiment(index_name)
    features = ["Close", "ret_1", "ret_3", "ma_5", "ma_10", "vol_5", "vol_10"]
    # Statistical fallback: simple range using today's close and vol
    last_close = float(df["Close"].iloc[-1])
    today_vol = float(df["Close"].pct_change().rolling(10).std().iloc[-1] or 0.0)
    # default fallback prediction
    low_fallback = last_close * (1 - 0.01 - today_vol)
    high_fallback = last_close * (1 + 0.01 + today_vol)
    pred_fallback = last_close * (1 + 0.001)  # tiny move
    confidence_fallback = 0.2

    # Try ML prediction
    model, feat_list = load_model_for_index(index_name)
    try:
        if model is not None:
            # prepare last row features
            df_feat = compute_features(df)
            last = df_feat.iloc[-1]
            x = np.array([last[f] for f in feat_list]).reshape(1, -1)
            pred_price = float(model.predict(x)[0])
            pred_return = (pred_price - last_close) / last_close
            # range using predicted return +/- volatility * factor
            vol = float(df_feat["ret_1"].rolling(10).std().iloc[-1] or 0.0)
            low = pred_price * (1 - 0.01 - vol)
            high = pred_price * (1 + 0.01 + vol)
            confidence = max(0.05, 1.0 - abs(pred_return) / 0.1)  # heuristic
            return {
                "spot": spot,
                "pred_price": pred_price,
                "pred_return": pred_return,
                "low": low,
                "high": high,
                "confidence": confidence,
                "sentiment": sentiment,
                "method": "ml"
            }
        else:
            raise RuntimeError("No model")
    except Exception as e:
        logger.warning("ML predict failed: %s. Using fallback", e)
        return {
            "spot": spot,
            "pred_price": pred_fallback,
            "pred_return": (pred_fallback - last_close) / last_close,
            "low": low_fallback,
            "high": high_fallback,
            "confidence": confidence_fallback,
            "sentiment": sentiment,
            "method": "fallback"
        }

# -------------------------
# Positions & Users helpers
# -------------------------
def get_user_row(tg_id: int):
    cur = DB_CONN.cursor()
    cur.execute("SELECT tg_id, tier FROM users WHERE tg_id = ?", (tg_id,))
    r = cur.fetchone()
    if r:
        return {"tg_id": r[0], "tier": r[1]}
    return None

def ensure_user(tg_id: int):
    cur = DB_CONN.cursor()
    row = get_user_row(tg_id)
    if row:
        return row
    cur.execute("INSERT INTO users (tg_id, tier, created_at) VALUES (?, ?, ?)", (tg_id, "free", datetime.now().isoformat()))
    DB_CONN.commit()
    return get_user_row(tg_id)

def save_position(parsed: dict):
    cur = DB_CONN.cursor()
    cur.execute("""
        INSERT INTO positions (user_tg_id, action, index_symbol, month, strike, opt_type, price, qty, raw_text, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (parsed.get("user_id"), parsed.get("action"), parsed.get("index_symbol"), parsed.get("month"),
          parsed.get("strike"), parsed.get("opt_type"), parsed.get("price"), parsed.get("qty"),
          parsed.get("raw_text"), datetime.now().isoformat()))
    DB_CONN.commit()

def get_positions_for_user(tg_id: int, limit=20) -> List[Dict[str, Any]]:
    cur = DB_CONN.cursor()
    cur.execute("SELECT action, index_symbol, month, strike, opt_type, price, qty, raw_text, created_at FROM positions WHERE user_tg_id = ? ORDER BY id DESC LIMIT ?", (tg_id, limit))
    rows = cur.fetchall()
    cols = ["action", "index_symbol", "month", "strike", "opt_type", "price", "qty", "raw_text", "created_at"]
    return [dict(zip(cols, r)) for r in rows]

# -------------------------
# Position text parser
# -------------------------
def parse_position_text(text: str, user_id: int) -> dict:
    """
    Expected patterns (flexible):
      BUY BANKNIFTY NOV 59500 CE @ 215 3 lots
      SELL NIFTY DEC 22400 PE @ 85 100
    Returns parsed dict or raises ValueError
    """
    raw = text.strip()
    parts = raw.upper().replace("/", " ").replace(",", " ").split()
    parsed = {"raw_text": raw, "user_id": user_id}
    try:
        # action
        parsed["action"] = parts[0] if parts else "BUY"
        # detect index
        parsed["index_symbol"] = detect_index_from_text(raw)
        # find month token (assume next after index or parts[1])
        # brute-force: find first 3-letter month token among parts
        month = None
        months = set(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])
        for p in parts:
            if p in months:
                month = p
                break
        parsed["month"] = month or ""
        # find strike (first integer > 1000)
        strike = None
        for p in parts:
            if p.isdigit() and int(p) > 1000:
                strike = int(p); break
        parsed["strike"] = strike or 0
        # option type CE/PE
        parsed["opt_type"] = "CE" if "CE" in parts else ("PE" if "PE" in parts else "")
        # price: find token after '@' or last numeric that's small (<10000)
        price = None
        if "@" in parts:
            atidx = parts.index("@")
            if atidx + 1 < len(parts):
                try: price = float(parts[atidx+1]); 
                except: price = None
        if price is None:
            nums = [p for p in parts if (p.replace(".","",1).isdigit())]
            for n in reversed(nums):
                val = float(n)
                if val < 10000:
                    price = val
                    break
        parsed["price"] = float(price) if price is not None else 0.0
        # qty: find last integer small (<=100000) or 'lots' mention
        qty = 1
        if "LOT" in raw.upper():
            # capture number before 'LOT' token
            tokens = raw.upper().split()
            for i,tok in enumerate(tokens):
                if "LOT" in tok and i>0:
                    try:
                        qty = int(tokens[i-1])
                    except:
                        qty = 1
                    break
            qty = qty * 1  # lot size handling is left to user (we keep raw qty)
        else:
            # find an integer that appears after price token or at end
            try:
                nums = [int(p) for p in parts if p.isdigit()]
                if nums:
                    qty = nums[-1]
            except:
                qty = 1
        parsed["qty"] = int(qty)
        return parsed
    except Exception as e:
        logger.exception("Position parse failed: %s", e)
        raise ValueError("Invalid position format")

# -------------------------
# Telegram bot handlers
# -------------------------
def get_tier_keyboard():
    keyboard = [
        [InlineKeyboardButton("Free", callback_data="tier_free")],
        [InlineKeyboardButton("Copper", callback_data="tier_copper")],
        [InlineKeyboardButton("Silver", callback_data="tier_silver")],
        [InlineKeyboardButton("Gold", callback_data="tier_gold")],
        [InlineKeyboardButton("Diamond", callback_data="tier_diamond")],
    ]
    return InlineKeyboardMarkup(keyboard)

def start_handler(update: Update, context: CallbackContext):
    user = update.effective_user
    ensure_user(user.id)
    update.message.reply_text("Welcome to GrokNifty AI Bot!\nSelect your tier:", reply_markup=get_tier_keyboard())

def tier_callback_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    user = query.from_user
    tier = query.data.split("_",1)[1] if "_" in query.data else "free"
    cur = DB_CONN.cursor()
    cur.execute("UPDATE users SET tier = ? WHERE tg_id = ?", (tier, user.id))
    DB_CONN.commit()
    query.answer(text=f"Tier set to {tier.capitalize()}")
    query.edit_message_text(text=f"Tier set to {tier.capitalize()}")

def position_message_handler(update: Update, context: CallbackContext):
    user = update.effective_user
    text = update.message.text
    ensure_user(user.id)
    try:
        parsed = parse_position_text(text, user.id)
        save_position(parsed)
        update.message.reply_text("✅ Position saved!", parse_mode=ParseMode.MARKDOWN)
    except ValueError:
        update.message.reply_text("❌ Invalid format.\nTry: BUY BANKNIFTY NOV 59500 CE @ 215 3 lots")

def predict_command_handler(update: Update, context: CallbackContext):
    user = update.effective_user
    ensure_user(user.id)
    # choose index based on latest position or user-provided param
    try:
        args = context.args or []
        if args:
            # user may choose explicit: /predict banknifty
            idx = detect_index_from_text(" ".join(args))
        else:
            positions = get_positions_for_user(user.id, limit=1)
            if positions:
                idx = positions[0].get("index_symbol", "BANKNIFTY")
            else:
                idx = "BANKNIFTY"
        # fetch df and predict
        df = fetch_historical_df(idx, days=120)
        out = predict_next(df, idx)
        spot = out["spot"]
        pred_price = out["pred_price"]
        pred_return_pct = out["pred_return"] * 100
        low = out["low"]
        high = out["high"]
        conf = out["confidence"] * 100
        method = out.get("method", "fallback")
        sentiment = out.get("sentiment", 0.0)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")
        # include user's last positions (up to 5)
        positions = get_positions_for_user(user.id, limit=5)
        pos_text = ""
        for p in positions:
            pos_text += f"\n• {p['action']} {p['index_symbol']} {p['month']} {p['strike']}{p['opt_type']} @ {p['price']} qty:{p['qty']}"
        resp = (
            f"*GrokNifty Prediction ({timestamp})*\n"
            f"*Index:* {idx}\n"
            f"*Spot:* {spot:.2f}\n"
            f"*Predicted Next Close:* {pred_price:.2f} ({pred_return_pct:.2f}%)\n"
            f"*Estimated Range:* Low {low:.2f} — High {high:.2f}\n"
            f"*Confidence:* {conf:.1f}%\n"
            f"*Sentiment:* {sentiment:.3f}\n"
            f"*Method:* {method}\n"
        )
        if pos_text:
            resp += f"\n*Positions:*{pos_text}"
        update.message.reply_text(resp, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.exception("Predict command failed: %s", e)
        update.message.reply_text("Prediction failed. Try again later.")

# Admin commands
def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

def admin_set_tier(update: Update, context: CallbackContext):
    user = update.effective_user
    if not is_admin(user.id):
        update.message.reply_text("Not authorized.")
        return
    try:
        args = context.args
        if len(args) < 2:
            update.message.reply_text("Usage: /settier <tg_id> <tier>")
            return
        tg = int(args[0])
        tier = args[1]
        cur = DB_CONN.cursor()
        cur.execute("UPDATE users SET tier = ? WHERE tg_id = ?", (tier, tg))
        if cur.rowcount == 0:
            # create user
            cur.execute("INSERT OR IGNORE INTO users (tg_id, tier, created_at) VALUES (?, ?, ?)", (tg, tier, datetime.now().isoformat()))
        DB_CONN.commit()
        update.message.reply_text(f"Set tier {tier} for {tg}")
    except Exception as e:
        logger.exception("settier failed: %s", e)
        update.message.reply_text("Failed to set tier.")

def admin_train_now(update: Update, context: CallbackContext):
    user = update.effective_user
    if not is_admin(user.id):
        update.message.reply_text("Not authorized.")
        return
    # train for both indices in background
    update.message.reply_text("Training started (both indices).")
    scheduler.add_job(lambda: train_job_wrapper("BANKNIFTY"), id=f"train_BN_{int(time.time())}")
    scheduler.add_job(lambda: train_job_wrapper("NIFTY"), id=f"train_NF_{int(time.time())}")

# training wrapper
def train_job_wrapper(index_name: str):
    logger.info("Manual training triggered for %s", index_name)
    meta = train_model_for_index(index_name)
    logger.info("Finished training: %s", meta)
    return meta

# -------------------------
# Scheduler tasks
# -------------------------
def scheduled_daily_train():
    # train both indexes at 02:00 IST
    try:
        logger.info("Daily scheduled training started")
        train_model_for_index("BANKNIFTY")
        train_model_for_index("NIFTY")
        logger.info("Daily scheduled training finished")
    except Exception as e:
        logger.exception("Daily train failed: %s", e)

def scheduled_auto_push(bot: Bot):
    # Push short auto prediction to all users based on their last index
    try:
        logger.info("Auto push task running")
        cur = DB_CONN.cursor()
        cur.execute("SELECT tg_id FROM users")
        rows = cur.fetchall()
        for (tg,) in rows:
            try:
                positions = get_positions_for_user(tg, limit=1)
                idx = positions[0]["index_symbol"] if positions else "BANKNIFTY"
                df = fetch_historical_df(idx, days=90)
                out = predict_next(df, idx)
                text = (
                    f"GrokNifty Auto ({idx})\n"
                    f"Spot: {out['spot']:.2f}\n"
                    f"Range: {out['low']:.2f} - {out['high']:.2f}\n"
                    f"Confidence: {out['confidence']*100:.1f}%\n"
                    f"Method: {out['method']}"
                )
                bot.send_message(chat_id=tg, text=text)
                time.sleep(0.1)
            except Exception as e:
                logger.exception("Failed to push for %s: %s", tg, e)
    except Exception as e:
        logger.exception("Auto push failed: %s", e)

# -------------------------
# Startup & main
# -------------------------
def main():
    # Updater & Dispatcher
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Handlers
    dp.add_handler(CommandHandler("start", start_handler))
    dp.add_handler(CallbackQueryHandler(tier_callback_handler, pattern=r"^tier_"))
    dp.add_handler(CommandHandler("predict", predict_command_handler))
    dp.add_handler(MessageHandler(Filters.text & (~Filters.command), position_message_handler))
    dp.add_handler(CommandHandler("settier", admin_set_tier))       # admin-only
    dp.add_handler(CommandHandler("trainnow", admin_train_now))    # admin-only

    # Start the updater
    updater.start_polling()
    bot = updater.bot
    logger.info("Bot polling started")

    # Scheduler jobs
    # daily training at 02:00 IST
    try:
        scheduler.add_job(scheduled_daily_train, "cron", hour=2, minute=0, id="daily_train")
        # auto push before market open (example times - adjust as you like)
        scheduler.add_job(partial(scheduled_auto_push, bot), "cron", hour=9, minute=10, id="auto_push_morning")
        scheduler.add_job(partial(scheduled_auto_push, bot), "cron", hour=12, minute=0, id="auto_push_midday")
        scheduler.start()
        logger.info("Scheduler started with jobs: %s", scheduler.get_jobs())
    except Exception as e:
        logger.exception("Scheduler start failed: %s", e)

    # Warm-start: if no model exists, train quickly (non-blocking)
    for idx in ("BANKNIFTY", "NIFTY"):
        model, _ = load_model_for_index(idx)
        if model is None:
            logger.info("No model found for %s, training now (background).", idx)
            # schedule immediate training without blocking
            scheduler.add_job(lambda i=idx: train_job_wrapper(i), id=f"startup_train_{idx}", replace_existing=True)

    updater.idle()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Fatal error in main: %s", e)
        raise