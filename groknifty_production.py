#!/usr/bin/env python3
"""
GrokNifty Production Bot
- Supports NIFTY & BANKNIFTY predictions
- Collects user positions
- Auto-trains daily
- Sends scheduled predictions (pre-market + every 30 minutes during market hours)
"""
import os
import sys
import time
import json
import sqlite3
import logging
from functools import partial
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional, Dict, Any, List

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler, CallbackContext

# ML
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import joblib

# Scheduler
from apscheduler.schedulers.background import BackgroundScheduler

# Sentiment
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# Logging
LOG_FILE = os.getenv("GROK_LOG", "groknifty.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("grok")

# ENV: TELEGRAM_TOKEN should be exported or stored in /etc/groknifty/groknifty.env
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN not found in environment. Exiting.")
    raise SystemExit("TELEGRAM_TOKEN not set in environment.")

# DB and model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "grok.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE_NIFTY = os.path.join(MODEL_DIR, "model_nifty.joblib")
MODEL_FILE_BANK = os.path.join(MODEL_DIR, "model_bank.joblib")

# Prediction config
MARKET_TZ = timezone(timedelta(hours=5, minutes=30))  # IST
MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)

# Create DB if missing
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        user_id INTEGER PRIMARY KEY,
        tier TEXT DEFAULT 'free',
        created_at TEXT
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS positions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        raw_text TEXT,
        action TEXT,
        symbol TEXT,
        month TEXT,
        strike INTEGER,
        opt_type TEXT,
        price REAL,
        qty INTEGER,
        created_at TEXT
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        symbol TEXT,
        spot REAL,
        pred_next REAL,
        low REAL,
        high REAL,
        confidence REAL,
        metadata TEXT
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS model_metrics(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        symbol TEXT,
        metric_name TEXT,
        value REAL
    )""")
    conn.commit()
    conn.close()

init_db()

# --- Utilities ---
def now_str():
    return datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M:%S")

def inside_market_hours(dt: datetime = None) -> bool:
    dt = dt or datetime.now(MARKET_TZ)
    start = dt.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
    end = dt.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0)
    return start <= dt <= end

# --- Data fetching ---
def fetch_spot(symbol: str) -> Optional[float]:
    """
    symbol: 'NIFTY' or 'BANKNIFTY'
    We use yfinance tickers:
      - NIFTY -> ^NSEI
      - BANKNIFTY -> ^NSEBANK
    """
    ticker_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
    tk = ticker_map.get(symbol.upper())
    try:
        if not tk:
            return None
        df = yf.download(tk, period="2d", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception as e:
        logger.exception("fetch_spot failed for %s: %s", symbol, e)
        return None

def fetch_fii_dii() -> Tuple[float, float]:
    """
    Try to scrape FII/DII from MoneyControl. Fallback to 0,0 on failure.
    """
    url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.html"
    try:
        resp = requests.get(url, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Best-effort parsing: find numbers with labels FII / DII
        fii_node = soup.find(text=lambda t: t and "FII" in t.upper())
        dii_node = soup.find(text=lambda t: t and "DII" in t.upper())
        def extract_near(node):
            if not node:
                return 0.0
            nxt = node.find_next(string=True)
            if nxt:
                s = "".join(ch for ch in nxt if ch.isdigit() or ch in ",.-")
                s = s.replace(",", "")
                try:
                    return float(s)
                except:
                    return 0.0
            return 0.0
        return extract_near(fii_node), extract_near(dii_node)
    except Exception as e:
        logger.warning("fetch_fii_dii failed: %s", e)
        return 0.0, 0.0

def fetch_news_sentiment(limit=10) -> float:
    """
    Fetch headlines from MoneyControl markets news and compute Vader sentiment average.
    """
    url = "https://www.moneycontrol.com/news/business/markets/"
    try:
        resp = requests.get(url, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        headlines = [h.get_text(strip=True) for h in soup.find_all(["h2","a","h3"], limit=limit)]
        if not headlines:
            return 0.0
        scores = [sia.polarity_scores(h)["compound"] for h in headlines if h]
        return float(np.mean(scores)) if scores else 0.0
    except Exception as e:
        logger.warning("fetch_news_sentiment failed: %s", e)
        return 0.0

# --- ML helpers ---
def load_training_history(symbol: str, days=180) -> pd.DataFrame:
    """
    Pull historical close prices from yfinance and augment with FII/DII, sentiment features.
    Used to train simple model.
    """
    map_tick = {"NIFTY":"^NSEI", "BANKNIFTY":"^NSEBANK"}
    tk = map_tick.get(symbol.upper())
    if not tk:
        return pd.DataFrame()
    try:
        df = yf.download(tk, period=f"{days}d", interval="1d", progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df[["Close", "Open", "High", "Low", "Volume"]].copy()
        df.rename(columns={"Close":"close"}, inplace=True)
        # Feature engineering: returns, vol, moving averages
        df["ret_1"] = df["close"].pct_change(1).fillna(0)
        df["ma_5"] = df["close"].rolling(5).mean().fillna(method="bfill")
        df["ma_10"] = df["close"].rolling(10).mean().fillna(method="bfill")
        df["vol_5"] = df["Volume"].rolling(5).mean().fillna(method="bfill")
        df["day"] = df.index.dayofweek
        # Simple placeholders for FII/DII and sentiment (static or approximated per day if you have more data)
        df["fii_minus_dii"] = 0.0
        df["sentiment"] = 0.0
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logger.exception("load_training_history failed for %s: %s", symbol, e)
        return pd.DataFrame()

def train_and_save_model(symbol: str):
    """
    Train a simple LinearRegression to predict next day's close using current day's features.
    Save model to disk and store metrics in DB.
    """
    df = load_training_history(symbol, days=365)
    if df.empty or len(df) < 30:
        logger.warning("Not enough history to train for %s", symbol)
        return False
    # target is next close
    df["target"] = df["close"].shift(-1)
    df = df.dropna()
    features = ["close", "ret_1", "ma_5", "ma_10", "vol_5", "day", "fii_minus_dii", "sentiment"]
    X = df[features].values
    y = df["target"].values
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    try:
        mape = float(mean_absolute_percentage_error(y, preds))
    except Exception:
        mape = 0.0
    # persist
    file = MODEL_FILE_NIFTY if symbol.upper()=="NIFTY" else MODEL_FILE_BANK
    joblib.dump({"model": model, "features": features}, file)
    # store metric
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO model_metrics(ts, symbol, metric_name, value) VALUES(?,?,?,?)",
              (now_str(), symbol.upper(), "mape", mape))
    conn.commit()
    conn.close()
    logger.info("Trained model for %s saved to %s (mape=%.4f)", symbol, file, mape)
    return True

def load_model(symbol: str):
    file = MODEL_FILE_NIFTY if symbol.upper()=="NIFTY" else MODEL_FILE_BANK
    if not os.path.exists(file):
        return None
    try:
        return joblib.load(file)
    except Exception:
        logger.exception("Failed to load model for %s", symbol)
        return None

def predict_next(symbol: str, spot: float, fii_minus_dii: float, sentiment: float) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Returns: (pred_next_close, low, high, confidence)
    If no model available, fallback to simple rule-of-thumb
    """
    model_pack = load_model(symbol)
    try:
        if model_pack:
            model = model_pack["model"]
            features = model_pack["features"]
            # create synthetic feature vector:
            ma5 = spot  # fallback: use spot for MA
            ma10 = spot
            vol5 = 0.0
            day = datetime.now(MARKET_TZ).weekday()
            x = np.array([[spot, 0.0, ma5, ma10, vol5, day, fii_minus_dii, sentiment]])
            pred = float(model.predict(x)[0])
            # confidence approx inverse of last MAPE metric
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT value FROM model_metrics WHERE symbol=? ORDER BY id DESC LIMIT 1", (symbol.upper(),))
            row = c.fetchone()
            conn.close()
            mape = float(row[0]) if row else 0.05
            confidence = max(0.01, 1.0 - mape)  # simple mapping
            low = pred - abs(pred - spot) - 0.01 * spot
            high = pred + abs(pred - spot) + 0.01 * spot
            return pred, low, high, confidence
        else:
            # fallback heuristic:
            bias = fii_minus_dii / 1000.0 + sentiment * 5.0
            pred = spot * (1.0 + bias * 0.0005)
            low = spot - 100
            high = spot + 100
            confidence = 0.05
            return pred, low, high, confidence
    except Exception as e:
        logger.exception("predict_next failed for %s: %s", symbol, e)
        return None, None, None, None

# --- DB helpers for users & positions ---
def ensure_user(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
    if c.fetchone() is None:
        c.execute("INSERT INTO users(user_id, tier, created_at) VALUES(?,?,?)",
                  (user_id, "free", now_str()))
        conn.commit()
    conn.close()

def set_user_tier(user_id: int, tier: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET tier=? WHERE user_id=?", (tier.lower(), user_id))
    conn.commit()
    conn.close()

def get_user_tier(user_id: int) -> str:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT tier FROM users WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    return (row[0] if row else "free")

def save_position(user_id: int, raw_text: str, parsed: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO positions(user_id, raw_text, action, symbol, month, strike, opt_type, price, qty, created_at)
        VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (user_id, raw_text, parsed.get("action"), parsed.get("symbol"), parsed.get("month"),
          parsed.get("strike"), parsed.get("opt_type"), parsed.get("price"), parsed.get("qty"), now_str()))
    conn.commit()
    conn.close()

def list_positions(user_id: int, limit=20) -> List[Dict[str,Any]]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, raw_text, action, symbol, strike, opt_type, price, qty, created_at FROM positions WHERE user_id=? ORDER BY id DESC LIMIT ?",
              (user_id, limit))
    rows = c.fetchall()
    conn.close()
    res = []
    for r in rows:
        res.append({
            "id": r[0], "raw": r[1], "action": r[2], "symbol": r[3],
            "strike": r[4], "opt_type": r[5], "price": r[6], "qty": r[7], "time": r[8]
        })
    return res

# --- Position parser ---
def parse_position_text(text: str) -> Optional[dict]:
    """
    Expected formats (case-insensitive):
      BUY BANKNIFTY NOV 59500 CE @ 215 3
      SELL NIFTY DEC 18000 PE @ 45 1
    Returns parsed dict or None
    """
    try:
        t = text.upper().strip().replace(",", " ")
        tokens = t.split()
        # minimum tokens: action symbol month strike opt @ price qty
        if len(tokens) < 7:
            return None
        action = tokens[0]
        if action not in ("BUY","SELL"):
            return None
        symbol = tokens[1]
        month = tokens[2]
        # strike may sometimes be like '59500CE' but we assume separate
        strike = None
        opt_type = None
        price = None
        qty = 1
        # find '@' marker
        if "@" in tokens:
            at_idx = tokens.index("@")
            # strike is token at 3
            try:
                strike = int(tokens[3])
            except:
                return None
            opt_type = tokens[4] if len(tokens) > 4 else ""
            # price token after @
            if len(tokens) > at_idx+1:
                try:
                    price = float(tokens[at_idx+1])
                except:
                    price = None
            # qty could be after price
            if len(tokens) > at_idx+2:
                try:
                    qty = int(tokens[at_idx+2])
                except:
                    # user might specify '3 LOT' or '3 LOTS' or '3'
                    qtoken = tokens[at_idx+2]
                    if qtoken.isdigit():
                        qty = int(qtoken)
                    elif qtoken.endswith("LOT") and qtoken[:-3].isdigit():
                        qty = int(qtoken[:-3])
            return {
                "action": action,
                "symbol": symbol,
                "month": month,
                "strike": strike,
                "opt_type": opt_type,
                "price": price,
                "qty": qty
            }
        else:
            return None
    except Exception as e:
        logger.exception("parse_position_text failed: %s", e)
        return None

# --- Bot command handlers ---
def get_tier_markup():
    keyboard = [
        [InlineKeyboardButton("Free", callback_data="tier_free")],
        [InlineKeyboardButton("Copper", callback_data="tier_copper")],
        [InlineKeyboardButton("Silver", callback_data="tier_silver")],
        [InlineKeyboardButton("Gold", callback_data="tier_gold")],
        [InlineKeyboardButton("Diamond", callback_data="tier_diamond")],
    ]
    return InlineKeyboardMarkup(keyboard)

def cmd_start(update: Update, context: CallbackContext):
    user = update.effective_user
    ensure_user(user.id)
    update.message.reply_text("Welcome to GrokNifty AI Bot! Choose your tier:",
                              reply_markup=get_tier_markup())

def tier_callback(update: Update, context: CallbackContext):
    query = update.callback_query
    user_id = query.from_user.id
    payload = query.data  # like 'tier_free'
    parts = payload.split("_")
    if len(parts) == 2:
        tier = parts[1]
        set_user_tier(user_id, tier)
        query.answer()
        query.edit_message_text(text=f"Tier set to {tier.capitalize()}")

def cmd_positions(update: Update, context: CallbackContext):
    user = update.effective_user
    ensure_user(user.id)
    rows = list_positions(user.id, limit=20)
    if not rows:
        update.message.reply_text("No positions found for you.")
        return
    lines = []
    for r in rows:
        lines.append(f"{r['time']} - {r['raw']}")
    update.message.reply_text("Your positions:\n" + "\n".join(lines))

def cmd_position_add(update: Update, context: CallbackContext):
    """Handles a free-text position message. We accept both direct messages and /position <text>"""
    user = update.effective_user
    text = update.message.text
    # if command style "/position BUY ..." then remove leading "/position"
    if text.startswith("/position"):
        text = text[len("/position"):].strip()
    parsed = parse_position_text(text)
    if not parsed:
        update.message.reply_text("❌ Invalid format. Try: BUY BANKNIFTY NOV 59500 CE @ 215 3\nUse: BUY/SELL SYMBOL MONTH STRIKE CE/PE @ PRICE QTY")
        return
    ensure_user(user.id)
    save_position(user.id, text, parsed)
    update.message.reply_text("✅ Position saved!")

def build_prediction_message(symbols: List[str], meta: Dict[str, Any]) -> str:
    ts = now_str()
    lines = [f"*GrokNifty Prediction ({ts}):*"]
    for s in symbols:
        info = meta.get(s, {})
        spot = info.get("spot")
        if spot is None:
            lines.append(f"\n*{s}*\nSpot: N/A\nPrediction: N/A\n")
            continue
        pred = info.get("pred")
        low = info.get("low")
        high = info.get("high")
        conf = info.get("conf")
        lines.append(f"\n*{s}*\nSpot: {spot:.2f}\nPredicted Next Close: {pred:.2f} ({(pred-spot)/spot*100:.2f}%)\nEstimated Range: Low {low:.2f} – High {high:.2f}\nConfidence: {conf*100:.1f}%\n")
    # metadata
    lines.append(f"\nFII-DII (diff): {meta.get('fii_diff',0):.1f}")
    lines.append(f"Sentiment (news avg): {meta.get('sentiment',0):.3f}")
    return "\n".join(lines)

def cmd_predict(update: Update, context: CallbackContext):
    user = update.effective_user
    ensure_user(user.id)
    # check tier
    tier = get_user_tier(user.id)
    if tier == "free":
        update.message.reply_text("Upgrade your tier to access predictions.")
        update.message.reply_text("Choose your subscription tier:", reply_markup=get_tier_markup())
        return
    # fetch data
    fio, dii = fetch_fii_dii()
    fii_diff = fio - dii
    sentiment = fetch_news_sentiment(limit=10)
    meta = {}
    for sym in ("NIFTY","BANKNIFTY"):
        spot = fetch_spot(sym)
        if spot is None:
            meta[sym] = {"spot": None}
            continue
        pred, low, high, conf = predict_next(sym, spot, fii_diff, sentiment)
        meta[sym] = {"spot": spot, "pred": pred, "low": low, "high": high, "conf": conf}
        # persist to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO predictions(ts, symbol, spot, pred_next, low, high, confidence, metadata) VALUES(?,?,?,?,?,?,?,?)",
                  (now_str(), sym, spot, pred, low, high, conf, json.dumps({"fii_diff": fii_diff, "sentiment": sentiment})))
        conn.commit()
        conn.close()
    text = build_prediction_message(["NIFTY","BANKNIFTY"], {**meta, "fii_diff": fii_diff, "sentiment": sentiment})
    update.message.reply_text(text, parse_mode="Markdown")

# --- Scheduled automatic pushes ---
def auto_push_predictions(bot):
    """Push predictions to all users (respecting tier restrictions)."""
    try:
        # Collect meta
        fio, dii = fetch_fii_dii()
        fii_diff = fio - dii
        sentiment = fetch_news_sentiment(limit=10)
        meta = {}
        for sym in ("NIFTY","BANKNIFTY"):
            spot = fetch_spot(sym)
            if spot is None:
                meta[sym] = {"spot": None}
                continue
            pred, low, high, conf = predict_next(sym, spot, fii_diff, sentiment)
            meta[sym] = {"spot": spot, "pred": pred, "low": low, "high": high, "conf": conf}
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO predictions(ts, symbol, spot, pred_next, low, high, confidence, metadata) VALUES(?,?,?,?,?,?,?,?)",
                      (now_str(), sym, spot, pred, low, high, conf, json.dumps({"fii_diff": fii_diff, "sentiment": sentiment})))
            conn.commit()
            conn.close()
        # Build message
        msg = build_prediction_message(["NIFTY","BANKNIFTY"], {**meta, "fii_diff": fii_diff, "sentiment": sentiment})
        # Send to users who are not free
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT user_id, tier FROM users")
        rows = c.fetchall()
        conn.close()
        for uid, tier in rows:
            if tier and tier.lower() != "free":
                try:
                    bot.send_message(chat_id=uid, text=msg, parse_mode="Markdown")
                except Exception as e:
                    logger.warning("Failed to send auto prediction to %s: %s", uid, e)
    except Exception as e:
        logger.exception("auto_push_predictions error: %s", e)

def daily_train_job():
    """Train both models after market close (or daily at fixed time)."""
    logger.info("Daily training job started.")
    try:
        train_and_save_model("NIFTY")
        train_and_save_model("BANKNIFTY")
    except Exception as e:
        logger.exception("daily_train_job error: %s", e)

# --- Admin helpers (simple) ---
ADMIN_IDS = set()  # optionally populate with admin user ids

def cmd_admin_set_tier(update: Update, context: CallbackContext):
    # usage: /set_tier user_id tier
    user = update.effective_user
    if user.id not in ADMIN_IDS:
        update.message.reply_text("Unauthorized")
        return
    args = context.args
    if len(args) < 2:
        update.message.reply_text("Usage: /set_tier <user_id> <tier>")
        return
    try:
        uid = int(args[0])
        tier = args[1]
        set_user_tier(uid, tier)
        update.message.reply_text(f"Tier for {uid} set to {tier}")
    except Exception as e:
        update.message.reply_text("Error: " + str(e))

# --- Setup bot & scheduler ---
def run_bot():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Command handlers
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CallbackQueryHandler(tier_callback))
    dp.add_handler(CommandHandler("positions", cmd_positions))
    dp.add_handler(CommandHandler("predict", cmd_predict))
    dp.add_handler(CommandHandler("position", cmd_position_add))
    dp.add_handler(CommandHandler("set_tier", cmd_admin_set_tier, pass_args=True))

    # Accept plain text position messages
    dp.add_handler(MessageHandler(Filters.text & (~Filters.command), cmd_position_add))

    # Scheduler
    scheduler = BackgroundScheduler(timezone=MARKET_TZ)
    # Pre-market at 09:10
    scheduler.add_job(partial(auto_push_predictions, updater.bot), "cron", hour="9", minute="10")
    # Then every 30 minutes during the day (09:15..15:30)
    scheduler.add_job(partial(auto_push_predictions, updater.bot), "cron", minute="0,30", hour="9-15")
    # Daily training job at 16:00 IST
    scheduler.add_job(daily_train_job, "cron", hour="16", minute="0")
    scheduler.start()
    logger.info("Scheduler started")

    # Start bot
    updater.start_polling()
    logger.info("Bot started, polling updates.")
    updater.idle()

if __name__ == "__main__":
    try:
        logger.info("Starting GrokNifty production bot")
        run_bot()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down.")
    except Exception as e:
        logger.exception("Bot crashed: %s", e)
        raise
