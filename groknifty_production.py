#!/usr/bin/env python3
"""
Groknifty Production (Option C + Admin Dashboard + ML + Greeks + Charts)

Save as ~/grok-nifty/groknifty_production.py and run inside your venv.
Requires environment variables (TELEGRAM_TOKEN, ADMIN_IDS optional).
"""

import os
import sys
import time
import math
import json
import logging
import sqlite3
import traceback
from datetime import datetime, timedelta
from functools import partial

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from telegram import Update, ParseMode, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, CallbackQueryHandler
from flask import Flask, jsonify, send_file, request, render_template_string, abort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# ---------------------------
# Config / Env Loading
# ---------------------------
ENV_FILE = "/etc/groknifty/groknifty.env"
def load_env(path=ENV_FILE):
    if os.path.exists(path):
        try:
            for line in open(path, "r").read().splitlines():
                if not line or line.strip().startswith("#"): continue
                if "=" in line:
                    k,v = line.split("=",1)
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k.strip(), v)
        except Exception as e:
            print("Failed to load env file:", e)

load_env()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
ADMIN_IDS = set()
if os.environ.get("ADMIN_IDS"):
    try:
        ADMIN_IDS = set(int(x.strip()) for x in os.environ.get("ADMIN_IDS","").split(",") if x.strip())
    except:
        ADMIN_IDS = set()

FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5000"))

if not TELEGRAM_TOKEN:
    print("ERROR: TELEGRAM_TOKEN not set")
    sys.exit(1)

# Paths
BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "models"); os.makedirs(MODEL_DIR, exist_ok=True)
CHARTS_DIR = os.path.join(BASE_DIR, "charts"); os.makedirs(CHARTS_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, "groknifty.db")
LOG_PATH = os.path.join(BASE_DIR, "groknifty.log")

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("grok-nifty")

# ---------------------------
# NLTK
# ---------------------------
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# ---------------------------
# Globals
# ---------------------------
NEWS_URL = "https://www.moneycontrol.com/news/business/markets/"
FII_DII_URL = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.html"
scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
app = Flask(__name__)

# ---------------------------
# Database (SQLite)
# ---------------------------
def init_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tg_id INTEGER UNIQUE,
            tier TEXT DEFAULT 'free',
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tg_id INTEGER,
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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            index_symbol TEXT,
            path TEXT,
            trained_at TEXT,
            mae REAL,
            accuracy REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tg_id INTEGER,
            index_symbol TEXT,
            spot REAL,
            pred_price REAL,
            pred_return REAL,
            low REAL,
            high REAL,
            confidence REAL,
            method TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    return conn

DB = init_db()

# ---------------------------
# Utility: Index helpers
# ---------------------------
def detect_index_from_text(text: str) -> str:
    t = text.upper()
    if "BANKNIFTY" in t or "BANK NIFTY" in t or ("BN" in t and "BANK" in t):
        return "BANKNIFTY"
    if "NIFTY" in t or "NSEI" in t:
        return "NIFTY"
    return "BANKNIFTY"

def index_to_ticker(index_name: str) -> str:
    if index_name == "BANKNIFTY": return "^NSEBANK"
    if index_name == "NIFTY": return "^NSEI"
    return "^NSEBANK"

# ---------------------------
# Fetchers: spot, news, FII/DII
# ---------------------------
def fetch_spot(index_name: str) -> float:
    ticker = index_to_ticker(index_name)
    try:
        df = yf.download(ticker, period="2d", interval="1d", progress=False)
        if not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception as e:
        logger.warning("yf fetch failed for %s: %s", ticker, e)
    # fallback via Ticker.history
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except:
        pass
    return 0.0

def fetch_fii_dii_and_sentiment() -> tuple:
    # returns (fii, dii, sentiment)
    fii = 0.0; dii = 0.0; sentiment = 0.0
    try:
        r = requests.get(FII_DII_URL, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        # best-effort parse
        spans = soup.find_all("span")
        # naive: find elements with 'FII' or 'DII' nearby - many pages vary; this is best-effort
        text = soup.get_text(" ").upper()
        if "FII" in text:
            # fallback to 0 if parsing fails
            pass
    except Exception as e:
        logger.debug("FII/DII fetch failed: %s", e)
    # Sentiment from Moneycontrol headlines
    try:
        r2 = requests.get(NEWS_URL, timeout=8)
        soup2 = BeautifulSoup(r2.text, "html.parser")
        heads = [h.text.strip() for h in soup2.find_all(["h2","h3"])][:12]
        if heads:
            sentiment = np.mean([sia.polarity_scores(h)["compound"] for h in heads])
    except Exception as e:
        logger.debug("News fetch failed: %s", e)
    return fii, dii, sentiment

# ---------------------------
# Historical & features
# ---------------------------
def fetch_historical(index_name: str, days:int=120) -> pd.DataFrame:
    ticker = index_to_ticker(index_name)
    df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No historical data for {ticker}")
    df = df.dropna()
    return df

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret1"] = df["Close"].pct_change(1)
    df["ret3"] = df["Close"].pct_change(3)
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma10"] = df["Close"].rolling(10).mean()
    df["vol5"] = df["Close"].rolling(5).std()
    df["vol10"] = df["Close"].rolling(10).std()
    df["target_next"] = df["Close"].shift(-1)
    df = df.dropna()
    return df

# ---------------------------
# Train models (regressor + classifier)
# ---------------------------
def train_models(index_name: str) -> dict:
    """
    Trains a regressor (RandomForestRegressor) to predict next close,
    and a classifier (RandomForestClassifier) to predict up/down probability.
    Saves both to joblib in MODEL_DIR. Returns metadata.
    """
    try:
        df = fetch_historical(index_name, days=240)
        df_feat = compute_features(df)
        features = ["Close", "ret1", "ret3", "ma5", "ma10", "vol5", "vol10"]
        X = df_feat[features].values
        y_reg = df_feat["target_next"].values
        # classifier target: 1 if next close > today close
        y_clf = (df_feat["target_next"].values > df_feat["Close"].values).astype(int)
        X_train, X_val, y_r_train, y_r_val = train_test_split(X, y_reg, test_size=0.2, shuffle=False)
        _, _, y_c_train, y_c_val = train_test_split(X, y_reg, y_clf, test_size=0.2, shuffle=False)
        # scaler
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        # regressor
        regr = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        regr.fit(X_train_s, y_r_train)
        preds = regr.predict(X_val_s)
        mae = float(mean_absolute_error(y_r_val, preds))
        # classifier
        clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        clf.fit(X_train_s, y_c_train)
        cpred = clf.predict(X_val_s)
        acc = float(accuracy_score(y_c_val, cpred))
        # save
        model_obj = {"regr": regr, "clf": clf, "scaler": scaler, "features": features}
        fname = os.path.join(MODEL_DIR, f"grok_{index_name}.joblib")
        joblib.dump(model_obj, fname)
        # record in DB
        cur = DB.cursor()
        cur.execute("INSERT INTO models (index_symbol,path,trained_at,mae,accuracy) VALUES (?,?,?,?,?)",
                    (index_name, fname, datetime.now().isoformat(), mae, acc))
        DB.commit()
        logger.info("Trained models for %s saved (mae=%.4f acc=%.4f)", index_name, mae, acc)
        return {"index": index_name, "path": fname, "mae": mae, "accuracy": acc}
    except Exception as e:
        logger.exception("Training failed for %s: %s", index_name, e)
        return {}

def load_model(index_name: str):
    fname = os.path.join(MODEL_DIR, f"grok_{index_name}.joblib")
    if os.path.exists(fname):
        try:
            return joblib.load(fname)
        except Exception as e:
            logger.warning("Failed to load model %s: %s", fname, e)
            return None
    return None

# ---------------------------
# Greeks (Black-Scholes) utilities
# ---------------------------
from math import log, sqrt, exp
from scipy.stats import norm

def bs_price(S, K, T, r, sigma, option_type="call"):
    # S: spot, K: strike, T: time (years), r: risk-free, sigma: vol
    if T <= 0:
        if option_type == "call":
            return max(0.0, S-K)
        else:
            return max(0.0, K-S)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if option_type.lower().startswith("c"):
        price = S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
    else:
        price = K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return float(price)

def bs_greeks(S,K,T,r,sigma,option_type="call"):
    if T <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    delta = norm.cdf(d1) if option_type.lower().startswith("c") else (-norm.cdf(-d1))
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    theta = (-S*norm.pdf(d1)*sigma / (2*math.sqrt(T)) - r*K*exp(-r*T)*norm.cdf(d2)) if option_type.lower().startswith("c") else (-S*norm.pdf(d1)*sigma/(2*math.sqrt(T)) + r*K*exp(-r*T)*norm.cdf(-d2))
    vega = S * norm.pdf(d1) * math.sqrt(T)
    rho = K * T * exp(-r*T) * norm.cdf(d2) if option_type.lower().startswith("c") else -K * T * exp(-r*T) * norm.cdf(-d2)
    return {"delta": float(delta), "gamma": float(gamma), "theta": float(theta), "vega": float(vega), "rho": float(rho)}

# ---------------------------
# Prediction wrapper (Hybrid)
# ---------------------------
def predict_index(index_name: str) -> dict:
    """
    Returns a dict: spot, pred_price, pred_return, low, high, confidence, method, prob_up
    """
    try:
        df = fetch_historical(index_name, days=120)
        spot = fetch_spot(index_name)
        fii, dii, sentiment = fetch_fii_dii_and_sentiment()
        bias = (dii - fii) / 1000.0 + sentiment * 5.0

        # Statistical fallback
        last_close = float(df["Close"].iloc[-1])
        vol = float(df["Close"].pct_change().rolling(10).std().iloc[-1] or 0.0)
        pred_fallback = last_close * (1 + 0.001)  # small move
        low_fb = last_close * (1 - 0.01 - vol)
        high_fb = last_close * (1 + 0.01 + vol)
        conf_fb = max(0.05, 1 - abs(vol)*5)

        # Try ML
        model = load_model(index_name)
        if model is None:
            raise RuntimeError("No model")
        regr = model["regr"]; clf = model["clf"]; scaler = model["scaler"]; features = model["features"]
        df_feat = compute_features(df)
        last = df_feat.iloc[-1]
        x = np.array([last[f] for f in features]).reshape(1, -1)
        xs = scaler.transform(x)
        pred_price = float(regr.predict(xs)[0])
        pred_return = (pred_price - last_close) / last_close
        prob_up = float(clf.predict_proba(xs)[0][1])  # probability of up
        vol_recent = float(df_feat["ret1"].rolling(10).std().iloc[-1] or 0.0)
        low = pred_price * (1 - 0.01 - vol_recent)
        high = pred_price * (1 + 0.01 + vol_recent)
        confidence = max(0.05, 1.0 - abs(pred_return) / 0.1)
        return {
            "spot": float(spot),
            "pred_price": float(pred_price),
            "pred_return": float(pred_return),
            "low": float(low),
            "high": float(high),
            "confidence": float(confidence),
            "prob_up": float(prob_up),
            "method": "ml",
            "sentiment": sentiment,
            "bias": bias
        }
    except Exception as e:
        logger.warning("ML predict failed for %s: %s", index_name, e)
        # fallback
        df = fetch_historical(index_name, days=30)
        last_close = float(df["Close"].iloc[-1])
        vol = float(df["Close"].pct_change().rolling(10).std().iloc[-1] or 0.0)
        pred_return = (0.001)
        pred_price = last_close * (1 + pred_return)
        low = last_close * (1 - 0.01 - vol)
        high = last_close * (1 + 0.01 + vol)
        return {
            "spot": float(fetch_spot(index_name)),
            "pred_price": float(pred_price),
            "pred_return": float(pred_return),
            "low": float(low),
            "high": float(high),
            "confidence": float(max(0.05, 1 - vol*5)),
            "prob_up": float(0.5),
            "method": "fallback",
            "sentiment": 0.0,
            "bias": 0.0
        }

# ---------------------------
# Persist position & helpers
# ---------------------------
def ensure_user(tg_id:int):
    cur = DB.cursor()
    cur.execute("SELECT tg_id FROM users WHERE tg_id = ?", (tg_id,))
    if not cur.fetchone():
        cur.execute("INSERT INTO users (tg_id,tier,created_at) VALUES (?,?,?)", (tg_id,"free",datetime.now().isoformat()))
        DB.commit()

def save_position_db(parsed:dict):
    cur = DB.cursor()
    cur.execute("""INSERT INTO positions (tg_id,action,index_symbol,month,strike,opt_type,price,qty,raw_text,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (parsed.get("user_id"), parsed.get("action"), parsed.get("index_symbol"), parsed.get("month"),
                 parsed.get("strike"), parsed.get("opt_type"), parsed.get("price"), parsed.get("qty"),
                 parsed.get("raw_text"), datetime.now().isoformat()))
    DB.commit()

def get_positions_for_user(tg_id:int, limit=20):
    cur = DB.cursor()
    cur.execute("SELECT action,index_symbol,month,strike,opt_type,price,qty,raw_text,created_at FROM positions WHERE tg_id=? ORDER BY id DESC LIMIT ?", (tg_id,limit))
    rows = cur.fetchall()
    cols = ["action","index_symbol","month","strike","opt_type","price","qty","raw_text","created_at"]
    return [dict(zip(cols,row)) for row in rows]

# ---------------------------
# Position parser (flexible)
# ---------------------------
def parse_position_text(text:str,user_id:int) -> dict:
    raw = text.strip()
    parts = raw.upper().replace(","," ").replace("/"," ").split()
    parsed = {"raw_text": raw, "user_id": user_id}
    # action
    parsed["action"] = parts[0] if parts else "BUY"
    # index detection
    parsed["index_symbol"] = detect_index_from_text(raw)
    # month detection (3-letter)
    months = set(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])
    parsed["month"] = ""
    for p in parts:
        if p in months:
            parsed["month"] = p; break
    # strike detection (first integer > 1000)
    strike = 0
    for p in parts:
        if p.isdigit() and int(p) > 1000:
            strike = int(p); break
    parsed["strike"] = strike
    parsed["opt_type"] = "CE" if "CE" in parts else ("PE" if "PE" in parts else "")
    # price detection (after @ or last small number)
    price = None
    if "@" in parts:
        try:
            idx = parts.index("@")
            if idx+1 < len(parts):
                price = float(parts[idx+1])
        except:
            price = None
    if price is None:
        # pick last numeric less than 10000
        nums = [p for p in parts if p.replace(".","",1).isdigit()]
        for n in reversed(nums):
            val = float(n)
            if val < 10000:
                price = val; break
    parsed["price"] = float(price) if price is not None else 0.0
    # qty detection
    qty = 1
    if "LOT" in raw.upper():
        toks = raw.upper().split()
        for i,tok in enumerate(toks):
            if "LOT" in tok and i>0:
                try: qty = int(toks[i-1])
                except: qty = 1
                break
        # standard lot size for BANKNIFTY=15? (depends on series) — we will keep qty as lots count
    else:
        ints = [int(p) for p in parts if p.isdigit()]
        if ints:
            qty = ints[-1]
    parsed["qty"] = int(qty)
    return parsed

# ---------------------------
# Telegram Handlers
# ---------------------------
def start_handler(update:Update, context:CallbackContext):
    user = update.effective_user
    ensure_user(user.id)
    kb = [[InlineKeyboardButton("Free", callback_data="tier_free"),
           InlineKeyboardButton("Gold", callback_data="tier_gold")]]
    update.message.reply_text("Welcome to GrokNifty — send a position or /predict for current prediction.", reply_markup=InlineKeyboardMarkup(kb))

def tier_callback(update:Update, context:CallbackContext):
    q = update.callback_query
    tg = q.from_user.id
    tier = q.data.split("_",1)[1] if "_" in q.data else "free"
    cur = DB.cursor()
    cur.execute("UPDATE users SET tier=? WHERE tg_id=?", (tier, tg))
    DB.commit()
    q.answer(text=f"Tier set to {tier}")
    q.edit_message_text(text=f"Tier set to {tier}")

def position_handler(update:Update, context:CallbackContext):
    user = update.effective_user
    ensure_user(user.id)
    text = update.message.text
    try:
        parsed = parse_position_text(text, user.id)
        save_position_db(parsed)
        update.message.reply_text("Position saved ✅\n" + str(parsed))
    except Exception as e:
        logger.exception("parse failed: %s", e)
        update.message.reply_text("Failed to parse. Try: BUY BANKNIFTY NOV 44500 CE @ 215 3 lots")

def predict_handler(update:Update, context:CallbackContext):
    user = update.effective_user
    ensure_user(user.id)
    # if args provided, respect them
    args = context.args or []
    if args:
        idx = detect_index_from_text(" ".join(args))
    else:
        # use last position if exists
        pos = get_positions_for_user(user.id, limit=1)
        idx = pos[0]["index_symbol"] if pos else "BANKNIFTY"
    # get both predictions
    try:
        out_nifty = predict_index("NIFTY")
        out_bank = predict_index("BANKNIFTY")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")
        text = f"*GrokNifty Combined Prediction* ({timestamp})\n\n"
        text += f"*NIFTY* — Spot {out_nifty['spot']:.2f} | Pred {out_nifty['pred_price']:.2f} ({out_nifty['pred_return']*100:.2f}%) | ProbUp {out_nifty['prob_up']*100:.1f}% | Range {out_nifty['low']:.0f}–{out_nifty['high']:.0f}\n\n"
        text += f"*BANKNIFTY* — Spot {out_bank['spot']:.2f} | Pred {out_bank['pred_price']:.2f} ({out_bank['pred_return']*100:.2f}%) | ProbUp {out_bank['prob_up']*100:.1f}% | Range {out_bank['low']:.0f}–{out_bank['high']:.0f}\n\n"
        # include user's last few positions
        positions = get_positions_for_user(user.id, limit=5)
        if positions:
            text += "*Your recent positions:*\n"
            for p in positions:
                text += f"• {p['action']} {p['index_symbol']} {p['month']} {p['strike']}{p['opt_type']} @ {p['price']} qty:{p['qty']}\n"
        update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        # save predictions in DB (for history)
        cur = DB.cursor()
        cur.execute("""INSERT INTO predictions (tg_id,index_symbol,spot,pred_price,pred_return,low,high,confidence,method,created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (user.id,"NIFTY", out_nifty['spot'], out_nifty['pred_price'], out_nifty['pred_return'], out_nifty['low'], out_nifty['high'], out_nifty['confidence'], out_nifty['method'], datetime.now().isoformat()))
        cur.execute("""INSERT INTO predictions (tg_id,index_symbol,spot,pred_price,pred_return,low,high,confidence,method,created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (user.id,"BANKNIFTY", out_bank['spot'], out_bank['pred_price'], out_bank['pred_return'], out_bank['low'], out_bank['high'], out_bank['confidence'], out_bank['method'], datetime.now().isoformat()))
        DB.commit()
    except Exception as e:
        logger.exception("predict error: %s", e)
        update.message.reply_text("Prediction failed. Try again later.")

def settle_tier_cmd(update:Update, context:CallbackContext):
    user = update.effective_user
    if user.id not in ADMIN_IDS:
        update.message.reply_text("Not authorized.")
        return
    args = context.args or []
    if len(args) < 2:
        update.message.reply_text("Usage: /settier <tg_id> <tier>")
        return
    try:
        tg = int(args[0]); tier = args[1]
        cur = DB.cursor()
        cur.execute("UPDATE users SET tier=? WHERE tg_id=?", (tier,tg))
        if cur.rowcount == 0:
            cur.execute("INSERT INTO users (tg_id,tier,created_at) VALUES (?,?,?)", (tg,tier,datetime.now().isoformat()))
        DB.commit()
        update.message.reply_text(f"Set tier {tier} for {tg}")
    except Exception as e:
        update.message.reply_text("Failed to settier: " + str(e))

def admin_train_cmd(update:Update, context:CallbackContext):
    user = update.effective_user
    if user.id not in ADMIN_IDS:
        update.message.reply_text("Not authorized.")
        return
    update.message.reply_text("Training both indexes in background...")
    scheduler.add_job(lambda: train_models("BANKNIFTY"), id=f"train_bn_{int(time.time())}")
    scheduler.add_job(lambda: train_models("NIFTY"), id=f"train_nifty_{int(time.time())}")

def admin_stats_cmd(update:Update, context:CallbackContext):
    user = update.effective_user
    if user.id not in ADMIN_IDS:
        update.message.reply_text("Not authorized.")
        return
    cur = DB.cursor()
    cur.execute("SELECT index_symbol, path, trained_at, mae, accuracy FROM models ORDER BY id DESC LIMIT 10")
    rows = cur.fetchall()
    text = "Recent models:\n"
    for r in rows:
        text += f"{r[0]} trained_at={r[2]} mae={r[3]} acc={r[4]}\n"
    update.message.reply_text(text)

# ---------------------------
# Scheduler tasks
# ---------------------------
def daily_train_job():
    logger.info("Scheduled daily training started")
    train_models("BANKNIFTY")
    train_models("NIFTY")
    logger.info("Scheduled daily training finished")

def auto_push_job():
    # send short prediction to all users (careful with rate limits)
    logger.info("Auto push job started")
    cur = DB.cursor()
    cur.execute("SELECT tg_id FROM users")
    rows = cur.fetchall()
    for (tg,) in rows:
        try:
            out_bank = predict_index("BANKNIFTY")
            text = f"Auto Update — BANKNIFTY Spot {out_bank['spot']:.0f} | Pred {out_bank['pred_price']:.0f} | ProbUp {out_bank['prob_up']*100:.1f}%"
            bot.send_message(chat_id=tg, text=text)
            time.sleep(0.1)
        except Exception as e:
            logger.debug("Auto push to %s failed: %s", tg, e)
    logger.info("Auto push job finished")

# ---------------------------
# Flask Admin Dashboard (simple)
# ---------------------------
DASH_TEMPLATE = """
<!doctype html>
<title>GrokNifty Admin</title>
<h1>GrokNifty Admin Dashboard</h1>
<p><a href="/models">Models</a> | <a href="/users">Users</a> | <a href="/positions">Positions</a> | <a href="/charts/nifty">Nifty Chart</a> | <a href="/charts/banknifty">BankNifty Chart</a></p>
"""

@app.route("/")
def dash_home():
    return render_template_string(DASH_TEMPLATE)

@app.route("/models")
def dash_models():
    cur = DB.cursor(); cur.execute("SELECT index_symbol,path,trained_at,mae,accuracy FROM models ORDER BY id DESC LIMIT 20")
    m = cur.fetchall()
    return jsonify([{"index":r[0],"path":r[1],"trained_at":r[2],"mae":r[3],"accuracy":r[4]} for r in m])

@app.route("/users")
def dash_users():
    cur = DB.cursor(); cur.execute("SELECT tg_id,tier,created_at FROM users")
    rows = cur.fetchall()
    return jsonify([{"tg_id":r[0],"tier":r[1],"created_at":r[2]} for r in rows])

@app.route("/positions")
def dash_positions():
    cur = DB.cursor(); cur.execute("SELECT tg_id,raw_text,created_at FROM positions ORDER BY id DESC LIMIT 200")
    rows = cur.fetchall()
    return jsonify([{"tg_id":r[0],"raw":r[1],"created_at":r[2]} for r in rows])

@app.route("/charts/<idx>")
def chart(idx):
    idx = idx.upper()
    if idx not in ("NIFTY","BANKNIFTY"):
        abort(404)
    img_path = os.path.join(CHARTS_DIR, f"{idx}.png")
    if not os.path.exists(img_path):
        # create chart
        try:
            df = fetch_historical(idx, days=120)
            plt.figure(figsize=(10,4))
            sns.lineplot(x=df.index, y=df["Close"])
            plt.title(f"{idx} Close - last 120 days")
            plt.tight_layout()
            plt.savefig(img_path)
            plt.close()
        except Exception as e:
            logger.exception("Chart generation failed: %s", e)
            abort(500)
    return send_file(img_path, mimetype="image/png")

@app.route("/train-now", methods=["POST"])
def dash_train_now():
    key = request.form.get("key")
    # simple security: require ADMIN_IDS string in form key or some secret in env in production
    if not key or key != os.environ.get("ADMIN_KEY", ""):
        return "Not authorized", 403
    scheduler.add_job(lambda: train_models("BANKNIFTY"), id=f"dash_train_bn_{int(time.time())}")
    scheduler.add_job(lambda: train_models("NIFTY"), id=f"dash_train_nifty_{int(time.time())}")
    return "Training queued", 200

# ---------------------------
# Main / startup
# ---------------------------
def main():
    global bot
    # Telegram
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start_handler))
    dp.add_handler(CallbackQueryHandler(tier_callback, pattern=r"^tier_"))
    dp.add_handler(MessageHandler(Filters.text & (~Filters.command), position_handler))
    dp.add_handler(CommandHandler("predict", predict_handler))
    dp.add_handler(CommandHandler("settier", settle_tier_cmd))
    dp.add_handler(CommandHandler("trainnow", admin_train_cmd))
    dp.add_handler(CommandHandler("adminstats", admin_stats_cmd))
    # start polling
    updater.start_polling()
    bot = updater.bot
    logger.info("Telegram bot started polling")

    # scheduler jobs
    try:
        # daily training at 02:00 IST
        scheduler.add_job(daily_train_job, "cron", hour=2, minute=0, id="daily_train")
        # auto push morning
        scheduler.add_job(auto_push_job, "cron", hour=9, minute=10, id="auto_push_morning")
        scheduler.start()
        logger.info("Scheduler started")
    except Exception as e:
        logger.exception("Scheduler start failed: %s", e)

    # warm-start training if no model present
    for idx in ("BANKNIFTY","NIFTY"):
        if load_model(idx) is None:
            logger.info("No model for %s: scheduling immediate training", idx)
            scheduler.add_job(lambda i=idx: train_models(i), id=f"startup_train_{idx}")

    # start flask in background process via scheduler job to avoid blocking (single-threaded dev use)
    def run_flask():
        logger.info("Starting Flask admin on %s:%s", FLASK_HOST, FLASK_PORT)
        app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True)

    scheduler.add_job(run_flask, 'date', run_date=datetime.now() + timedelta(seconds=1), id='start_flask')

    updater.idle()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Fatal exception: %s", e)
        raise