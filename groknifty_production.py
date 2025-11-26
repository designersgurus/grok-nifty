#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import logging
import requests
import sqlite3
import traceback
from datetime import datetime
from bs4 import BeautifulSoup

import yfinance as yf
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from apscheduler.schedulers.background import BackgroundScheduler
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    Filters,
    CallbackContext,
)

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    filename="groknifty.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -------------------------
# ENV
# -------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_IDS = ["9898989898"]  # <---- Replace with your Telegram User ID

if not TELEGRAM_TOKEN:
    print("ERROR: TELEGRAM_TOKEN not set")
    exit(1)

# -------------------------
# DB
# -------------------------
DB_PATH = "grok.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            tier TEXT DEFAULT 'free',
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def set_user_tier(user_id, tier):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO users(user_id, tier) VALUES (?,?)", (user_id, tier))
    conn.commit()
    conn.close()

def get_user_tier(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT tier FROM users WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else "free"

# -------------------------
# Sentiment
# -------------------------
sia = SentimentIntensityAnalyzer()

# -------------------------
# Data Fetching
# -------------------------
def fetch_pre_market(index="BANKNIFTY"):

    if index == "NIFTY":
        symbol = "^NSEI"
    else:
        symbol = "^NSEBANK"

    try:
        data = yf.download(symbol, period="1d")
        spot = float(data["Close"].iloc[-1]) if not data.empty else 20000
    except:
        spot = 20000

    try:
        r = requests.get("https://www.moneycontrol.com/news/business/markets/")
        soup = BeautifulSoup(r.text, "html.parser")
        headlines = [h.text.strip() for h in soup.find_all("h2")[:10]]
        sentiment = sum(sia.polarity_scores(h)["compound"] for h in headlines) / len(headlines)
    except:
        sentiment = 0

    try:
        fii, dii = 0, 0
        r = requests.get("https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.html")
        soup = BeautifulSoup(r.text, "html.parser")
        nums = soup.find_all("span", {"class": "stprc"})
        if len(nums) >= 2:
            fii = float(nums[0].text.replace(",", ""))
            dii = float(nums[1].text.replace(",", ""))
    except:
        fii, dii = 0, 0

    return spot, fii, dii, sentiment

# -------------------------
# Prediction Logic
# -------------------------
def predict_range(index="BANKNIFTY", user_positions=None):
    spot, fii, dii, sentiment = fetch_pre_market(index)
    
    bias = (dii - fii) / 1000 + (sentiment * 10)

    if index == "BANKNIFTY":
        base = 250
    else:
        base = 120

    low = spot - base + (bias * 40)
    high = spot + base + (bias * 40)

    if user_positions:
        for pos in user_positions:
            if pos["action"] == "BUY":
                high += 20
            else:
                low -= 20

    return {
        "spot": round(spot, 2),
        "low": round(low, 2),
        "high": round(high, 2),
        "fii": fii,
        "dii": dii,
        "sentiment": sentiment
    }

# -------------------------
# Telegram Bot Core
# -------------------------
def send_tier_menu(update, context):
    keyboard = [
        [InlineKeyboardButton("Free", callback_data="tier_free")],
        [InlineKeyboardButton("Copper", callback_data="tier_copper")],
        [InlineKeyboardButton("Silver", callback_data="tier_silver")],
        [InlineKeyboardButton("Gold", callback_data="tier_gold")],
        [InlineKeyboardButton("Diamond", callback_data="tier_diamond")],
    ]
    update.message.reply_text(
        "Choose your subscription tier:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

def start(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    set_user_tier(user_id, get_user_tier(user_id))

    send_tier_menu(update, context)

def tier_callback(update: Update, context: CallbackContext):
    query = update.callback_query
    user_id = str(query.from_user.id)
    tier = query.data.split("_")[1]

    set_user_tier(user_id, tier)

    query.answer("Tier updated to " + tier.capitalize())

def admin_set_tier(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    if user_id not in ADMIN_IDS:
        update.message.reply_text("Not authorized.")
        return

    try:
        target = context.args[0]
        tier = context.args[1]
        set_user_tier(target, tier)
        update.message.reply_text(f"Updated {target} to {tier}")
    except:
        update.message.reply_text("Use: /settier <user_id> <tier>")

# -------------------------
# User Message Parsing (Positions)
# -------------------------
user_positions = {}

def parse_position(text):
    try:
        words = text.upper().split()
        action = words[0]  # BUY/SELL
        index = words[1]   # NIFTY/BANKNIFTY
        strike = int(words[2])
        opt_type = words[3]  # CE/PE
        price = float(words[words.index("@") + 1])
        return {
            "action": action,
            "index": index,
            "strike": strike,
            "type": opt_type,
            "price": price
        }
    except:
        return None

def handle_message(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    text = update.message.text.strip()

    pos = parse_position(text)
    if pos:
        if user_id not in user_positions:
            user_positions[user_id] = []
        user_positions[user_id].append(pos)
        update.message.reply_text(f"Position added: {pos}")
        return

    update.message.reply_text("Invalid input. Example:\nBUY BANKNIFTY 45200 CE @ 215")

# -------------------------
# Prediction Command
# -------------------------
def predict(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    tier = get_user_tier(user_id)

    if tier == "free":
        update.message.reply_text("Upgrade your tier to access predictions.")
        return

    p1 = predict_range("NIFTY", user_positions.get(user_id))
    p2 = predict_range("BANKNIFTY", user_positions.get(user_id))

    msg = f"""
📊 **GrokNifty Prediction**

**NIFTY**
Spot: {p1['spot']}
Range: {p1['low']} → {p1['high']}

**BANKNIFTY**
Spot: {p2['spot']}
Range: {p2['low']} → {p2['high']}

FII: {p1['fii']}
DII: {p1['dii']}
Sentiment: {round(p1['sentiment'], 3)}
"""
    update.message.reply_text(msg, parse_mode="Markdown")

# -------------------------
# Scheduler
# -------------------------
scheduler = BackgroundScheduler()

def auto_push(context):
    for user_id in list(user_positions.keys()):
        try:
            p1 = predict_range("NIFTY", user_positions.get(user_id))
            p2 = predict_range("BANKNIFTY", user_positions.get(user_id))

            text = f"""
📈 **Auto Update**

NIFTY → {p1['low']} - {p1['high']}
BANKNIFTY → {p2['low']} - {p2['high']}
"""
            context.bot.send_message(chat_id=user_id, text=text, parse_mode="Markdown")
        except:
            logging.error(traceback.format_exc())

def start_scheduler(updater):
    scheduler.add_job(auto_push, "cron", hour=9, minute=15, args=[updater.bot])
    scheduler.add_job(auto_push, "cron", hour=12, minute=0, args=[updater.bot])
    scheduler.start()

# -------------------------
# Main
# -------------------------
def main():
    init_db()

    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(tier_callback))
    dp.add_handler(CommandHandler("predict", predict))
    dp.add_handler(CommandHandler("settier", admin_set_tier))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    start_scheduler(updater)

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
