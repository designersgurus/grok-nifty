import os
import time
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

# ML & Sentiment
from sklearn.linear_model import LinearRegression
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# Telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# -------------------------------------------------------------------
# ENV & DB
# -------------------------------------------------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
users = {}  # {user_id: {tier, positions[], predictions_today, positions_month}}
updater = None

# Tier limits
TIER_LIMITS = {
    "free":    {"pred/day": 2,  "pos/month": 0},
    "copper":  {"pred/day": 5,  "pos/month": 5},
    "silver":  {"pred/day": 12, "pos/month": 12},
    "gold":    {"pred/day": 999,"pos/month": 999},
    "diamond": {"pred/day": 999,"pos/month": 999},
}

# -------------------------------------------------------------------
# Data Fetching
# -------------------------------------------------------------------
def fetch_live_spot(index: str):
    ticker = "^NSEI" if index == "NIFTY" else "^NSEBANK"
    try:
        data = yf.download(ticker, period="1d", progress=False)
        return round(float(data["Close"].iloc[-1]), 2)
    except:
        return 24800.0 if index == "NIFTY" else 59000.0

def fetch_pre_market():
    spot = fetch_live_spot("BANKNIFTY")
    try:
        resp = requests.get("https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.html", timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        fii = float(soup.select_one(".fii span.value").text.replace(",", "")) if soup.select_one(".fii span.value") else 0
        dii = float(soup.select_one(".dii span.value").text.replace(",", "")) if soup.select_one(".dii span.value") else 0
    except:
        fii, dii = 0, 0

    try:
        resp = requests.get("https://www.moneycontrol.com/news/business/markets/", timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        headlines = [h.text.strip() for h in soup.find_all("h2", limit=10)]
        sentiment = sum(sia.polarity_scores(h)["compound"] for h in headlines) / len(headlines) if headlines else 0
    except:
        sentiment = 0

    return spot, fii, dii, sentiment

# -------------------------------------------------------------------
# Prediction Engine
# -------------------------------------------------------------------
def get_prediction(spot, fii, dii, sentiment):
    bias = (dii - fii) / 1000 + sentiment * 5
    low = spot - 200 + (bias * 50)
    high = spot + 200 + (bias * 50)
    return round(low), round(high), round(bias, 2)

# -------------------------------------------------------------------
# Telegram Commands
# -------------------------------------------------------------------
def start(update, context):
    user_id = update.message.from_user.id
    if user_id not in users:
        users[user_id] = {
            "tier": "free",
            "positions": [],
            "predictions_today": 0,
            "positions_month": 0,
        }
    update.message.reply_text(
        "Welcome to *GrokNifty Pro* \n\nChoose your plan:",
        reply_markup=get_tier_menu(),
        parse_mode="Markdown"
    )

def get_tier_menu():
    keyboard = [
        [InlineKeyboardButton("Free", callback_data="tier_free")],
        [InlineKeyboardButton("Copper ₹999", callback_data="tier_copper")],
        [InlineKeyboardButton("Silver ₹2999", callback_data="tier_silver")],
        [InlineKeyboardButton("Gold ₹5999", callback_data="tier_gold")],
        [InlineKeyboardButton("Diamond ₹9999", callback_data="tier_diamond")],
    ]
    return InlineKeyboardMarkup(keyboard)

def tier_callback(update, context):
    query = update.callback_query
    tier = query.data.split("_")[1]
    users[query.from_user.id]["tier"] = tier
    query.answer(f"{tier.capitalize()} Activated!")

def manual_predict(update, context):
    user_id = update.message.from_user.id
    tier = users[user_id]["tier"]
    if users[user_id]["predictions_today"] >= TIER_LIMITS[tier]["pred/day"]:
        update.message.reply_text("Daily prediction limit reached!")
        return

    spot, fii, dii, sentiment = fetch_pre_market()
    low, high, bias = get_prediction(spot, fii, dii, sentiment)
    users[user_id]["predictions_today"] += 1

    update.message.reply_text(
        f"*Live Prediction*\n\n"
        f"Spot: {spot:,.0f}\n"
        f"FII: ₹{fii:,.0f}cr | DII: ₹{dii:,.0f}cr\n"
        f"Sentiment: {sentiment:+.2f} | Bias: {bias:+.1f}\n\n"
        f"*Range*: {low:,.0f} - {high:,.0f}",
        parse_mode="Markdown"
    )

# -------------------------------------------------------------------
# POSITION INPUT – MAIN FEATURE YOU WANTED
# -------------------------------------------------------------------
def position_input(update, context):
    user_id = update.message.from_user.id
    tier = users[user_id]["tier"]

    if TIER_LIMITS[tier]["pos/month"] == 0:
        update.message.reply_text("Your tier doesn't allow position tracking.")
        return
    if users[user_id]["positions_month"] >= TIER_LIMITS[tier]["pos/month"]:
        update.message.reply_text("Monthly position limit reached!")
        return

    text = update.message.text.upper().strip()
    words = text.split()

    try:
        action = words[0]           # BUY / SELL
        index = words[1]            # NIFTY / BANKNIFTY
        strike = int(words[3])
        opt_type = words[4]         # CE / PE
        price = float(words[words.index("@") + 1])

        qty = 35
        if "QTY" in text or len(words) > words.index("@") + 2:
            for w in words[words.index("@")+2:]:
                if w.isdigit():
                    qty = int(w)
                    break
            if index == "NIFTY":
                qty = (qty // 25) * 25

        # Live spot
        spot = fetch_live_spot(index)
        distance = spot - strike if opt_type == "CE" else strike - spot

        # Bias
        _, fii, dii, sentiment = fetch_pre_market()
        bias = (dii - fii)/1000 + sentiment*5
        bias_text = "Bullish" if bias > 0 else "Bearish"

        # Target & SL
        if action == "BUY":
            target = price * 1.8
            sl = price * 0.6
        else:
            target = price * 0.4
            sl = price * 1.6

        # Store
        position = {
            "index": index,
            "action": action,
            "strike": strike,
            "type": opt_type,
            "ltp": price,
            "qty": qty,
            "spot": spot,
            "distance": distance,
            "target": target,
            "sl": sl,
            "time": datetime.now().strftime("%H:%M")
        }
        users[user_id]["positions"].append(position)
        users[user_id]["positions_month"] += 1

        update.message.reply_text(
            f"*Position Recorded!*\n\n"
            f"{action} {index} {strike} {opt_type} × {qty//35} lot\n"
            f"Entry: ₹{price} | Spot: {spot:,.0f}\n"
            f"Distance: {distance:+.0f} pts | Bias: {bias_text} ({bias:+.1f})\n\n"
            f"Target: ₹{target:.0f} | Stop-Loss: ₹{sl:.0f}\n"
            f"Status: {'Strong setup' if abs(distance) < 400 else 'Needs momentum'}",
            parse_mode="Markdown"
        )

    except Exception as e:
        update.message.reply_text(
            "Wrong format!\n\n"
            "Correct examples:\n"
            "BUY BANKNIFTY NOV 59500 CE @ 215\n"
            "SELL NIFTY 24800 PE @ 180 75\n"
            "BUY NIFTY 25000 CE @ 320"
        )

# -------------------------------------------------------------------
# Auto Push (APScheduler)
# -------------------------------------------------------------------
scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
def auto_push():
    if not updater:
        return
    for user_id, data in users.items():
        tier = data["tier"]
        if data["predictions_today"] >= TIER_LIMITS[tier]["pred/day"]:
            continue
        spot, fii, dii, sentiment = fetch_pre_market()
        low, high, bias = get_prediction(spot, fii, dii, sentiment)
        data["predictions_today"] += 1
        updater.bot.send_message(
            chat_id=user_id,
            text=f"*Auto Update*\nSpot: {spot:,.0f}\nRange: {low:,.0f} - {high:,.0f}\nBias: {bias:+.1f}",
            parse_mode="Markdown"
        )

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    global updater
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN not set!")
        return

    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("predict", manual_predict))
    dp.add_handler(CallbackQueryHandler(tier_callback))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, position_input))

    # Schedule auto updates
    scheduler.add_job(auto_push, "cron", hour=9, minute=10)
    scheduler.add_job(auto_push, "cron", hour=12, minute=0)
    scheduler.add_job(auto_push, "cron", hour=15, minute=0)
    scheduler.start()

    print("GrokNifty Pro is LIVE on Oracle Cloud!")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
