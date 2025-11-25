import os
import time
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

# ML
from sklearn.linear_model import LinearRegression
import joblib

# Telegram Bot
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackQueryHandler,
)
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# Sentiment
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# -------------------------------------------------------------------
# ENV
# -------------------------------------------------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

NEWS_URL = "https://www.moneycontrol.com/news/business/markets/"
FII_DII_URL = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.html"

users = {}  # Simple in-memory database

# Global updater so scheduler can use it
updater = None


# -------------------------------------------------------------------
# Data Fetching
# -------------------------------------------------------------------
def fetch_pre_market():
    try:
        gift = yf.download("^NSEBANK", period="1d")
        spot = float(gift["Close"].iloc[-1]) if not gift.empty else 45000.0
    except:
        spot = 45000.0

    try:
        resp = requests.get(FII_DII_URL, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        fii = soup.select_one(".fii span.value")
        dii = soup.select_one(".dii span.value")
        fii_val = float(fii.text.replace(",", "")) if fii else 0
        dii_val = float(dii.text.replace(",", "")) if dii else 0
    except:
        fii_val, dii_val = 0, 0

    try:
        resp = requests.get(NEWS_URL, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        headlines = [h.text.strip() for h in soup.find_all("h2", limit=10)]
        sentiment = (
            sum(sia.polarity_scores(h)["compound"] for h in headlines) / len(headlines)
            if headlines
            else 0
        )
    except:
        sentiment = 0

    return spot, fii_val, dii_val, sentiment


# -------------------------------------------------------------------
# Prediction Model (Simple ML)
# -------------------------------------------------------------------
def predict_range(bias, spot):
    low = spot - 200 + (bias * 50)
    high = spot + 200 + (bias * 50)
    return low, high


# -------------------------------------------------------------------
# Telegram Bot Logic
# -------------------------------------------------------------------
def start(update, context):
    user_id = update.message.from_user.id
    if user_id not in users:
        users[user_id] = {
            "tier": "free",
            "positions": [],
            "credits": 0,
        }

    update.message.reply_text(
        "Welcome to *GrokNifty AI Bot* 👋\n\nSelect your plan:",
        reply_markup=get_tier_menu(),
        parse_mode="Markdown",
    )


def get_tier_menu():
    keyboard = [
        [InlineKeyboardButton("Free", callback_data="tier_free")],
        [InlineKeyboardButton("Copper", callback_data="tier_copper")],
        [InlineKeyboardButton("Silver", callback_data="tier_silver")],
        [InlineKeyboardButton("Gold", callback_data="tier_gold")],
        [InlineKeyboardButton("Diamond", callback_data="tier_diamond")],
    ]
    return InlineKeyboardMarkup(keyboard)


def tier_callback(update, context):
    query = update.callback_query
    tier = query.data.split("_")[1]
    users[query.from_user.id]["tier"] = tier
    query.answer(text=f"Tier selected: {tier.capitalize()}")


def manual_predict(update, context):
    spot, fii, dii, sentiment = fetch_pre_market()
    bias = (dii - fii) / 1000 + sentiment * 5
    low, high = predict_range(bias, spot)

    update.message.reply_text(
        f"📊 *Manual Prediction*\n\n"
        f"Spot: {spot}\n"
        f"FII: {fii}\nDII: {dii}\n"
        f"Sentiment: {sentiment}\n\n"
        f"🔮 *Range*: {low} - {high}",
        parse_mode="Markdown",
    )


# -------------------------------------------------------------------
# Auto Scheduler
# -------------------------------------------------------------------
scheduler = BackgroundScheduler(timezone="Asia/Kolkata")


def auto_push_predictions():
    global updater
    bot = updater.bot

    for user_id, data in users.items():
        spot, fii, dii, sentiment = fetch_pre_market()
        bias = (dii - fii) / 1000 + sentiment * 5
        low, high = predict_range(bias, spot)

        bot.send_message(
            chat_id=user_id,
            text=f"📈 *Auto Update*\nNifty Range: {low} - {high}",
            parse_mode="Markdown",
        )


# -------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------
def main():
    global updater

    if not TELEGRAM_TOKEN:
        print("❌ ERROR: TELEGRAM_TOKEN not found! Set it in /etc/groknifty/groknifty.env")
        return

    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("predict", manual_predict))
    dp.add_handler(CallbackQueryHandler(tier_callback))

    # Scheduled tasks
    scheduler.add_job(auto_push_predictions, "cron", hour="9", minute="10")
    scheduler.add_job(auto_push_predictions, "cron", hour="12", minute="00")
    scheduler.start()

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
