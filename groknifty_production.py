# ============================
#   GROKNIFTY PRODUCTION BOT
#   Option C – Full Dynamic Bot
#   With Tiers, Admin Controls,
#   NIFTY + BANKNIFTY Predictions
#   pytz Scheduler (APScheduler Safe)
# ============================

import os
import logging
import pytz
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

import requests
import yfinance as yf
from bs4 import BeautifulSoup
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon")

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Updater,
    CommandHandler,
    CallbackContext,
    MessageHandler,
    Filters,
    CallbackQueryHandler,
)

# ============ LOGGER =============

logging.basicConfig(
    filename="groknifty.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# ============ READ ENV =============

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    print("ERROR: TELEGRAM_TOKEN missing")
    exit()

ADMIN_USER_ID = 123456789   # <<< REPLACE WITH YOUR USER ID

TZ = pytz.timezone("Asia/Kolkata")

# ============ GLOBAL DB =============

users = {}  # {user_id: {"tier":..., "positions": []}}
sia = SentimentIntensityAnalyzer()

NEWS_URL = "https://www.moneycontrol.com/news/business/markets/"
FII_DII_URL = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.html"

# -------------------------------------------
# FETCH SENTIMENT + FII DII + PRICES
# -------------------------------------------

def fetch_pre_market(index="NIFTY"):
    try:
        symbol = "^NSEI" if index == "NIFTY" else "^NSEBANK"
        df = yf.download(symbol, period="1d")
        spot = float(df["Close"].iloc[-1]) if not df.empty else 0
    except:
        spot = 0

    try:
        resp = requests.get(FII_DII_URL, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        fii = soup.select_one(".fii span.value")
        dii = soup.select_one(".dii span.value")
        fii_val = float(fii.text.replace(",", "")) if fii else 0
        dii_val = float(dii.text.replace(",", "")) if dii else 0
    except:
        fii_val = 0
        dii_val = 0

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

    bias = (dii_val - fii_val) / 1000 + sentiment * 5
    return spot, fii_val, dii_val, sentiment, bias


def predict_range(spot, bias):
    low = spot - 200 + (bias * 50)
    high = spot + 200 + (bias * 50)
    return low, high


# -------------------------------------------
# TIER SYSTEM
# -------------------------------------------

TIERS = ["free", "silver", "gold", "diamond"]

def tier_menu():
    kb = [
        [InlineKeyboardButton("Free", callback_data="tier_free")],
        [InlineKeyboardButton("Silver", callback_data="tier_silver")],
        [InlineKeyboardButton("Gold", callback_data="tier_gold")],
        [InlineKeyboardButton("Diamond", callback_data="tier_diamond")],
    ]
    return InlineKeyboardMarkup(kb)


def start(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    if user_id not in users:
        users[user_id] = {"tier": "free", "positions": []}

    update.message.reply_text(
        "👋 Welcome to GrokNifty AI Bot!\nChoose your tier:",
        reply_markup=tier_menu(),
    )


def tier_callback(update: Update, context: CallbackContext):
    query = update.callback_query
    tier = query.data.split("_")[1]
    user_id = query.from_user.id

    users[user_id]["tier"] = tier
    query.answer(f"Tier selected: {tier.capitalize()}")
    query.edit_message_text(f"✅ Tier Updated: {tier.capitalize()}")


# -------------------------------------------
# ADMIN SET TIER
# -------------------------------------------

def admin_set_tier(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    if user_id != ADMIN_USER_ID:
        return update.message.reply_text("❌ You are not admin")

    try:
        target = int(context.args[0])
        tier = context.args[1].lower()

        if tier not in TIERS:
            return update.message.reply_text("❌ Invalid tier")

        if target not in users:
            users[target] = {"tier": tier, "positions": []}
        else:
            users[target]["tier"] = tier

        update.message.reply_text(f"✅ Updated Tier for {target} → {tier}")

    except:
        update.message.reply_text("Format:\n/settier <user_id> <tier>")


# -------------------------------------------
# PROCESS USER POSITION INPUT
# -------------------------------------------

def parse_position(text):
    """
    BUY BANKNIFTY 48000 CE @ 215 2 lots
    BUY NIFTY 22000 PE @ 165 50 qty
    """
    try:
        words = text.upper().split()
        action = words[0]
        index = words[1]
        strike = int(words[2])
        opt_type = words[3]
        price = float(words[words.index("@") + 1])

        qty = 50  # default
        if "LOTS" in words:
            lot_idx = words.index("LOTS")
            lots = int(words[lot_idx - 1])
            qty = lots * (50 if index == "NIFTY" else 25)
        elif words[-1].isdigit():
            qty = int(words[-1])

        return {
            "action": action,
            "index": index,
            "strike": strike,
            "type": opt_type,
            "price": price,
            "qty": qty,
        }
    except:
        return None


def handle_position(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id

    if users[user_id]["tier"] == "free":
        return update.message.reply_text("❌ Free tier users cannot add positions")

    pos = parse_position(update.message.text)
    if not pos:
        return update.message.reply_text("❌ Invalid format\n\nUse: BUY BANKNIFTY 48000 CE @ 215 2 lots")

    users[user_id]["positions"].append(pos)
    update.message.reply_text(f"✅ Position Added:\n{pos}")


# -------------------------------------------
# MANUAL PREDICTION
# -------------------------------------------

def command_predict(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id

    for index in ["NIFTY", "BANKNIFTY"]:
        spot, fii, dii, sentiment, bias = fetch_pre_market(index)
        low, high = predict_range(spot, bias)

        update.message.reply_text(
            f"📊 *{index} Prediction*\n"
            f"Spot: {spot}\n"
            f"FII: {fii}\nDII: {dii}\n"
            f"Sentiment: {sentiment}\n"
            f"Range: {low} - {high}",
            parse_mode="Markdown",
        )


# -------------------------------------------
# AUTO DAILY PUSH
# -------------------------------------------

def auto_push(bot):
    for uid in users:
        for index in ["NIFTY", "BANKNIFTY"]:
            spot, fii, dii, sentiment, bias = fetch_pre_market(index)
            low, high = predict_range(spot, bias)

            bot.send_message(
                chat_id=uid,
                text=(
                    f"📈 *Auto {index} Prediction*\n"
                    f"Spot: {spot}\n"
                    f"Range: {low} - {high}"
                ),
                parse_mode="Markdown",
            )


# -------------------------------------------
# SCHEDULER
# -------------------------------------------

def start_scheduler(updater):
    scheduler = BackgroundScheduler(timezone=TZ)

    scheduler.add_job(auto_push, "cron", hour=9, minute=10, args=[updater.bot], timezone=TZ)
    scheduler.add_job(auto_push, "cron", hour=12, minute=0, args=[updater.bot], timezone=TZ)

    scheduler.start()


# -------------------------------------------
# MAIN
# -------------------------------------------

def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("predict", command_predict))
    dp.add_handler(CommandHandler("settier", admin_set_tier))
    dp.add_handler(CallbackQueryHandler(tier_callback))
    dp.add_handler(MessageHandler(Filters.text, handle_position))

    start_scheduler(updater)

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
