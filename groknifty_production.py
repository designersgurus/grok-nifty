import os
import time
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackContext,
    CallbackQueryHandler
)

# -----------------------------
# ENVIRONMENT
# -----------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

NEWS_URL = "https://www.moneycontrol.com/news/business/markets/"
FII_DII_URL = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.html"

users = {}  # Simple dictionary DB

sia = SentimentIntensityAnalyzer()
model = LinearRegression()


# -----------------------------
# DATA FETCHING
# -----------------------------
def fetch_pre_market():
    try:
        banknifty = yf.download("^NSEBANK", period="1d")
        spot = float(banknifty["Close"].iloc[-1]) if not banknifty.empty else 45000.0
    except:
        spot = 45000.0

    try:
        r = requests.get(FII_DII_URL, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        fii = soup.select_one(".fii span.value")
        dii = soup.select_one(".dii span.value")
        fii_val = float(fii.text.replace(",", "")) if fii else 0
        dii_val = float(dii.text.replace(",", "")) if dii else 0
    except:
        fii_val, dii_val = 0, 0

    try:
        r = requests.get(NEWS_URL, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        headlines = [h.text.strip() for h in soup.find_all("h2", limit=10)]
        sentiment = (
            sum(sia.polarity_scores(h)["compound"] for h in headlines) / len(headlines)
            if headlines else 0
        )
    except:
        sentiment = 0

    return spot, fii_val, dii_val, sentiment


# -----------------------------
# PREDICTION LOGIC
# -----------------------------
def predict_range(bias, spot):
    low = spot - 200 + (bias * 50)
    high = spot + 200 + (bias * 50)
    return low, high


# -----------------------------
# POSITION HANDLING
# -----------------------------
def parse_position(text):
    try:
        text = text.upper().strip()
        parts = text.split()

        action = parts[0]       # BUY/SELL
        index = parts[1]        # BANKNIFTY/NIFTY
        month = parts[2]        # NOV
        strike = int(parts[3])  # strike price
        opt_type = parts[4]     # CE/PE

        price_index = parts.index("@") + 1
        entry_price = float(parts[price_index])

        qty = 25  # default
        if len(parts) > price_index + 1:
            nxt = parts[price_index + 1]
            if nxt.isdigit():
                qty = int(nxt)

        return {
            "action": action,
            "index_symbol": index,
            "month": month,
            "strike": strike,
            "opt_type": opt_type,
            "price": entry_price,
            "qty": qty
        }
    except:
        return None


def save_position(user_id, pos):
    if user_id not in users:
        users[user_id] = {"tier": "free", "positions": []}

    users[user_id]["positions"].append(pos)


def get_positions_for_user(user_id):
    if user_id not in users:
        return []
    return users[user_id].get("positions", [])


# -----------------------------
# TELEGRAM COMMANDS
# -----------------------------
def start(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    if user_id not in users:
        users[user_id] = {"tier": "free", "positions": []}

    keyboard = [
        [InlineKeyboardButton("Free", callback_data="tier_free")],
        [InlineKeyboardButton("Copper", callback_data="tier_copper")],
        [InlineKeyboardButton("Silver", callback_data="tier_silver")],
        [InlineKeyboardButton("Gold", callback_data="tier_gold")],
        [InlineKeyboardButton("Diamond", callback_data="tier_diamond")]
    ]

    update.message.reply_text(
        "Welcome to GrokNifty AI Bot!\nSelect your tier:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


def tier_callback(update: Update, context: CallbackContext):
    query = update.callback_query
    tier = query.data.split("_")[1]
    users[query.from_user.id]["tier"] = tier
    query.answer(f"Tier set to {tier.capitalize()}")


def predict_on_demand(update: Update, context: CallbackContext):
    try:
        spot, fii, dii, sentiment = fetch_pre_market()
        bias = (dii - fii) / 1000 + sentiment * 5
        low, high = predict_range(bias, spot)
        pred_price = (low + high) / 2
        pred_return = ((pred_price - spot) / spot)

        positions = get_positions_for_user(update.message.from_user.id)

        pos_text = ""
        for p in positions[-5:]:
            pos_text += (f"{p['action']} {p['index_symbol']} {p['month']} "
                         f"{p['strike']}{p['opt_type']} @ {p['price']} qty:{p['qty']}\n")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")

        text = (
            f"GrokNifty Prediction ({timestamp}):\n"
            f"Spot: {spot:.2f}\n"
            f"Predicted Next Close: {pred_price:.2f} ({pred_return*100:.2f}%)\n"
            f"Estimated Range: Low {low:.2f} – High {high:.2f}\n"
            f"Confidence: {abs(pred_return)*100:.1f}%\n"
        )

        if pos_text:
            text += f"\nPositions:\n{pos_text}"

        update.message.reply_text(text)

    except Exception as e:
        update.message.reply_text(f"Prediction failed: {e}")


def position_handler(update: Update, context: CallbackContext):
    text = update.message.text
    pos = parse_position(text)

    if not pos:
        update.message.reply_text("❌ Invalid format.\nTry:\nBUY BANKNIFTY NOV 59500 CE @ 215 3")
        return

    save_position(update.message.from_user.id, pos)
    update.message.reply_text("✅ Position saved!")


# -----------------------------
# MAIN
# -----------------------------
def main():
    if not TELEGRAM_TOKEN:
        print("ERROR: TELEGRAM_TOKEN missing.")
        return

    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("predict", predict_on_demand))
    dp.add_handler(CallbackQueryHandler(tier_callback))
    dp.add_handler(MessageHandler(Filters.text, position_handler))

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
