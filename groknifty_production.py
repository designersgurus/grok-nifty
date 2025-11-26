#!/usr/bin/env python3
"""
GrokNifty - production-ready fixed version
Fetches NIFTY and BANKNIFTY spots, simple prediction ranges,
and pushes via Telegram. Robust error handling + logging.
"""

import os
import logging
import traceback
from datetime import datetime, timezone, timedelta
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure vader lexicon available
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# ----- Configuration -----
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # must be set
NEWS_URL = "https://www.moneycontrol.com/news/business/markets/"
FII_DII_URL = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.html"
LOGFILE = "/home/ubuntu/grok-nifty/groknifty_production_fixed.log"

# ----- Logging -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("grok")

if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN not set in environment. Exiting.")
    raise SystemExit("TELEGRAM_TOKEN not set")

# In-memory DB for users -> simple structure (persist later if needed)
users = {}  # user_id: {'tier': 'free', 'positions': []}

# ----- Utilities -----
def safe_get(url, timeout=8):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.warning("safe_get failed for %s : %s", url, e)
        return ""

def fetch_fii_dii():
    """Try to scrape FII/DII numbers from page; fallback to 0."""
    txt = safe_get(FII_DII_URL)
    if not txt:
        return 0.0, 0.0
    try:
        soup = BeautifulSoup(txt, "html.parser")
        # Attempt common selectors (site may change)
        fii_node = soup.select_one(".fii span.value") or soup.find(text="FII")
        dii_node = soup.select_one(".dii span.value") or soup.find(text="DII")
        # If we found nodes with numbers as text, parse them; else fallback 0
        def parse_node(n):
            if not n:
                return 0.0
            text = n.text if hasattr(n, "text") else str(n)
            text = text.strip().replace(",", "")
            # try extract digits
            try:
                return float(text)
            except:
                # try find float inside string
                import re
                m = re.search(r"[-+]?\d+(\.\d+)?", text)
                return float(m.group(0)) if m else 0.0
        fii_val = parse_node(fii_node)
        dii_val = parse_node(dii_node)
        return fii_val, dii_val
    except Exception as e:
        logger.exception("Error parsing FII/DII: %s", e)
        return 0.0, 0.0

def fetch_headlines_and_sentiment():
    txt = safe_get(NEWS_URL)
    if not txt:
        return [], 0.0
    try:
        soup = BeautifulSoup(txt, "html.parser")
        # grab visible headlines: try several tags
        headlines = []
        # moneycontrol uses various tags; we try h2/h3 links, limit 10
        for tag in soup.find_all(["h2","h3","a"], limit=20):
            t = tag.get_text(strip=True)
            if t and len(t) > 10:
                headlines.append(t)
            if len(headlines) >= 10:
                break
        if not headlines:
            return [], 0.0
        compounds = [sia.polarity_scores(h)["compound"] for h in headlines]
        sentiment = sum(compounds) / len(compounds)
        return headlines, sentiment
    except Exception as e:
        logger.exception("Error fetching headlines: %s", e)
        return [], 0.0

def fetch_spot(ticker):
    """Fetch latest close (spot) for supplied ticker using yfinance."""
    try:
        df = yf.download(ticker, period="1d", progress=False, threads=False)
        if df is None or df.empty:
            logger.warning("yfinance returned empty for %s", ticker)
            return None
        close = df["Close"].iloc[-1]
        return float(close)
    except Exception as e:
        logger.exception("fetch_spot error for %s: %s", ticker, e)
        return None

def predict_simple(spot, fii, dii, sentiment):
    """
    Simple heuristic prediction:
    - predicted_next = spot * (1 + small_delta)
    - small_delta formed from (dii-fii) and sentiment
    This is a placeholder. Replace with real model later.
    """
    if spot is None:
        return None, None, None, 0.0
    # small biases
    bias_from_funds = (dii - fii) / 1_000_00.0  # scaled down
    bias_from_sent = sentiment * 0.002          # small scale
    delta = bias_from_funds + bias_from_sent
    predicted_next = spot * (1 + delta)
    # range +/- 0.5% plus bias
    range_width = 0.005 * spot
    low = predicted_next - range_width
    high = predicted_next + range_width
    # confidence: small function of absolute sentiment + scaled fund flow
    confidence = max(0.0, min(1.0, (abs(sentiment) * 0.5) + min(0.5, abs(dii - fii) / 5000.0)))
    return float(predicted_next), float(low), float(high), float(confidence)

def format_prediction_block(index_name, spot, pred, low, high):
    if spot is None:
        return f"{index_name}\nSpot: N/A\nPrediction: N/A\n"
    return (
        f"{index_name}\n"
        f"Spot: {spot:,.2f}\n"
        f"Predicted Next Close: {pred:,.2f} ({((pred-spot)/spot*100):+.2f}%)\n"
        f"Estimated Range: Low {low:,.2f} → High {high:,.2f}\n"
    )

# ----- Telegram bot logic -----
def get_tier_menu():
    keyboard = [
        [InlineKeyboardButton("Free", callback_data="tier_free")],
        [InlineKeyboardButton("Copper", callback_data="tier_copper")],
        [InlineKeyboardButton("Silver", callback_data="tier_silver")],
        [InlineKeyboardButton("Gold", callback_data="tier_gold")],
        [InlineKeyboardButton("Diamond", callback_data="tier_diamond")],
    ]
    return InlineKeyboardMarkup(keyboard)

def start(update, context):
    user_id = update.message.from_user.id
    if user_id not in users:
        users[user_id] = {"tier": "free", "positions": []}
    update.message.reply_text("Welcome to GrokNifty AI Bot! Choose your tier:", reply_markup=get_tier_menu())

def tier_callback(update, context):
    query = update.callback_query
    user_id = query.from_user.id
    tier = query.data.split("_")[1]
    users.setdefault(user_id, {"tier":"free","positions":[]})
    users[user_id]["tier"] = tier
    query.answer(text=f"Tier set to {tier.capitalize()}")
    query.edit_message_text(text=f"Tier set to {tier.capitalize()}")

def predict_cmd(update, context):
    """Manual trigger - compute and reply prediction for both indexes."""
    try:
        send_prediction_to_chat(update.message.chat_id, context.bot)
    except Exception as e:
        logger.exception("predict_cmd failed: %s", e)
        update.message.reply_text("Prediction failed. Try again later.")

# ----- Core push function used by both scheduler and manual command -----
def send_prediction_to_chat(chat_id, bot):
    # fetch data
    headlines, sentiment = fetch_headlines_and_sentiment()
    fii, dii = fetch_fii_dii()
    # fetch spots separately
    nifty_spot = fetch_spot("^NSEI")      # NIFTY 50
    bank_spot = fetch_spot("^NSEBANK")    # BANK NIFTY
    # predictions
    nifty_pred, nifty_low, nifty_high, nifty_conf = predict_simple(nifty_spot, fii, dii, sentiment)
    bank_pred, bank_low, bank_high, bank_conf = predict_simple(bank_spot, fii, dii, sentiment)
    # format message carefully - avoid nested f-string pitfalls
    header_ts = datetime.now(tz=timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M IST")
    parts = []
    parts.append(f"GrokNifty Prediction ({header_ts}):")
    parts.append("")
    parts.append(format_prediction_block("NIFTY", nifty_spot, nifty_pred, nifty_low, nifty_high))
    parts.append(format_prediction_block("BANKNIFTY", bank_spot, bank_pred, bank_low, bank_high))
    parts.append(f"FII: {fii}")
    parts.append(f"DII: {dii}")
    parts.append(f"Sentiment (news avg): {sentiment:.3f}")
    message_text = "\n".join(parts)
    # send (can be long)
    bot.send_message(chat_id=chat_id, text=message_text)

# ----- Scheduler -----
scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

def auto_push_all_users():
    """Push to all subscribed users (respect tier limits later)."""
    # run for each user
    from telegram import Bot
    bot = Bot(token=TELEGRAM_TOKEN)
    for user_id in list(users.keys()):
        try:
            send_prediction_to_chat(user_id, bot)
        except Exception as e:
            logger.exception("auto_push failed for %s: %s", user_id, e)

# ----- Runner -----
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    # commands
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("predict", predict_cmd))
    dp.add_handler(CallbackQueryHandler(tier_callback))
    # start scheduler jobs
    # sample schedule: pre-market and mid-day + EOD
    scheduler.add_job(auto_push_all_users, "cron", hour="9", minute="10")
    scheduler.add_job(auto_push_all_users, "cron", hour="12", minute="0")
    scheduler.add_job(auto_push_all_users, "cron", hour="15", minute="25")
    scheduler.start()
    # start bot
    updater.start_polling()
    logger.info("GrokNifty bot started")
    updater.idle()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error in main()")
        raise
