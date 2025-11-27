#!/usr/bin/env python3
"""
GrokNifty production bot
- dynamic NIFTY / BANKNIFTY predictions per user
- positions input influence on bias
- admin tier management
- APScheduler with pytz timezone (Asia/Kolkata)
- logging, error handling
"""

import os
import re
import json
import logging
from datetime import datetime
import pytz
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ParseMode,
)
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackQueryHandler,
    CallbackContext,
)

# -------------------------
# Config / ENV
# -------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))  # set admin Telegram user id
DB_FILE = "users.json"
TZ = pytz.timezone("Asia/Kolkata")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("groknifty.log"), logging.StreamHandler()],
)
logger = logging.getLogger("groknifty")

# Moneycontrol endpoints
NEWS_URL = "https://www.moneycontrol.com/news/business/markets/"
FII_DII_URL = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.html"

# Tickers
TICKERS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
}

# Default model/config params
DEFAULT_RANGE_PTS = 200  # +/- points around spot
POSITION_BIAS_FACTOR = 0.001  # how position P/L influences bias
SENTIMENT_FACTOR = 50  # scale sentiment to points

# -------------------------
# Utilities: DB save/load
# -------------------------
def load_users():
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.exception("load_users failed")
        return {}

def save_users(data):
    try:
        with open(DB_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.exception("save_users failed")

users = load_users()  # in-memory + persisted file

# -------------------------
# Data fetchers
# -------------------------
def fetch_spot(index_name: str):
    """
    Return last price for TICKERS[index_name].
    If intraday data is empty, use history(period='5d')['Close'].iloc[-1]
    """
    try:
        ticker = TICKERS.get(index_name.upper(), TICKERS["NIFTY"])
        df = yf.download(ticker, period="1d", interval="1m", progress=False)
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
        # fallback to last close
        df2 = yf.download(ticker, period="5d", interval="1d", progress=False)
        if df2 is not None and not df2.empty:
            return float(df2["Close"].iloc[-1])
    except Exception as e:
        logger.warning("fetch_spot failed for %s: %s", index_name, e)
    return 0.0


def fetch_fii_dii():
    """
    Try to scrape FII / DII numeric values from Moneycontrol.
    If scraping fails, return 0,0
    """
    try:
        resp = requests.get(FII_DII_URL, timeout=8)
        if resp.status_code != 200:
            return 0.0, 0.0
        soup = BeautifulSoup(resp.text, "html.parser")

        # Moneycontrol page structure changes often; attempt robust parsing
        text = soup.get_text(separator=" ")
        # find patterns like "FII 1,234" or "FII: 1,234"
        fii = 0.0
        dii = 0.0
        # regex search for FII followed by optional chars and number
        m = re.search(r"FII[^0-9\-]*([\-]?[0-9,]+)", text)
        if m:
            fii = float(m.group(1).replace(",", ""))
        m2 = re.search(r"DII[^0-9\-]*([\-]?[0-9,]+)", text)
        if m2:
            dii = float(m2.group(1).replace(",", ""))
        return fii, dii
    except Exception as e:
        logger.warning("fetch_fii_dii failed: %s", e)
        return 0.0, 0.0


def fetch_sentiment():
    """
    Scrape a few headlines from the news page and compute a simple sentiment via
    polarity score using a tiny heuristic (since nltk download & vader might be heavy).
    We'll use a naive approach: count positive/negative keywords.
    """
    positive_words = {"gain", "rise", "rally", "up", "surge", "beat", "optimis", "improve"}
    negative_words = {"fall", "drop", "decline", "down", "loss", "weak", "worry", "cut"}
    try:
        resp = requests.get(NEWS_URL, timeout=8)
        if resp.status_code != 200:
            return 0.0
        soup = BeautifulSoup(resp.text, "html.parser")
        # collect a few headline strings
        headings = []
        for h in soup.find_all(["h1", "h2", "h3"], limit=12):
            text = h.get_text(separator=" ").strip()
            if text:
                headings.append(text.lower())
        if not headings:
            return 0.0
        score = 0.0
        for h in headings:
            for w in positive_words:
                if w in h:
                    score += 1.0
            for w in negative_words:
                if w in h:
                    score -= 1.0
        # normalize
        return score / max(1.0, len(headings))
    except Exception as e:
        logger.warning("fetch_sentiment failed: %s", e)
        return 0.0

# -------------------------
# Prediction / logic
# -------------------------
def compute_bias_from_positions(positions):
    """Very simple bias: long positions -> positive bias, shorts negative.
       positions is a list of dicts with keys action (BUY/SELL), qty, price
    """
    if not positions:
        return 0.0
    bias = 0.0
    for p in positions:
        action = p.get("action", "BUY").upper()
        qty = float(p.get("qty", 0) or 0)
        entry = float(p.get("price", 0) or 0)
        # if entry is 0 skip
        if entry == 0 or qty == 0:
            continue
        # simplistic: assume expected delta = 1 * qty (this is arbitrary)
        # use qty to add/subtract bias. Later we scale.
        if action == "BUY":
            bias += qty
        else:
            bias -= qty
    # scale down
    return bias * POSITION_BIAS_FACTOR


def predict_for_index(index_name, positions=None):
    """
    Returns dict with spot, fii, dii, sentiment, low, high, bias
    """
    spot = fetch_spot(index_name)
    fii, dii = fetch_fii_dii()
    sentiment = fetch_sentiment()

    pos_bias = compute_bias_from_positions(positions or [])
    # compose bias: FII/DII net + sentiment + positions
    fii_dii_net = (dii - fii) / 1000.0  # scale down
    bias = fii_dii_net + sentiment * 0.1 + pos_bias

    # range calculation scaled by bias and DEFAULT_RANGE_PTS
    rng = DEFAULT_RANGE_PTS + bias * SENTIMENT_FACTOR
    low = spot - rng
    high = spot + rng

    return {
        "spot": spot,
        "fii": fii,
        "dii": dii,
        "sentiment": round(sentiment, 4),
        "bias": round(bias, 6),
        "low": round(low, 2),
        "high": round(high, 2),
    }

# -------------------------
# Telegram bot handlers
# -------------------------
def get_tier_keyboard():
    kb = [
        [InlineKeyboardButton("Free", callback_data="tier_free")],
        [InlineKeyboardButton("Copper", callback_data="tier_copper")],
        [InlineKeyboardButton("Silver", callback_data="tier_silver")],
        [InlineKeyboardButton("Gold", callback_data="tier_gold")],
    ]
    return InlineKeyboardMarkup(kb)


def start(update: Update, context: CallbackContext):
    uid = str(update.effective_user.id)
    if uid not in users:
        users[uid] = {"tier": "free", "positions": [], "last_index": "NIFTY"}
        save_users(users)
    update.message.reply_text(
        "Welcome to GrokNifty ✅\nChoose your tier:",
        reply_markup=get_tier_keyboard(),
    )


def tier_callback(update: Update, context: CallbackContext):
    query = update.callback_query
    data = query.data  # e.g. tier_gold
    _, tier = data.split("_", 1)
    uid = str(query.from_user.id)
    if uid not in users:
        users[uid] = {"tier": tier, "positions": [], "last_index": "NIFTY"}
    else:
        users[uid]["tier"] = tier
    save_users(users)
    query.answer(f"Tier set to {tier}")
    query.edit_message_text(f"✅ Tier set to *{tier.upper()}*", parse_mode=ParseMode.MARKDOWN)


def parse_position_text(text: str):
    """
    Expect formats like:
    BUY BANKNIFTY 59500 CE @ 150 3 lots
    BUY NIFTY 18000 PE @ 30 1
    Returns dict or None
    """
    try:
        t = text.upper().replace(",", " ").strip()
        # tokens
        # find action
        m = re.match(r"(BUY|SELL)\s+(\w+)\s+(\d+)\s+(CE|PE)\s+@?\s*([0-9]*\.?[0-9]+)\s*(\d+)?", t)
        if not m:
            return None
        action = m.group(1)
        index = m.group(2)  # BANKNIFTY or NIFTY
        strike = int(m.group(3))
        opt_type = m.group(4)
        price = float(m.group(5))
        qty = int(m.group(6) or 1)
        # convert lots to qty assumption: banknifty lot size 25, nifty 50 (common)
        lot_size = 25 if "BANK" in index else 50
        if qty <= 10:  # assume provided as lots if small int
            qty = qty * lot_size
        return {
            "action": action,
            "index": index,
            "strike": strike,
            "type": opt_type,
            "price": price,
            "qty": qty,
        }
    except Exception as e:
        logger.exception("parse_position_text error")
        return None


def position_handler(update: Update, context: CallbackContext):
    uid = str(update.effective_user.id)
    text = update.message.text.strip()
    parsed = parse_position_text(text)
    if not parsed:
        update.message.reply_text("❌ Invalid format\nUse: BUY BANKNIFTY 48000 CE @ 215 2 (lots)\nExamples in README.")
        return
    # store
    if uid not in users:
        users[uid] = {"tier": "free", "positions": [], "last_index": parsed["index"]}
    users[uid]["positions"].append(parsed)
    users[uid]["last_index"] = parsed["index"]
    save_users(users)
    update.message.reply_text(f"✅ Position Added:\n`{parsed}`", parse_mode=ParseMode.MARKDOWN)


def cmd_predict(update: Update, context: CallbackContext):
    uid = str(update.effective_user.id)
    u = users.get(uid, {"tier": "free", "positions": [], "last_index": "NIFTY"})
    # compute NIFTY and BANKNIFTY predictions and send both (user requested both)
    try:
        # By default both; if user set last_index we still compute both
        for idx in ["NIFTY", "BANKNIFTY"]:
            # positions relevant only if they match index
            positions = [p for p in u.get("positions", []) if p.get("index") == idx]
            res = predict_for_index(idx, positions=positions)
            text = (
                f"📊 *{idx} Prediction*\n"
                f"Spot: {res['spot']}\n"
                f"FII: {res['fii']}\n"
                f"DII: {res['dii']}\n"
                f"Sentiment: {res['sentiment']}\n"
                f"Bias: {res['bias']}\n"
                f"Range: {res['low']} - {res['high']}\n"
            )
            update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.exception("cmd_predict failed")
        update.message.reply_text("Prediction failed. Try again later.")


def cmd_positions(update: Update, context: CallbackContext):
    uid = str(update.effective_user.id)
    u = users.get(uid)
    if not u or not u.get("positions"):
        update.message.reply_text("No positions found.")
        return
    s = "Your positions:\n"
    for i, p in enumerate(u["positions"][-10:], start=1):
        s += f"{i}. {p}\n"
    update.message.reply_text(s)


def cmd_clear_positions(update: Update, context: CallbackContext):
    uid = str(update.effective_user.id)
    if uid in users:
        users[uid]["positions"] = []
        save_users(users)
    update.message.reply_text("Positions cleared.")


# Admin commands
def cmd_admin(update: Update, context: CallbackContext):
    if update.effective_user.id != ADMIN_ID:
        update.message.reply_text("Not allowed.")
        return
    # list users and quick set buttons
    if not users:
        update.message.reply_text("No users.")
        return
    kb = []
    for uid in users:
        kb.append([InlineKeyboardButton(uid, callback_data=f"adminuser_{uid}")])
    update.message.reply_text("Select user:", reply_markup=InlineKeyboardMarkup(kb))


def admin_user_select(update: Update, context: CallbackContext):
    query = update.callback_query
    _, uid = query.data.split("_", 1)
    kb = [
        [InlineKeyboardButton("Set Free", callback_data=f"adminset_{uid}_free")],
        [InlineKeyboardButton("Set Copper", callback_data=f"adminset_{uid}_copper")],
        [InlineKeyboardButton("Set Silver", callback_data=f"adminset_{uid}_silver")],
        [InlineKeyboardButton("Set Gold", callback_data=f"adminset_{uid}_gold")],
    ]
    query.edit_message_text(f"Change tier for {uid}", reply_markup=InlineKeyboardMarkup(kb))


def admin_set_tier(update: Update, context: CallbackContext):
    query = update.callback_query
    _, uid, tier = query.data.split("_", 2)
    if uid not in users:
        users[uid] = {"tier": tier, "positions": [], "last_index": "NIFTY"}
    else:
        users[uid]["tier"] = tier
    save_users(users)
    query.edit_message_text(f"Updated {uid} -> {tier}")


# -------------------------
# Scheduler (auto push)
# -------------------------
def auto_push(bot):
    logger.info("Running auto_push job")
    for uid, u in users.items():
        try:
            tier = u.get("tier", "free")
            if tier == "free":
                continue
            # choose index to send (user preference or both)
            idx = u.get("last_index", "NIFTY")
            positions = [p for p in u.get("positions", []) if p.get("index") == idx]
            res = predict_for_index(idx, positions=positions)
            text = (
                f"⏰ Auto Update — *{idx}*\n"
                f"Spot: {res['spot']}\nRange: {res['low']} - {res['high']}\nBias: {res['bias']}"
            )
            bot.send_message(chat_id=int(uid), text=text, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            logger.exception("auto_push error for user %s", uid)


def start_scheduler(updater):
    scheduler = BackgroundScheduler(timezone=TZ)
    # push at 09:15 and 12:00 server IST
    scheduler.add_job(auto_push, "cron", hour=9, minute=15, args=[updater.bot], timezone=TZ)
    scheduler.add_job(auto_push, "cron", hour=12, minute=0, args=[updater.bot], timezone=TZ)
    scheduler.start()
    logger.info("Scheduler started")


# -------------------------
# Main entry
# -------------------------
def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not set. Export TELEGRAM_TOKEN in env file.")
        return

    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Commands
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("predict", cmd_predict))
    dp.add_handler(CommandHandler("positions", cmd_positions))
    dp.add_handler(CommandHandler("clearpos", cmd_clear_positions))
    dp.add_handler(CommandHandler("admin", cmd_admin))

    # Callbacks
    dp.add_handler(CallbackQueryHandler(tier_callback, pattern=r"^tier_"))
    dp.add_handler(CallbackQueryHandler(admin_user_select, pattern=r"^adminuser_"))
    dp.add_handler(CallbackQueryHandler(admin_set_tier, pattern=r"^adminset_"))

    # Position text handler (any non-command text)
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, position_handler))

    # Start scheduler
    start_scheduler(updater)

    # Start polling
    logger.info("Starting polling...")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
