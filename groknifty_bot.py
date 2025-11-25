import os
import time
from datetime import datetime, timedelta, timezone
import schedule
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, CallbackQueryHandler

# Environment vars (set in Heroku or local)
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
NEWS_URL = 'https://www.moneycontrol.com/news/business/markets/'
FII_DII_URL = 'https://www.moneycontrol.com/news/business/markets/'  # Scrape or use API

# DB (simple dict for start, use SQLite later)
users = {}  # user_id: {'tier': 'free', 'positions': [], 'data_opt_in': False, 'credits': 0, 'predictions_today': 0, 'positions_month': 0, 'paid': False, 'admin': False}

# Prediction model (simple)
model = LinearRegression()  # Simple ML for range
historical_data = pd.DataFrame()  # Load or fetch

# Sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Admin ID (set your Telegram user ID here for security)
ADMIN_ID = 123456789  # Replace with your actual Telegram user ID

def fetch_pre_market():
    # Gift Nifty (yfinance for Bank Nifty)
    gift = yf.download('^NSEBANK', period='1d')
    spot = gift['Close'].iloc[-1] if not gift.empty else 59000  # Fallback
    # FII/DII (scrape)
    response = requests.get(FII_DII_URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    fii = float(soup.find(text='FII').find_next('span').text) if soup.find(text='FII') else 0
    dii = float(soup.find(text='DII').find_next('span').text) if soup.find(text='DII') else 0
    # News (top 10 headlines)
    news_resp = requests.get(NEWS_URL)
    soup = BeautifulSoup(news_resp.text, 'html.parser')
    headlines = [h.text for h in soup.find_all('h2', class_='news_head')[:10]]
    sentiment = sum(sia.polarity_scores(h)['compound'] for h in headlines) / 10 if headlines else 0
    return spot, fii, dii, sentiment

def predict_range(bias, spot):
    # Simple regression (dummy example – replace with trained model)
    low = spot - 200 + bias * 50
    high = spot + 200 + bias * 50
    return low, high

def update_logic(previous, current):
    # Simple update
    if abs(previous - current) > 100:
        print("Logic updated based on change")

def start(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    if user_id not in users:
        users[user_id] = {'tier': 'free', 'positions': [], 'data_opt_in': False, 'credits': 0, 'predictions_today': 0, 'positions_month': 0, 'paid': False, 'admin': user_id == ADMIN_ID}
    if users[user_id]['admin']:
        users[user_id]['tier'] = 'diamond'  # Admin always diamond
    if not users[user_id]['paid']:
        update.message.reply_text("Please complete payment to select tier or contact admin.")
        return
    update.message.reply_text("Welcome to GrokNifty! Choose tier:", reply_markup=get_tier_menu())

def get_tier_menu():
    keyboard = [
        [InlineKeyboardButton("Free", callback_data='tier_free')],
        [InlineKeyboardButton("Copper", callback_data='tier_copper')],
        [InlineKeyboardButton("Silver", callback_data='tier_silver')],
        [InlineKeyboardButton("Gold", callback_data='tier_gold')],
        [InlineKeyboardButton("Diamond", callback_data='tier_diamond')]
    ]
    return InlineKeyboardMarkup(keyboard)

def tier_callback(update: Update, context: CallbackContext):
    query = update.callback_query
    user_id = query.from_user.id
    tier = query.data.split('_')[1]
    users[user_id]['tier'] = tier
    query.answer("Tier selected: " + tier.capitalize())

def position_input(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    tier = users[user_id]['tier']
    if tier == 'free':
        update.message.reply_text("Free tier doesn't support position input.")
        return
    # Parse input: BUY BANKNIFTY NOV 59500 CE @ 215 105 qty or 3 lots
    text = update.message.text.upper().strip()
    words = text.split()
    try:
        action = words[0]  # BUY/SELL
        index = words[1]  # BANKNIFTY/NIFTY
        month = words[2]  # NOV
        strike = int(words[3])
        opt_type = words[4]  # CE/PE
        price_idx = words.index('@') + 1
        price = float(words[price_idx])
        qty = 35  # Default 1 lot
        if len(words) > price_idx + 1:
            next_word = words[price_idx + 1]
            if next_word.isdigit():
                qty = int(next_word)
            if len(words) > price_idx + 2 and words[price_idx + 2].lower() == 'lots':
                qty *= 35  # Convert lots to qty
        # Store
        position = {'action': action, 'index': index, 'month': month, 'strike': strike, 'opt_type': opt_type, 'price': price, 'qty': qty}
        users[user_id]['positions'].append(position)
        # Dummy prediction (add real)
        update.message.reply_text("Position added: " + str(position))
    except:
        update.message.reply_text("Invalid format. Try: BUY BANKNIFTY NOV 59500 CE @ 215 3 lots")

def auto_fetch():
    for user_id in users:
        tier = users[user_id]['tier']
        # ... (same as before, add diamond/gold schedules in schedule_tasks)

def schedule_tasks():
    # ... (same as before)

def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(tier_callback))
    dp.add_handler(MessageHandler(Filters.text, position_input))
    updater.start_polling()
    schedule_tasks()
    updater.idle()

if __name__ == '__main__':
    main()
