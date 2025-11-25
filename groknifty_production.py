#!/usr/bin/env python3
"""
GrokNifty Production with Position Support
- Adds position parsing (BUY/SELL) and position evaluation against predictions
- Keeps previous production features: SQLite DB, ML model (RandomForest), scheduler, Telegram bot
"""

import os
import sys
import time
import json
import logging
import sqlite3
import signal
from contextlib import closing
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, Filters

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure NLTK resource exists
try:
    nltk.data.find('sentiment/vader_lexicon')
except Exception:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# ------------------ Configuration & Paths ------------------
ENV_PATH = '/etc/groknifty/groknifty.env'
DEFAULT_DATA_PATH = '/home/ubuntu/grok-nifty/data'
MODEL_PATH = '/home/ubuntu/grok-nifty/model.pkl'
DB_PATH = '/home/ubuntu/grok-nifty/groknifty.db'
LOG_PATH = '/home/ubuntu/grok-nifty/groknifty.log'

NEWS_URL = 'https://www.moneycontrol.com/news/business/markets/'
FII_DII_URL = 'https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.html'
TICKER = '^NSEI'
HIST_PERIOD = '5y'

scheduler = BackgroundScheduler(timezone='Asia/Kolkata')

updater = None
bot = None
MODEL_FEATURES = None
MODEL = None

# ------------------ Logging ------------------
logger = logging.getLogger('groknifty')
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

# ------------------ Utils ------------------
def load_env(path: str = ENV_PATH) -> Dict[str, str]:
    env = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
    return env

def ensure_dirs(data_path: str):
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ------------------ Database ------------------
def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        tier TEXT DEFAULT 'free',
        subscribed INTEGER DEFAULT 1,
        created_at TEXT
    )''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        dt TEXT,
        spot REAL,
        low REAL,
        high REAL,
        confidence REAL
    )''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        action TEXT,
        index_symbol TEXT,
        month TEXT,
        strike INTEGER,
        opt_type TEXT,
        price REAL,
        qty INTEGER,
        created_at TEXT
    )''')
    conn.commit()
    conn.close()
    logger.info('DB initialized at %s', db_path)

def add_or_get_user_db(user_id: int) -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT user_id, tier, subscribed FROM users WHERE user_id=?', (user_id,))
    row = c.fetchone()
    if row:
        conn.close()
        return {'user_id': row[0], 'tier': row[1], 'subscribed': bool(row[2])}
    c.execute('INSERT INTO users(user_id, tier, subscribed, created_at) VALUES (?, ?, ?, ?)',
              (user_id, 'free', 1, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    return {'user_id': user_id, 'tier': 'free', 'subscribed': True}

def get_users_by_tier_db(tier: str) -> List[int]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT user_id FROM users WHERE tier=? AND subscribed=1', (tier,))
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]

def save_prediction(user_id: int, spot: float, low: float, high: float, confidence: float):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO predictions(user_id, dt, spot, low, high, confidence) VALUES (?, ?, ?, ?, ?, ?)',
              (user_id, datetime.utcnow().isoformat(), spot, low, high, confidence))
    conn.commit()
    conn.close()

def save_position_db(user_id:int, position:Dict[str,Any]):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO positions(user_id, action, index_symbol, month, strike, opt_type, price, qty, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
              (user_id, position['action'], position['index_symbol'], position['month'], position['strike'], position['opt_type'], position['price'], position['qty'], datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_positions_for_user(user_id:int) -> List[Dict[str,Any]]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, action, index_symbol, month, strike, opt_type, price, qty, created_at FROM positions WHERE user_id=? ORDER BY id DESC', (user_id,))
    rows = c.fetchall()
    conn.close()
    result=[]
    for r in rows:
        result.append({'id':r[0],'action':r[1],'index_symbol':r[2],'month':r[3],'strike':r[4],'opt_type':r[5],'price':r[6],'qty':r[7],'created_at':r[8]})
    return result

# ------------------ Data & Features ------------------
def fetch_historical(ticker: str = TICKER, period: str = HIST_PERIOD) -> pd.DataFrame:
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, interval='1d', progress=False)
            if df is not None and not df.empty:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df.dropna(inplace=True)
                return df
        except Exception as e:
            logger.warning('fetch_historical attempt %s failed: %s', attempt+1, e)
            time.sleep(2)
    raise RuntimeError('Unable to fetch historical data for ' + ticker)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ret_1'] = df['Close'].pct_change(1)
    df['ret_5'] = df['Close'].pct_change(5)
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_10'] = df['Close'].rolling(10).mean()
    df['vol_5'] = df['ret_1'].rolling(5).std()
    df['mom_5'] = df['Close'] / df['Close'].shift(5) - 1
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['target'] = df['Close'].shift(-1) / df['Close'] - 1
    df.dropna(inplace=True)
    return df

# ------------------ Model ------------------
def train_and_persist_model(model_path: str = MODEL_PATH) -> Tuple[Any, List[str]]:
    logger.info('Training model...')
    df = fetch_historical()
    df = compute_features(df)
    features = ['ret_1','ret_5','sma_5','sma_10','vol_5','mom_5','rsi_14']
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    logger.info('Model trained. Test MSE: %.8f', mse)
    joblib.dump({'model':model, 'features':features}, model_path)
    logger.info('Model persisted to %s', model_path)
    return model, features

def load_model(model_path: str = MODEL_PATH):
    global MODEL, MODEL_FEATURES
    if os.path.exists(model_path):
        logger.info('Loading persisted model...')
        data = joblib.load(model_path)
        MODEL = data['model']
        MODEL_FEATURES = data['features']
    else:
        MODEL, MODEL_FEATURES = train_and_persist_model()
    return MODEL, MODEL_FEATURES

def predict_next_move(latest_df: pd.DataFrame) -> Tuple[float, float, float, float, float]:
    global MODEL, MODEL_FEATURES
    if MODEL is None:
        load_model()
    last = latest_df.copy().iloc[-1:]
    X = last[MODEL_FEATURES]
    pred_return = float(MODEL.predict(X)[0])
    spot = float(last['Close'].iloc[0])
    pred_price = spot * (1 + pred_return)
    hist_vol = latest_df['ret_1'].rolling(20).std().iloc[-1]
    band = max(0.002, hist_vol if not np.isnan(hist_vol) else 0.01)
    low = pred_price * (1 - 1.5 * band)
    high = pred_price * (1 + 1.5 * band)
    confidence = max(0.0, 1 - min(1.0, abs(pred_return) / (3 * (hist_vol if hist_vol>0 else 0.01))))
    return pred_return, low, high, spot, pred_price, confidence

# ------------------ Fetch helpers ------------------
def fetch_fii_dii() -> Tuple[float,float]:
    try:
        resp = requests.get(FII_DII_URL, timeout=8, headers={'User-Agent':'Mozilla/5.0'})
        soup = BeautifulSoup(resp.text, 'html.parser')
        fii = soup.select_one('.fii span.value')
        dii = soup.select_one('.dii span.value')
        fii_val = float(fii.text.replace(',','')) if fii else 0.0
        dii_val = float(dii.text.replace(',','')) if dii else 0.0
        return fii_val, dii_val
    except Exception as e:
        logger.warning('FII/DII fetch failed: %s', e)
        return 0.0, 0.0

def fetch_headlines_sentiment() -> float:
    try:
        resp = requests.get(NEWS_URL, timeout=8, headers={'User-Agent':'Mozilla/5.0'})
        soup = BeautifulSoup(resp.text, 'html.parser')
        headlines = [h.get_text(strip=True) for h in soup.find_all('h2', limit=10)]
        if not headlines:
            return 0.0
        sentiment = sum(sia.polarity_scores(h)['compound'] for h in headlines) / len(headlines)
        return sentiment
    except Exception as e:
        logger.warning('Headline fetch failed: %s', e)
        return 0.0

# ------------------ Position parsing & evaluation ------------------
def parse_position_text(text: str) -> Optional[Dict[str,Any]]:
    """
    Accepts text like:
      BUY BANKNIFTY NOV 59500 CE @ 215 3 lots
      SELL NIFTY DEC 18750 PE @ 40 2
    Returns dict or None
    """
    try:
        txt = text.upper().strip()
        # normalize separators
        txt = txt.replace(',', ' ')
        words = txt.split()
        # basic minimum length check
        if len(words) < 6:
            return None
        action = words[0]
        if action not in ('BUY','SELL'):
            return None
        index_symbol = words[1]
        month = words[2]
        # strike might be word 3, but if month is like 'NOV', strike at 3
        strike = int(words[3])
        opt_type = words[4]
        # find '@'
        price = None
        qty = 1
        if '@' in words:
            at_idx = words.index('@')
            if at_idx+1 < len(words):
                price = float(words[at_idx+1])
                # check for qty after price
                if at_idx+2 < len(words):
                    nxt = words[at_idx+2]
                    # "3" or "3 lots"
                    if nxt.isdigit():
                        qty = int(nxt)
                        if at_idx+3 < len(words) and words[at_idx+3].startswith('LOT'):
                            qty = qty  # qty means lots; convert to qty below when saving
                    else:
                        # maybe "3lots" or "3LO" etc, try numbers in string
                        digits = ''.join(ch for ch in nxt if ch.isdigit())
                        if digits:
                            qty = int(digits)
        else:
            # no @; fallback: the next token maybe price
            try:
                price = float(words[5])
            except:
                price = None
        if price is None:
            return None
        # convert lots to quantity when user used "lots" as token; assume 1 lot = 1 lot size default (for BankNifty 1 lot=15? your earlier used 35 for eq options)
        # We will keep qty as 'lots' numeric and send both lots and qty (user can interpret)
        position = {
            'action': action,
            'index_symbol': index_symbol,
            'month': month,
            'strike': strike,
            'opt_type': opt_type,
            'price': price,
            'qty': qty
        }
        return position
    except Exception as e:
        logger.exception('parse_position_text error: %s', e)
        return None

def estimate_option_pl(position: Dict[str,Any], predicted_price: float) -> Tuple[float,str]:
    """
    Very rough estimate of option P/L per lot:
    - We do NOT compute Greeks exactly. We use a quick heuristic:
      * If strike is OTM relative to predicted_price, assume low delta (0.1-0.3)
      * If ATM, delta ~ 0.4-0.6
      * If ITM, delta ~ 0.6-0.9
    - Then estimated option price change ~ delta * underlying_move
    - This is only a heuristic to give a 'directional' signal.
    """
    try:
        strike = position['strike']
        opt_type = position['opt_type']
        entry_price = position['price']
        # Underlying predicted price is predicted_price
        move = predicted_price - strike if opt_type == 'CE' else strike - predicted_price
        # underlying percent move relative to strike
        underlying_move = move / max(1.0, strike)
        # determine delta heuristic
        abs_pct = abs(underlying_move)
        if abs_pct < 0.005:
            delta = 0.15
        elif abs_pct < 0.01:
            delta = 0.3
        elif abs_pct < 0.02:
            delta = 0.45
        else:
            delta = 0.65
        # estimated option price change
        est_change = delta * (move if opt_type=='CE' else -move)
        # new option price approx
        est_new_price = max(0.0, entry_price + abs(est_change)*100) if False else entry_price + (delta * (predicted_price - strike))
        # Simpler: estimate option move = delta * (predicted_price - current_spot)
        # We'll compute P/L relative to entry_price (per option)
        approx_pl = (delta * (predicted_price - strike)) - entry_price if opt_type=='CE' else (delta * (strike - predicted_price)) - entry_price
        # But to keep numbers meaningful, provide approximate % vs entry price
        sign = 'Favourable' if approx_pl > 0 else ('Neutral' if abs(approx_pl) < entry_price*0.1 else 'Risky')
        return approx_pl, sign
    except Exception as e:
        logger.exception('estimate_option_pl error: %s', e)
        return 0.0, 'Unknown'

# ------------------ Telegram Bot Handlers ------------------
def get_tier_menu_markup():
    keyboard = [
        [InlineKeyboardButton('Free', callback_data='tier_free')],
        [InlineKeyboardButton('Copper', callback_data='tier_copper')],
        [InlineKeyboardButton('Silver', callback_data='tier_silver')],
        [InlineKeyboardButton('Gold', callback_data='tier_gold')],
        [InlineKeyboardButton('Diamond', callback_data='tier_diamond')],
    ]
    return InlineKeyboardMarkup(keyboard)

def start_handler(update, context):
    user = add_or_get_user_db(update.message.from_user.id)
    update.message.reply_text('Welcome to GrokNifty! Choose tier:', reply_markup=get_tier_menu_markup())

def tier_callback_handler(update, context):
    query = update.callback_query
    user_id = query.from_user.id
    tier = query.data.split('_')[1]
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE users SET tier=? WHERE user_id=?', (tier, user_id))
    conn.commit()
    conn.close()
    query.answer()
    query.edit_message_text(text=f'Tier set to {tier.capitalize()}')

def predict_on_demand(update, context):
    try:
        df = fetch_historical()
        df = compute_features(df)
        pred_return, low, high, spot, pred_price, confidence = predict_next_move(df)
        # Evaluate user's positions (if any)
        positions = get_positions_for_user(update.message.from_user.id)
        pos_text = ''
        if positions:
            for p in positions[:5]:  # show up to 5 latest
                approx_pl, sign = estimate_option_pl(p, pred_price)
                pos_text += f"\n- {p['action']} {p['index_symbol']} {p['month']} {p['strike']}{p['opt_type']} @ {p['price']} qty:{p['qty']} => EstPL:{approx_pl:.2f} ({sign})"
        text = (f"GrokNifty Prediction ({datetime.now().strftime('%Y-%m-%d %H:%M IST')}):\n"
                f"Spot: {spot:.2f}\n"
                f"Predicted Next Close: {pred_price:.2f} ({pred_return*100:.2f}%)\n"
                f"Estimated Range: Low {low:.2f} — High {high:.2f}\n"
                f"Confidence: {confidence*100:.1f}%\n"
                f"{('\nPositions:'+pos_text) if pos_text else ''}"
               )
        update.message.reply_text(text)
    except Exception as e:
        logger.exception('On-demand prediction failed: %s', e)
        update.message.reply_text('Prediction failed. Try again later.')

def position_handler(update, context):
    txt = update.message.text
    parsed = parse_position_text(txt)
    if not parsed:
        update.message.reply_text("Invalid position format. Try: BUY BANKNIFTY NOV 59500 CE @ 215 3")
        return
    # Save to DB
    save_position_db(update.message.from_user.id, parsed)
    update.message.reply_text(f"Position saved: {parsed}")

# ------------------ Scheduler jobs ------------------
def send_forecasts_to_tier(tier: str):
    users = get_users_by_tier_db(tier)
    if not users:
        logger.info('No users for tier %s', tier)
        return
    try:
        df = fetch_historical()
        df = compute_features(df)
        pred_return, low, high, spot, pred_price, confidence = predict_next_move(df)
        text = (f"GrokNifty ({datetime.now().strftime('%Y-%m-%d %H:%M IST')})\n"
                f"Spot: {spot:.2f}\nPred: {pred_price:.2f} ({pred_return*100:.2f}%)\nRange: {low:.2f}-{high:.2f}\nConfidence: {confidence*100:.1f}%")
        for user_id in users:
            try:
                # Append user's positions summary (1-3 latest) to message
                positions = get_positions_for_user(user_id)
                pos_text = ''
                if positions:
                    for p in positions[:3]:
                        approx_pl, sign = estimate_option_pl(p, pred_price)
                        pos_text += f"\n- {p['action']} {p['index_symbol']} {p['month']} {p['strike']}{p['opt_type']} @ {p['price']} qty:{p['qty']} => EstPL:{approx_pl:.2f} ({sign})"
                final_text = text + (("\nYour positions:\n" + pos_text) if pos_text else "")
                bot.send_message(chat_id=user_id, text=final_text)
                save_prediction(user_id, spot, low, high, confidence)
            except Exception as e:
                logger.warning('Failed to send to %s: %s', user_id, e)
    except Exception as e:
        logger.exception('send_forecasts_to_tier failed: %s', e)

def schedule_jobs():
    for job in scheduler.get_jobs():
        scheduler.remove_job(job.id)
    tier_times = {
        'free': ['09:10'],
        'copper': ['09:10','12:30'],
        'silver': ['09:10','11:00','14:30'],
        'gold': ['09:10','09:30','12:30','14:30'],
        'diamond': ['09:10','09:15','09:20','09:25','09:30','10:00','11:00','12:00','13:00','14:00']
    }
    for tier, times in tier_times.items():
        for hm in times:
            hour, minute = map(int, hm.split(':'))
            trigger = CronTrigger(hour=hour, minute=minute, timezone='Asia/Kolkata')
            scheduler.add_job(lambda t=tier: send_forecasts_to_tier(t), trigger=trigger, id=f'{tier}_{hm.replace(":","")}_{hour}_{minute}')
    scheduler.start()
    logger.info('Scheduler started with %d jobs', len(scheduler.get_jobs()))

# ------------------ Graceful shutdown ------------------
def shutdown(signum, frame):
    logger.info('Shutting down GrokNifty...')
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        pass
    try:
        if updater:
            updater.stop()
    except Exception:
        pass
    logger.info('Shutdown complete')
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

# ------------------ CLI helpers & Runner ------------------
def print_help():
    print('Commands: --init-db | --train | --run')

def main_cli():
    env = load_env()
    token = env.get('TELEGRAM_TOKEN') or os.getenv('TELEGRAM_TOKEN')
    data_path = env.get('DATA_PATH', DEFAULT_DATA_PATH)
    ensure_dirs(data_path)

    if '--init-db' in sys.argv:
        init_db()
        print('DB initialized')
        return
    if '--train' in sys.argv:
        init_db()
        train_and_persist_model()
        print('Training complete')
        return

    init_db()
    load_model()

    global updater, bot
    if not token:
        logger.error('TELEGRAM_TOKEN not found. Set /etc/groknifty/groknifty.env or env var.')
        print('Set TELEGRAM_TOKEN in /etc/groknifty/groknifty.env')
        return

    updater = Updater(token, use_context=True)
    bot = updater.bot
    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start_handler))
    dp.add_handler(CommandHandler('predict', predict_on_demand))
    dp.add_handler(CallbackQueryHandler(tier_callback_handler))
    dp.add_handler(MessageHandler(Filters.text & (~Filters.command), position_handler))

    schedule_jobs()
    retrain_trigger = CronTrigger(hour=18, minute=45, timezone='Asia/Kolkata')
    scheduler.add_job(train_and_persist_model, trigger=retrain_trigger, id='daily_retrain')

    logger.info('Starting updater (polling)...')
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main_cli()
