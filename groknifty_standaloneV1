# ================================================
# GROKNIFTY PERSONAL PREDICTION + POSITION ADVISOR
# Run anytime — gives live forecast + your trade advice
# ================================================

import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

print("Fetching live data...")

# 1. Live Bank Nifty Spot
bnf = yf.download("^NSEBANK", period="2d", interval="1m", progress=False)
spot = round(bnf["Close"].iloc[-1], 2)

# 2. FII/DII
try:
    html = requests.get("https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.html", timeout=10).text
    soup = BeautifulSoup(html, "html.parser")
    fii = float(soup.select_one(".fii span.value").text.replace(",", ""))
    dii = float(soup.select_one(".dii span.value").text.replace(",", ""))
except:
    fii = dii = 0

# 3. News Sentiment
try:
    html = requests.get("https://www.moneycontrol.com/news/business/markets/", timeout=10).text
    soup = BeautifulSoup(html, "html.parser")
    headlines = [h.text.strip() for h in soup.find_all("h2", limit=10)]
    sentiment = sum(sia.polarity_scores(h)["compound"] for h in headlines) / len(headlines)
except:
    sentiment = 0

# 4. Today’s Market Bias & Range
bias = (dii - fii) / 1000 + sentiment * 5
low  = spot - 220 + (bias * 45)
high = spot + 220 + (bias * 45)
bias_text = "BULLISH" if bias > 0 else "BEARISH"
strength = "STRONG" if abs(bias) > 3 else "MILD" if abs(bias) > 1 else "NEUTRAL"

print("\n" + "="*55)
print(f"   GROKNIFTY LIVE PREDICTION — {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
print("="*55)
print(f"Spot        : {spot:,.0f}")
print(f"FII / DII   : ₹{fii:,+} Cr  /  ₹{dii:,+} Cr")
print(f"Bias        : {bias_text} ({strength}) → {bias:+.2f}")
print(f"Expected Range : {low:,.0f} → {high:,.0f}")
print(f"Target Close   : {spot + bias*85:,.0f}")
print("="*55)

# 5. Ask for your position (optional
print("\nDo you have an open Bank Nifty option position? (y/n)")
choice = input("→ ").strip().lower()

if choice in ['y', 'yes', '1']:
    print("\nEnter your position in this format:")
    print("Example: BUY 59200 CE 35 @ 245    or    SELL 59000 PE 70 @ 180")
    pos = input("Your position → ").upper().split()

    try:
        action   = pos[0]          # BUY or SELL
        strike   = int(pos[1])
        opt_type = pos[2]          # CE or PE
        qty      = int(pos[3]) if len(pos) > 3 and pos[3].isdigit() else 35
        price    = float(pos[-1])   # LTP at entry

        distance = spot - strike if opt_type == "CE" else strike - spot
        current_ltp = price * (1 + (bias/10)) * (1 + (distance/spot))  # Rough live LTP estimate
        current_ltp = max(current_ltp, 1)  # avoid negative

        pl = (current_ltp - price) * qty if action == "BUY" else (price - current_ltp) * qty

        target = price * (2.0 if action == "BUY" else 0.4)
        stop   = price * (0.55 if action == "BUY" else 1.7)

        print("\n" + "═"*50)
        print("       YOUR POSITION ANALYSIS")
        print("═"*50)
        print(f"{action} {qty} × {strike} {opt_type} @ ₹{price}")
        print(f"Current Spot : {spot:,.0f} | Distance: {distance:+.0f} pts")
        print(f"Est. Live LTP : ₹{current_ltp:.1f}")
        print(f"Current P&L    : ₹{pl:+,.0f} ({pl/price/qty*100:+.1f}%)")
        print(f"Target         : ₹{target:.0f} → +₹{(target-price)*qty if action=='BUY' else (price-target)*qty:+,.0f}")
        print(f"Stop-Loss      : ₹{stop:.0f}")
        print(f"Advice         : {'HOLD – Trend in your favour' if (action=='BUY' and bias>0) or (action=='SELL' and bias<0) else 'CAUTION – Against trend'}")
        print("═"*50)
    except:
        print("Wrong format — skipping position analysis")

print("\nDone! Save this output or run again anytime.")
