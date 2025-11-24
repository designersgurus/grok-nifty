\# GrokNifty AI Predictive Bot



A Telegram bot that predicts Nifty/BankNifty ranges using:



\- Market data

\- News sentiment

\- FII/DII activity

\- Lightweight ML model

\- Scheduled auto-updates



\## ⭐ Features

\- Pre-market prediction

\- Mid-day prediction

\- Tier-based system

\- Auto push notifications (cron)

\- Sentiment analysis from headlines

\- FII/DII scraping

\- Fully compatible with Oracle Cloud Always-Free



---



\## 🚀 Deploy on Oracle Cloud (Ubuntu 20/22)



\### 1. Install dependencies

```

sudo apt update \&\& sudo apt install python3 python3-venv python3-pip git -y

```



\### 2. Clone repo

```

git clone https://github.com/<yourname>/grok-nifty.git

cd grok-nifty

```



\### 3. Create ENV file

```

sudo mkdir -p /etc/groknifty

sudo nano /etc/groknifty/groknifty.env

```



Add:

```

TELEGRAM\_TOKEN="123456:ABC-DEF"

```



\### 4. Create venv \& install requirements

```

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

```



\### 5. Run the bot

```

python groknifty\_bot.py

```



\### 6. (Optional) Create systemd service for 24/7 running



```

sudo nano /etc/systemd/system/groknifty.service

```



Paste:



```

\[Unit]

Description=GrokNifty Telegram Bot

After=network.target



\[Service]

User=ubuntu

WorkingDirectory=/home/ubuntu/grok-nifty

EnvironmentFile=/etc/groknifty/groknifty.env

ExecStart=/home/ubuntu/grok-nifty/venv/bin/python /home/ubuntu/grok-nifty/groknifty\_bot.py

Restart=always

RestartSec=5



\[Install]

WantedBy=multi-user.target

```



Enable:



```

sudo systemctl daemon-reload

sudo systemctl enable groknifty

sudo systemctl start groknifty

```



---



\# ✔ You're ready to deploy!



