# AI-Powered Stock Trading Bot

This is a sophisticated, fully autonomous stock trading bot that leverages the analytical power of Google's Gemini LLM to make intelligent, data-driven trading decisions. It operates in a dynamic terminal dashboard, providing real-time data, analysis, and controls.

---

## Key Features

* **AI-Powered Decisions:** Uses Google's Gemini API to analyze market data, news sentiment, and technical indicators to make calculated trading decisions.
* **Dynamic Market Scanner:** Automatically scans the market at regular intervals to identify and trade the most volatile and promising stocks.
* **Advanced Technical Analysis:** The bot automatically calculates RSI, MACD, and Bollinger Bands, feeding this quantitative data to the AI for deeper analysis.
* **Self-Correction Loop:** Remembers the profit/loss outcomes of past trades for each stock to inform and improve future decisions.
* **Dynamic Risk Management:** The AI provides a "confidence score" with each decision, allowing the bot to dynamically size its positions—risking more on high-confidence trades and less on low-confidence ones.
* **Automated Risk Controls:** Places bracket orders with pre-defined take-profit and trailing stop-loss levels to manage risk and protect profits on every trade.
* **Live Terminal Dashboard:** A rich, real-time UI displays a price chart, P/L tracking, technical indicators, and an event log.
* **Manual Override:** Full command-line control to pause trading, manually sell a position, or trigger a new market scan at any time.

---

## How It Works

The bot operates in a continuous loop with two main phases:

1.  **Scanning Phase:** Every 15 minutes, the bot fetches a list of hundreds of tradable stocks from the Alpaca API. It asks the Gemini LLM to analyze this list and return a ranked top 3 of the most promising assets for short-term trading. The bot then selects the #1 ranked stock.
2.  **Trading Phase:** Once an asset is selected, the bot enters a high-frequency (60-second) loop where it:
    * Fetches the latest price, news headlines, and 5-minute historical data.
    * Calculates a full suite of technical indicators (RSI, MACD, etc.).
    * Sends all of this data—including the history of its own past trades for that stock—to the Gemini LLM for analysis.
    * Receives a `Buy`, `Sell`, or `Hold` decision, along with a confidence score.
    * Executes the trade using a dynamically sized bracket order to manage risk.

---

## Terminal UI

The bot's dashboard is designed to provide all critical information at a glance.

```

\+--------------------------------------------------------------------------------------+
|                      AI Stock Trading Bot v7.9 - Logging Fix                         |
\+--------------------------------------------------------------------------------------+
| Bot Status                  | NVDA Price Chart                                       |
|-----------------------------|                                                        |
| Current Price:    $120.50   |   ... Chart Data ...                                   |
| Buying Power:  $95,000.00   |                                                        |
| NVDA Holdings: 100 ($12,050) |                                                        |
| Unrealized P/L:    +$50.00   |                                                        |
| Session P/L:      +$150.00   |                                                        |
| Portfolio Value: $100,150.00|                                                        |
|-----------------------------|                                                        |
| Next Cycle In               |                                                        |
|-----------------------------|                                                        |
|       12s                   |                                                        |
|-----------------------------|                                                        |
| Technical Indicators        |                                                        |
|-----------------------------|                                                        |
| RSI (14):          55.32   |                                                        |
| ... more indicators ...     |                                                        |
|-----------------------------|                                                        |
| Manual Controls             |                                                        |
|-----------------------------|                                                        |
| |s   Sell Current Position  |                                                        |
| |p   Pause / Resume Trading |                                                        |
| |n   Scan for New Asset     |                                                        |
| |q   Quit Bot               |                                                        |
\+--------------------------------------------------------------------------------------+
| Event Log                                                                            |
\+--------------------------------------------------------------------------------------+
| Decision: Buy NVDA (Confidence: 8/10).                                               |
| Submitted BUY bracket order for $12,000.00 of NVDA.                                  |
\+--------------------------------------------------------------------------------------+

````

---

## Setup and Installation

Follow these steps to get the bot running.

**1. Create a Virtual Environment**

It is highly recommended to run this project in a dedicated Python virtual environment.

```bash
# Create the environment
python -m venv venv

# Activate the environment
# On Windows (Git Bash)
source venv/Scripts/activate
# On macOS/Linux
source venv/bin/activate
````

**2. Create a `requirements.txt` file**

Create a file named `requirements.txt` in your project directory and paste the following content into it:

```
alpaca-py
rich
python-dotenv
asciichartpy
requests
pandas
pandas-ta
pynput
```

**3. Install Dependencies**

Run the following command in your activated terminal:

```bash
pip install -r requirements.txt
```

**4. Set Up API Keys**

Create a file named `.env` in your project directory. This file will securely store your API keys. Add your keys to it in the following format:

```
# .env file

# Get this from Google AI Studio
GOOGLE_API_KEY="your_google_api_key_here"

# Get these from your Alpaca dashboard
APCA_API_KEY_ID="your_alpaca_key_id_here"
APCA_API_SECRET_KEY="your_alpaca_secret_key_here"
```

-----

## Usage

Once the setup is complete, simply run the script from your activated terminal:

```bash
python main.py
```

The bot will start, display the loading animation, and then launch the main dashboard.

### Manual Controls

While the bot is running, you can use the following commands in the terminal where the bot is running. Type the command and press Enter.

  * `|s` - **Sell:** Immediately sells any open position for the current stock.
  * `|p` - **Pause/Resume:** Toggles the trading logic on and off. The bot will continue to update the UI but will not place any new trades while paused.
  * `|n` - **New Scan:** Immediately sells any open position and forces the bot to scan for a new stock to trade.
  * `|q` - **Quit:** Gracefully shuts down the bot.

-----

## ⚠️ Disclaimer

This is a powerful trading tool provided for educational and experimental purposes only. Algorithmic trading involves substantial risk and is not suitable for all investors. The creators of this bot are not liable for any financial losses you may incur. Always trade responsibly and never risk more than you are willing to lose. Paper trading is strongly recommended.

```
```
