"""
AI-Powered Stock Trading Bot with Market Scanner & Gemini API (v6.2 - Connection Fix)

This script implements a dynamic stock trading bot that uses Google's Gemini LLM
for market analysis, asset selection, and trade execution decisions. It now includes
stop-loss/take-profit orders, P/L tracking, news analysis, and manual controls.

This version adds a self-healing mechanism to automatically reconnect after network disruptions.

Key Features:
- Auto-Reconnect: Bot now survives network interruptions (like computer sleep) without crashing.
- Stop-Loss & Take-Profit: Automatically places bracket orders to manage risk.
- P/L Tracking: Displays real-time unrealized and session-totaled profit/loss.
- Smarter AI Scanning: AI now analyzes and ranks the top 3 most promising assets.
- News Sentiment Analysis: AI incorporates the latest news headlines in its decisions.
- Manual Keyboard Controls: Pause (p), Sell (s), New Scan (n), and Quit (q).
"""

import os
import time
import json
import csv
import getpass
import requests
import random
import threading
from pathlib import Path
from collections import deque
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.layout import Layout
import asciichartpy
from datetime import datetime, timedelta
from pynput import keyboard

# Load environment variables from .env file
load_dotenv()

# === Rich Console Initialization ===
console = Console()

# === Bot State & Manual Controls ===
class BotState:
    def __init__(self):
        self.paused = False
        self.force_sell = False
        self.force_scan = False
        self.should_exit = False
        self.session_pl = 0.0
        self.current_trading_asset = None
        self.position_entry_price = 0.0

bot_state = BotState()

def on_press(key):
    """Handles keyboard input for manual controls."""
    try:
        if key.char == 'p':
            bot_state.paused = not bot_state.paused
            log_messages.append(f"[bold yellow]Trading {'PAUSED' if bot_state.paused else 'RESUMED'} by user.[/bold yellow]")
        elif key.char == 's':
            bot_state.force_sell = True
            log_messages.append("[bold red]Manual SELL triggered by user![/bold red]")
        elif key.char == 'n':
            bot_state.force_scan = True
            log_messages.append("[bold cyan]Manual asset scan triggered by user![/bold cyan]")
        elif key.char == 'q':
            bot_state.should_exit = True
            log_messages.append("[bold]Exit signal received. Shutting down gracefully...[/bold]")
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

# === Configuration & API Key Management ===
def get_api_key(key_name: str, is_secret: bool = True) -> str:
    key = os.environ.get(key_name)
    if not key:
        prompt_text = f"Enter your {key_name}: "
        try:
            key = getpass.getpass(prompt_text) if is_secret else input(prompt_text)
        except (KeyboardInterrupt, EOFError):
            console.print("[bold red]Operation cancelled.[/bold red]"); exit()
    if not key: console.print(f"[bold red]Error: {key_name} is required.[/bold red]"); exit()
    return key

# --- API Credentials ---
GOOGLE_API_KEY = get_api_key("GOOGLE_API_KEY")
APCA_API_KEY_ID = get_api_key("APCA_API_KEY_ID", is_secret=False)
APCA_API_SECRET_KEY = get_api_key("APCA_API_SECRET_KEY")
APCA_PAPER = True

# --- Trading Parameters ---
SCAN_INTERVAL_MINUTES = 15
TRADE_INTERVAL_SECONDS = 15
MAX_HISTORY_LENGTH = 100
PRICE_WINDOW = 20

# --- Risk Management ---
TRADE_SIZE_PERCENT = 0.90
TAKE_PROFIT_PERCENT = 5.0
STOP_LOSS_PERCENT = 2.5

# === Data & Log Storage ===
price_history = deque(maxlen=MAX_HISTORY_LENGTH)
log_messages = deque(maxlen=10)
price_fetch_failures = 0
TRADE_LOG_FILE = Path("trades.csv")

# Initialize API clients globally so they can be re-initialized in main
trading_client = None
stock_data_client = None

def initialize_api_clients():
    """Initializes or re-initializes the Alpaca API clients."""
    global trading_client, stock_data_client
    try:
        trading_client = TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=APCA_PAPER)
        stock_data_client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
        # Test connection
        trading_client.get_account()
        log_messages.append("[bold green]Successfully connected to Alpaca API.[/bold green]")
        return True
    except Exception as e:
        console.print(f"[bold red]Fatal Error: Could not connect to Alpaca API: {e}[/bold red]")
        return False


def setup_trade_log():
    """Creates the trade log CSV file if it doesn't exist."""
    if not TRADE_LOG_FILE.exists():
        with open(TRADE_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Symbol", "Decision", "Amount", "Price", "OrderID", "Reasoning"])
        log_messages.append(f"Trade log created at: {TRADE_LOG_FILE}")

def log_trade_to_csv(timestamp, symbol, decision, amount, price, order_id, reasoning):
    """Appends a trade record to the CSV log file."""
    with open(TRADE_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, symbol, decision, f"{amount:.8f}", f"{price:.2f}", order_id, reasoning])

# === Core Bot Logic ===

def get_tradable_assets():
    """Fetches a list of tradable stocks efficiently."""
    log_messages.append("Scanning for tradable assets...")
    search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status='active')
    all_assets = trading_client.get_all_assets(search_params)
    eligible_assets = [a for a in all_assets if a.marginable and a.shortable and a.tradable]
    asset_sample = random.sample(eligible_assets, min(len(eligible_assets), 200))
    asset_symbols = [a.symbol for a in asset_sample]
    tradable_assets = []
    try:
        latest_quotes = stock_data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=asset_symbols))
        for asset in asset_sample:
            quote = latest_quotes.get(asset.symbol)
            if quote and quote.ask_price is not None and 5 < float(quote.ask_price) < 500:
                tradable_assets.append(asset)
    except Exception as e:
        log_messages.append(f"[yellow]Could not fetch batch quotes: {e}[/yellow]")
    log_messages.append(f"Found {len(tradable_assets)} potential assets.")
    return tradable_assets

def choose_asset_to_trade(assets):
    """Uses Gemini to pick the most promising asset from a list."""
    log_messages.append("Asking AI to rank top 3 assets...")
    asset_list_str = ", ".join([f"{a.symbol} ({a.name})" for a in assets])
    prompt = f"You are a market analyst. From the following list, identify the top 3 stocks with the highest volatility and short-term profit potential. Provide a brief reason for each choice.\n\nList: {asset_list_str}\n\nYour response must be in the specified JSON format."
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GOOGLE_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json", "responseSchema": {"type": "OBJECT", "properties": {"top_picks": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"symbol": {"type": "STRING"}, "reasoning": {"type": "STRING"}}, "required": ["symbol", "reasoning"]}}}, "required": ["top_picks"]}}}
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        picks = json.loads(result['candidates'][0]['content']['parts'][0]['text'])['top_picks']
        log_messages.append(f"AI Analysis: 1. {picks[0]['symbol']} ({picks[0]['reasoning']})")
        chosen_symbol = picks[0]['symbol']
        for asset in assets:
            if asset.symbol == chosen_symbol:
                log_messages.append(f"[bold green]AI has chosen to trade: {asset.symbol}[/bold green]")
                return asset
        return random.choice(assets)
    except Exception as e:
        log_messages.append(f"[yellow]AI asset selection failed: {e}. Choosing random.[/yellow]")
        return random.choice(assets)

def get_news(asset_symbol):
    """Fetches recent news headlines for an asset."""
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=2)
        news = stock_data_client.get_stock_news(
            symbol_or_symbols=[asset_symbol], 
            start=start_time, 
            end=end_time, 
            limit=5
        )
        return [item.headline for item in news.get(asset_symbol, [])]
    except Exception:
        return []

def get_account_details(asset_symbol):
    """Fetches account details and position info."""
    try:
        account = trading_client.get_account()
        buying_power = float(account.buying_power)
        equity = float(account.equity)
        position = None
        try:
            position = trading_client.get_open_position(asset_symbol)
        except APIError as e:
            if e.status_code != 404: raise e
        return buying_power, position, equity
    except Exception as e:
        log_messages.append(f"[yellow]Warning: Could not fetch account details: {e}[/yellow]")
        return 0.0, None, 0.0

def get_current_price(asset_symbol):
    """Fetch the latest price for a given stock symbol."""
    global price_fetch_failures
    try:
        request_params = StockLatestQuoteRequest(symbol_or_symbols=asset_symbol)
        latest_quote = stock_data_client.get_stock_latest_quote(request_params)
        if asset_symbol in latest_quote and latest_quote[asset_symbol]:
            price = float(latest_quote[asset_symbol].bid_price)
            price_fetch_failures = 0
            return price
        price_fetch_failures += 1
        return list(price_history)[-1] if price_history else 0
    except Exception:
        price_fetch_failures += 1
        return list(price_history)[-1] if price_history else 0

def get_historical_data(asset_symbol):
    """Fetches the last 3 days of hourly bar data for an asset."""
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=3)
        request_params = StockBarsRequest(
            symbol_or_symbols=[asset_symbol],
            timeframe=TimeFrame.Hour,
            start=start_time,
            end=end_time
        )
        bars = stock_data_client.get_stock_bars(request_params)
        return bars.df.reset_index().to_dict('records')
    except Exception as e:
        log_messages.append(f"[yellow]Could not fetch historical data for {asset_symbol}: {e}[/yellow]")
        return []

def call_llm(asset_symbol, buying_power, position, historical_data, news_headlines):
    """Get trading advice from the Gemini LLM."""
    if len(price_history) < PRICE_WINDOW:
        return "Hold", f"Waiting for {PRICE_WINDOW - len(price_history)} more data points."
    
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GOOGLE_API_KEY}"
    price_window_data = list(price_history)[-PRICE_WINDOW:]
    price_summary = {"current_price": price_window_data[-1], "min": min(price_window_data), "max": max(price_window_data)}
    
    prompt = f"""You are a data-driven financial analyst. Your goal is to maximize profit by identifying high-probability trades. Holding is a valid strategy.
Analyze the following data for {asset_symbol}:
- Recent Price Ticks: {json.dumps(price_summary)}
- Historical Data (last 3 days, hourly): {json.dumps(historical_data, default=str)}
- Latest News Headlines: {json.dumps(news_headlines)}
- My Current Position: {position.qty if position else 0} shares
- My Buying Power: ${buying_power:,.2f}
Your decision must be in the required JSON format.
"""
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json", "responseSchema": {"type": "OBJECT", "properties": {"decision": {"type": "STRING", "enum": ["Buy", "Sell", "Hold"]}, "reasoning": {"type": "STRING"}}, "required": ["decision", "reasoning"]}, "temperature": 0.4}}

    try:
        response = requests.post(API_URL, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        if 'candidates' in result and result['candidates']:
            content = result['candidates'][0]['content']['parts'][0]['text']
            llm_response = json.loads(content)
            return llm_response.get('decision', 'Hold'), llm_response.get('reasoning', 'No reasoning.')
        return "Hold", f"Invalid Gemini response: {result.get('promptFeedback', '')}"
    except Exception as e:
        return "Hold", f"Gemini API error: {e}"

def execute_trade(decision, asset, buying_power, position, current_price, reasoning):
    """Executes a trade on Alpaca and logs it."""
    asset_symbol = asset.symbol
    if decision == "Hold": return
    log_messages.append(f"Decision: [cyan]{decision} {asset_symbol}[/cyan].")
    try:
        if decision == "Buy" and position is None:
            usd_to_spend = round(buying_power * TRADE_SIZE_PERCENT, 2)
            if usd_to_spend < 1.0:
                log_messages.append(f"[yellow]Skipped Buy: Trade size (${usd_to_spend:.2f}) is below minimum.[/yellow]")
                return
            
            order_data = MarketOrderRequest(
                symbol=asset_symbol,
                notional=usd_to_spend,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(current_price * (1 + TAKE_PROFIT_PERCENT / 100), 2)),
                stop_loss=StopLossRequest(stop_price=round(current_price * (1 - STOP_LOSS_PERCENT / 100), 2))
            )
            order = trading_client.submit_order(order_data=order_data)
            log_trade_to_csv(time.strftime('%Y-%m-%d %H:%M:%S'), asset_symbol, "Buy", usd_to_spend, current_price, order.id, reasoning)
            log_messages.append(f"[green]Submitted BUY bracket order for ${usd_to_spend:.2f} of {asset_symbol}.[/green]")
        
        elif decision == "Sell" and position is not None:
            trading_client.close_position(asset_symbol)
            realized_pl = float(position.unrealized_pl)
            bot_state.session_pl += realized_pl
            log_messages.append(f"[red]AI initiated SELL for {position.qty} shares of {asset_symbol}. Realized P/L: ${realized_pl:,.2f}[/red]")

    except APIError as e:
        log_messages.append(f"[bold red]Alpaca API error on trade: {e.message}[/bold red]")
    except Exception as e:
        log_messages.append(f"[bold red]Trade execution error: {e}[/bold red]")

# === UI Generation Functions ===
def make_layout() -> Layout:
    layout = Layout(name="root")
    layout.split(Layout(name="header", size=3), Layout(ratio=1, name="main"), Layout(size=10, name="footer"))
    layout["main"].split_row(Layout(name="side"), Layout(name="body", ratio=2))
    return layout

def generate_header(asset_symbol) -> Panel:
    title = "[bold magenta]AI Stock Trading Bot[/] [dim]v6.2 - Connection Fix[/]"
    trading_info = f"[bold]{asset_symbol or 'SCANNING...'}[/] | [yellow]{'PAPER' if APCA_PAPER else 'LIVE'}[/]"
    grid = Table.grid(expand=True); grid.add_column(justify="center", ratio=1); grid.add_column(justify="right")
    grid.add_row(title, trading_info)
    return Panel(grid, style="white on blue")

def generate_status_panel(asset_symbol, current_price, buying_power, position, decision, reasoning, equity) -> Panel:
    color = {"Buy": "green", "Sell": "red", "Hold": "yellow"}.get(decision, "white")
    
    asset_balance = float(position.qty) if position else 0.0
    asset_value_usd = asset_balance * current_price
    unrealized_pl = float(position.unrealized_pl) if position else 0.0
    pl_color = "green" if unrealized_pl >= 0 else "red"

    table = Table.grid(expand=True, padding=(0, 1)); table.add_column(justify="left"); table.add_column(justify="right")
    table.add_row("[bold]Current Price:[/]", f"[bold yellow]${current_price:,.2f}[/]")
    table.add_row("[bold]Buying Power:[/]", f"${buying_power:,.2f} USD")
    table.add_row(f"[bold]{asset_symbol} Holdings:[/]", f"{asset_balance:.4f} Shares (${asset_value_usd:,.2f})")
    table.add_row(f"[bold]Unrealized P/L:[/]", f"[bold {pl_color}]${unrealized_pl:,.2f}[/bold {pl_color}]")
    table.add_row("[bold]Session P/L:[/]", f"[bold {'green' if bot_state.session_pl >= 0 else 'red'}]${bot_state.session_pl:,.2f}[/bold {'green' if bot_state.session_pl >= 0 else 'red'}]")
    table.add_row("[bold]Portfolio Value:[/]", f"[bold magenta]${equity:,.2f}[/]")
    table.add_row("-" * 25, "-" * 25)
    table.add_row("[bold]LLM Decision:[/]", f"[bold {color}]{decision}[/]")
    table.add_row(Text("LLM Reasoning:", overflow="fold"), f"[italic]{reasoning}[/]")
    table.add_row("-" * 25, "-" * 25)
    table.add_row("[bold]Status:[/]", "[bold red]PAUSED[/bold red]" if bot_state.paused else "[bold green]RUNNING[/bold green]")
    return Panel(table, title="[bold]Bot Status[/bold]", border_style="blue")

def generate_chart_panel(asset_symbol) -> Panel:
    prices = list(price_history)
    title = f"[bold]{asset_symbol} Price Chart[/bold]"
    if not prices: return Panel(Text("Waiting for price data...", justify="center"), title=title, border_style="green")
    chart_height = console.height - 3 - 10 - 4
    if chart_height <= 3: return Panel(Text("Terminal too small.", justify="center"), title=title, border_style="red")
    try:
        chart_content = asciichartpy.plot(prices, {'height': chart_height})
        return Panel(Text(chart_content, justify="left"), title=title, border_style="green")
    except Exception as e:
        return Panel(Text(f"Chart Error: {e}", justify="center"), title=title, border_style="red")

def generate_log_panel() -> Panel:
    controls = "Controls: [bold](s)[/] Sell Position | [bold](p)[/] Pause/Resume | [bold](n)[/] New Scan | [bold](q)[/] Quit"
    log_renderable = Text("\n".join(log_messages), justify="left")
    
    log_table = Table.grid(expand=True)
    log_table.add_row(log_renderable)
    log_table.add_row() # Spacer
    log_table.add_row(controls)
    return Panel(log_table, title="[bold]Event Log[/bold]", border_style="yellow")

def show_startup_animation():
    """Displays an ASCII art and status message on startup."""
    rocket_art = "[bold magenta]\n          / \\\n         / _ \\\n        | / \\ |\n        | | | |\n       _|_|_|_|_\n      /   / \\   \\\n     /   /   \\   \\\n    /   /     \\   \\\n   /   /       \\   \\\n  /___/_________\\___\\\n[/bold magenta]"
    console.print(rocket_art); console.print("[bold green]Initializing AI Trading Bot...[/bold green]", justify="center")

def main():
    setup_trade_log()
    layout = make_layout()
    last_scan_time = 0
    
    show_startup_animation()
    if not initialize_api_clients(): return

    with console.status("[bold green]Scanning markets for opportunities...", spinner="dots") as status:
        assets = get_tradable_assets()
        if assets:
            status.update("[bold green]Asking AI to select the best asset...")
            bot_state.current_trading_asset = choose_asset_to_trade(assets)
            price_history.clear()
            last_scan_time = time.time()
        else:
            console.print("[red]Could not find any assets to trade on startup. Exiting.[/red]"); return

    with Live(layout, screen=True, redirect_stderr=False, auto_refresh=False) as live:
        try:
            while not bot_state.should_exit:
                try:
                    # --- Handle Manual Controls ---
                    if bot_state.force_scan:
                        trading_client.close_all_positions(cancel_orders=True)
                        log_messages.append("All positions closed for new scan.")
                        bot_state.force_scan = False
                        last_scan_time = 0; continue
                    
                    if bot_state.force_sell:
                        trading_client.close_position(bot_state.current_trading_asset.symbol)
                        log_messages.append(f"Position in {bot_state.current_trading_asset.symbol} closed by user.")
                        bot_state.force_sell = False
                    
                    # --- Asset Scanning Loop ---
                    if time.time() - last_scan_time > SCAN_INTERVAL_MINUTES * 60:
                        with console.status("[bold green]Scanning for new assets...", spinner="dots"):
                            assets = get_tradable_assets()
                            if assets:
                                bot_state.current_trading_asset = choose_asset_to_trade(assets)
                                price_history.clear()
                                last_scan_time = time.time()
                            else:
                                log_messages.append("[red]No tradable assets found. Retrying scan later.[/red]"); time.sleep(60); continue

                    # --- Trading Loop ---
                    asset = bot_state.current_trading_asset
                    current_price = get_current_price(asset.symbol)
                    if current_price == 0 or price_fetch_failures >= 5:
                        log_messages.append(f"[red]Price fetch failed for {asset.symbol}. Pausing.[/red]"); time.sleep(TRADE_INTERVAL_SECONDS); continue
                    
                    price_history.append(current_price)
                    
                    buying_power, position, equity = get_account_details(asset.symbol)
                    
                    decision, reasoning = ("Hold", "Trading is paused.") if bot_state.paused else call_llm(asset.symbol, buying_power, position, get_historical_data(asset.symbol), get_news(asset.symbol))
                    
                    if not bot_state.paused:
                        execute_trade(decision, asset, buying_power, position, current_price, reasoning)
                    
                    time.sleep(2)
                    final_buying_power, final_position, final_equity = get_account_details(asset.symbol)
                    
                    # Update UI components
                    layout["header"].update(generate_header(asset.symbol))
                    layout["side"].update(generate_status_panel(asset.symbol, current_price, final_buying_power, final_position, decision, reasoning, final_equity))
                    layout["body"].update(generate_chart_panel(asset.symbol))
                    layout["footer"].update(generate_log_panel())
                    live.update(layout, refresh=True)
                    
                    time.sleep(TRADE_INTERVAL_SECONDS)

                except requests.exceptions.ConnectionError as e:
                    log_messages.append("[bold red]Network connection lost! Attempting to reconnect...[/bold red]")
                    live.update(layout, refresh=True)
                    time.sleep(5)
                    if not initialize_api_clients():
                        log_messages.append("[bold red]Reconnect failed. Please check network and restart.[/bold red]")
                        bot_state.should_exit = True
                    continue

        finally:
            listener.stop()
            console.print("\n[bold cyan]Bot stopped. Goodbye![/bold cyan]")

if __name__ == "__main__":
    # Before running, ensure you have a requirements.txt file with:
    # alpaca-py, rich, python-dotenv, asciichartpy, requests, pynput
    main()
