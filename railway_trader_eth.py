"""
Bollinger Band Breakout - Live Trading on Alpaca (Railway 24/7 deployment)
Ethereum (ETH/USD) version - identical strategy and logic to the BTC bot,
just pointed at a different symbol.

Same strategy and fixes as the BTC trader:
- Bollinger Band breakouts with ATR and RSI filters
- Risk-managed position sizing (1% risk per trade), capped by live buying power
  with a slippage buffer, floor-rounded so we never overshoot balance
- Trend filtering with EMA slopes
- Stop loss / take profit checked every loop
- Crypto orders use TimeInForce.GTC (Alpaca crypto rejects DAY)

Hardened for running non-stop on Railway:
- Logs to stdout only (Railway captures stdout; container disk is ephemeral,
  so a log FILE would just be lost on every redeploy anyway)
- On startup, and after any restart, re-syncs with Alpaca for an already-open
  position on the symbol - state was previously only in memory, so a Railway
  redeploy/restart would silently "forget" an open position and either double
  up or stop managing its SL/TP
- Per-iteration try/except so one bad network call or a transient Alpaca/Yahoo
  error doesn't kill the whole process - only genuine setup failures exit
- Handles SIGTERM (Railway sends this on redeploy/scale-down) for a clean,
  logged shutdown instead of an abrupt kill
- Optional tiny HTTP health-check endpoint on $PORT, since Railway can expect
  a listening port depending on how the service is configured

Install: pip install -r requirements.txt
"""

import math
import os
import signal
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

import logging

# ============================================================================
# LOGGING SETUP - stdout only (Railway captures stdout as logs)
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# On Railway, set these as project Variables (dashboard -> Variables tab),
# not a .env file. load_dotenv() is a harmless no-op if no .env is present.
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET")
BASE_URL = (
    os.getenv("ALPACA_BASE_URL")
    or os.getenv("BASE_URL")
    or os.getenv("ENDPOINT")
    or ""
).rstrip("/")
if BASE_URL.endswith("/v2"):
    BASE_URL = BASE_URL[:-3]

if not API_KEY or not SECRET_KEY:
    logger.error("ERROR: Alpaca API key and secret not set!")
    logger.error("On Railway: Project -> Variables -> add ALPACA_KEY and ALPACA_SECRET.")
    raise EnvironmentError("Missing Alpaca credentials in environment variables")

CONFIG = {
    "data_symbol": "ETH/USD",   # Alpaca market-data symbol format (slash)
    "alpaca_symbol": "ETHUSD",  # Alpaca trading symbol (no slash, used for orders)
    "timeframe": TimeFrame.Hour,
    "lookback_days": 90,
    "bb_period": 25,       # Bollinger Band period (same as BTC bot - see note below)
    "bb_dev": 1.477,       # Bollinger Band deviation (same as BTC bot - see note below)
    "sl_pct": 0.003,
    "tp_pct": 0.01,
    "use_trend_filter": True,
    "trail_pct": 0.005,
    "cooldown_bars": 5,
    "rsi_period": 14,
    "atr_period": 14,
    "trend_ema_period": 50,
    "trend_slope_bars": 10,
    "capital": 10_000.0,        # Placeholder - overwritten at startup with real account equity
    "risk_per_trade_pct": 1.0,
    "leverage": 1,
}

# How often (seconds) the main loop wakes up to check exits/signals
POLL_SECONDS = 60
# How often (seconds) to pull fresh OHLCV data
DATA_REFRESH_SECONDS = 300
# Cap on consecutive loop errors before backing off harder (still never exits)
MAX_BACKOFF_SECONDS = 300

# ============================================================================
# GRACEFUL SHUTDOWN (Railway sends SIGTERM on redeploy / scale-down)
# ============================================================================

shutdown_event = threading.Event()


def _handle_shutdown_signal(signum, frame):
    logger.info(f"Received signal {signum}, will shut down after this loop iteration...")
    shutdown_event.set()


signal.signal(signal.SIGTERM, _handle_shutdown_signal)
signal.signal(signal.SIGINT, _handle_shutdown_signal)

# ============================================================================
# OPTIONAL HEALTH-CHECK SERVER
# Only starts if Railway provides a $PORT. Harmless if unused.
# ============================================================================


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):
        pass  # silence default request logging, we have our own logger


def _maybe_start_health_server():
    port = os.getenv("PORT")
    if not port:
        return
    try:
        server = HTTPServer(("0.0.0.0", int(port)), _HealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Health check server listening on port {port}")
    except Exception as e:
        logger.warning(f"Could not start health check server: {e}")

# ============================================================================
# INDICATOR CALCULATIONS
# ============================================================================


def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df, bb_period, bb_dev):
    df = df.copy()

    mid = df["close"].rolling(bb_period).mean()
    std = df["close"].rolling(bb_period).std()
    df["bb_mid"] = mid
    df["bb_upper"] = mid + bb_dev * std
    df["bb_lower"] = mid - bb_dev * std

    df["trend_ema"] = df["close"].ewm(span=CONFIG["trend_ema_period"], adjust=False).mean()
    df["trend_slope"] = df["trend_ema"].diff(CONFIG["trend_slope_bars"]) / df["trend_ema"].shift(
        CONFIG["trend_slope_bars"]
    )

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(CONFIG["atr_period"]).mean()

    df["rsi"] = rsi(df["close"], CONFIG["rsi_period"])
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    return df.dropna()


def fetch_market_data(data_client, symbol, timeframe, lookback_days, max_attempts=5):
    """Fetch OHLCV bars from Alpaca's own market data API.

    Previously this used yfinance against Yahoo Finance. Yahoo actively rate
    limits / blocks requests coming from cloud provider IP ranges (AWS, GCP,
    Railway, Render, etc.) - it commonly works fine from a home/local IP and
    then silently returns empty data (or 429s) once deployed to a cloud host.
    That would explain a bot that runs with no errors but never sees enough
    data to ever generate a signal. Using Alpaca's own data API sidesteps
    this entirely since it's the same authenticated connection already used
    for trading, and it doesn't distinguish between cloud and local traffic.
    """
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td

    end = _dt.now(_tz.utc)
    start = end - _td(days=lookback_days)

    for attempt in range(1, max_attempts + 1):
        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
            )
            bars = data_client.get_crypto_bars(request)
            df = bars.df

            if df is None or df.empty:
                logger.warning(f"Empty data from Alpaca (attempt {attempt}/{max_attempts})")
            else:
                # bars.df is multi-indexed by (symbol, timestamp) for crypto
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(symbol, level=0)
                df = df.rename(columns={c: c.lower() for c in df.columns})
                return df
        except Exception as e:
            logger.warning(f"Error fetching data from Alpaca (attempt {attempt}/{max_attempts}): {e}")

        if attempt < max_attempts:
            time.sleep(min(2 ** attempt, 30))

    logger.error(f"Failed to fetch data for {symbol} after {max_attempts} attempts")
    return pd.DataFrame()

# ============================================================================
# ALPACA CLIENT
# ============================================================================

trading_client = TradingClient(
    API_KEY,
    SECRET_KEY,
    paper=True,  # flip to False only once you're intentionally trading a live account
    url_override=BASE_URL if BASE_URL else None,
)

# Crypto market data on Alpaca doesn't require paper vs live distinction and
# uses its own endpoint (data.alpaca.markets), separate from trading.
data_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)


def validate_asset(symbol):
    try:
        asset = trading_client.get_asset(symbol)
        if not asset.tradable:
            logger.error(f"Asset '{symbol}' exists but is not tradable on Alpaca")
            raise SystemExit(1)
        logger.info(f"Asset '{symbol}' verified as tradable on Alpaca")
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Asset '{symbol}' not found on Alpaca: {e}")
        raise SystemExit(1)

# ============================================================================
# BOLLINGER BAND STRATEGY
# ============================================================================


class BollingerBandTrader:
    def __init__(self, config):
        self.config = config
        self.symbol = config["alpaca_symbol"]
        self.data_symbol = config["data_symbol"]

        self.position = None
        self.last_signal_index = None
        self.bars_since_close = config["cooldown_bars"]

        self.df_cache = pd.DataFrame()
        self.last_update = None
        self.consecutive_empty_fetches = 0

    def sync_position_with_broker(self):
        """Reconcile in-memory position state with what Alpaca actually holds.

        Critical on Railway: process restarts (redeploys, crashes, scaling
        events) wipe self.position from memory. Without this check, a restart
        while a position is open would make the bot think it's flat and try
        to open a second, doubled-up position - or simply stop managing the
        SL/TP of the position it already has.
        """
        try:
            broker_position = trading_client.get_open_position(self.symbol)
        except APIError as e:
            if e.status_code == 404:
                # Genuinely no open position on the broker - trust local state
                if self.position is not None:
                    logger.warning(
                        "Local state showed an open position but broker has none. "
                        "Clearing local state."
                    )
                    self.position = None
                return
            # Any other API error (auth, rate limit, 5xx, etc.) is NOT the same
            # as "no position" - swallowing it here would risk the bot thinking
            # it's flat when it might not be, and opening a duplicate position.
            # Let it propagate so the caller's error handling/backoff applies.
            logger.error(f"Error checking existing position (status {e.status_code}): {e}")
            raise

        qty = abs(float(broker_position.qty))
        side = "buy" if float(broker_position.qty) > 0 else "sell"
        entry_price = float(broker_position.avg_entry_price)

        if self.position is None:
            logger.warning(
                f"Found an existing open {side.upper()} position on {self.symbol} "
                f"(qty={qty}, avg_entry={entry_price:.2f}) that this process didn't "
                f"place - likely from before a restart. Reconstructing SL/TP from "
                f"config percentages so it keeps being managed."
            )
            sl_dist = max(self.config["sl_pct"] * entry_price, 0.00001)
            tp_dist = sl_dist * 2.0
            if side == "buy":
                stop_loss = entry_price - sl_dist
                take_profit = entry_price + tp_dist
            else:
                stop_loss = entry_price + sl_dist
                take_profit = entry_price - tp_dist

            self.position = {
                "side": side,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "qty": qty,
                "entry_time": datetime.now(),
            }
            self.bars_since_close = 0

    def fetch_recent_data(self):
        df = fetch_market_data(
            data_client, self.data_symbol, self.config["timeframe"], self.config["lookback_days"]
        )

        if df.empty:
            self.consecutive_empty_fetches += 1
            logger.warning(
                f"No data fetched (consecutive empty fetches: {self.consecutive_empty_fetches}). "
                f"No signals can be generated until this resolves."
            )
            return df

        self.consecutive_empty_fetches = 0
        df = add_indicators(df, self.config["bb_period"], self.config["bb_dev"])
        self.df_cache = df
        self.last_update = datetime.now()
        return df

    def get_current_signal(self):
        if len(self.df_cache) < 2:
            return None

        row_prev = self.df_cache.iloc[-2]
        row = self.df_cache.iloc[-1]

        if self.bars_since_close < self.config["cooldown_bars"]:
            self.bars_since_close += 1
            return None

        if row_prev.name == self.last_signal_index:
            return None

        bb_upper = row_prev["bb_upper"]
        bb_lower = row_prev["bb_lower"]
        entry_price = row["open"]
        slope = row_prev.get("trend_slope", 0)

        slope_threshold = self.df_cache["trend_slope"].abs().median() if not self.df_cache.empty else 0

        signal = None
        side = None

        breakout_buy = row_prev["low"] <= bb_lower or row_prev["close"] <= bb_lower
        breakout_sell = row_prev["high"] >= bb_upper or row_prev["close"] >= bb_upper

        block_buy = self.config["use_trend_filter"] and slope < -slope_threshold
        block_sell = self.config["use_trend_filter"] and slope > slope_threshold

        if breakout_buy and not block_buy:
            signal = "BUY"
            side = "buy"
        elif breakout_sell and not block_sell:
            signal = "SELL"
            side = "sell"

        if signal:
            logger.info(f"Signal: {signal} at {entry_price:.5f} (BB Upper: {bb_upper:.5f}, BB Lower: {bb_lower:.5f})")
            self.last_signal_index = row_prev.name
            return {"signal": signal, "side": side, "entry_price": entry_price, "row": row_prev}

        return None

    def calculate_position_size(self, entry_price, stop_loss):
        """1% risk sizing, capped by live buying power with a slippage buffer."""
        risk_amount = (self.config["capital"] * self.config["risk_per_trade_pct"]) / 100
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit <= 0:
            return 0

        position_size = risk_amount / risk_per_unit

        max_position = (self.config["capital"] * self.config["leverage"]) / entry_price
        position_size = min(position_size, max_position)

        # entry_price is the last fetched bar's open, which can be stale by
        # minutes on a poll-based data feed - assume price could move up to
        # 1.5% by the time the order fills, plus a 5% buying-power buffer.
        account_info = self.check_account_status()
        if account_info:
            SLIPPAGE_BUFFER = 1.015
            effective_price = entry_price * SLIPPAGE_BUFFER
            max_affordable = (account_info["buying_power"] * 0.95) / effective_price
            position_size = min(position_size, max_affordable)

        # Floor, not round - never let 8-decimal rounding push us over budget.
        position_size = math.floor(max(position_size, 0) * 1e8) / 1e8
        return position_size

    def place_buy_signal(self, entry_price, row_prev):
        sl_dist = max(self.config["sl_pct"] * entry_price, 0.00001)
        stop_loss = entry_price - sl_dist
        tp_dist = sl_dist * 2.0
        take_profit = entry_price + tp_dist

        position_size = self.calculate_position_size(entry_price, stop_loss)
        if position_size <= 0:
            logger.warning("Position size too small, skipping BUY order")
            return

        logger.info(f"BUY ORDER: Entry={entry_price:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}, Size={position_size}")

        try:
            order = trading_client.submit_order(
                MarketOrderRequest(
                    symbol=self.symbol,
                    qty=position_size,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,  # crypto only supports GTC or IOC, not DAY
                )
            )
            logger.info(f"BUY order submitted: {order.id}")

            self.position = {
                "side": "buy",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "qty": position_size,
                "entry_time": datetime.now(),
            }
            self.bars_since_close = 0

        except Exception as e:
            logger.error(f"Error placing buy order: {e}")

    def place_sell_signal(self, entry_price, row_prev):
        sl_dist = max(self.config["sl_pct"] * entry_price, 0.00001)
        stop_loss = entry_price + sl_dist
        tp_dist = sl_dist * 2.0
        take_profit = entry_price - tp_dist

        position_size = self.calculate_position_size(entry_price, stop_loss)
        if position_size <= 0:
            logger.warning("Position size too small, skipping SELL order")
            return

        logger.info(f"SELL ORDER: Entry={entry_price:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}, Size={position_size}")

        try:
            order = trading_client.submit_order(
                MarketOrderRequest(
                    symbol=self.symbol,
                    qty=position_size,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,  # crypto only supports GTC or IOC, not DAY
                )
            )
            logger.info(f"SELL order submitted: {order.id}")

            self.position = {
                "side": "sell",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "qty": position_size,
                "entry_time": datetime.now(),
            }
            self.bars_since_close = 0

        except Exception as e:
            logger.error(f"Error placing sell order: {e}")

    def check_exit_conditions(self):
        if self.position is None or self.df_cache.empty:
            return

        current_price = self.df_cache.iloc[-1]["close"]
        side = self.position["side"]

        should_exit = False
        reason = ""

        if side == "buy":
            if current_price <= self.position["stop_loss"]:
                should_exit, reason = True, "Stop Loss"
            elif current_price >= self.position["take_profit"]:
                should_exit, reason = True, "Take Profit"
        else:
            if current_price >= self.position["stop_loss"]:
                should_exit, reason = True, "Stop Loss"
            elif current_price <= self.position["take_profit"]:
                should_exit, reason = True, "Take Profit"

        if should_exit:
            self.close_position(reason)

    def close_position(self, reason="Manual"):
        if self.position is None:
            return

        try:
            side = OrderSide.SELL if self.position["side"] == "buy" else OrderSide.BUY

            # Use the broker's actual current qty, not the locally cached
            # order-time qty. Alpaca deducts crypto fees from the asset
            # itself, so a BUY for 0.01487938 BTC can leave slightly less
            # than that actually sitting in the account (e.g. 0.014842181),
            # and closing with the stale, larger qty gets rejected as
            # insufficient balance.
            close_qty = self.position["qty"]
            try:
                live_position = trading_client.get_open_position(self.symbol)
                live_qty = abs(float(live_position.qty))
                if live_qty > 0:
                    # Small extra safety margin below the live figure too,
                    # in case of any further rounding between check and submit.
                    close_qty = math.floor(live_qty * 0.9995 * 1e8) / 1e8
            except Exception as e:
                logger.warning(
                    f"Could not fetch live position qty before closing, "
                    f"falling back to cached qty ({close_qty}): {e}"
                )

            order = trading_client.submit_order(
                MarketOrderRequest(
                    symbol=self.symbol,
                    qty=close_qty,
                    side=side,
                    time_in_force=TimeInForce.GTC,  # crypto only supports GTC or IOC, not DAY
                )
            )

            logger.info(f"Position closed ({reason}): {order.id}")
            self.position = None
            self.bars_since_close = 0

        except Exception as e:
            logger.error(f"Error closing position: {e}")

    def check_account_status(self):
        try:
            account = trading_client.get_account()
            return {
                "status": account.status,
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
            }
        except Exception as e:
            logger.error(f"Error getting account status: {e}")
            return None

# ============================================================================
# MAIN LOOP
# ============================================================================


def run():
    logger.info("=" * 80)
    logger.info(f"Starting Bollinger Band Live Trader (Railway) - {CONFIG['data_symbol']}")
    logger.info(f"Symbol: {CONFIG['data_symbol']} (Alpaca trading symbol: {CONFIG['alpaca_symbol']})")
    logger.info(f"BB Period: {CONFIG['bb_period']}, BB Dev: {CONFIG['bb_dev']}")
    logger.info(f"SL %: {CONFIG['sl_pct']}, Risk per trade: {CONFIG['risk_per_trade_pct']}%")
    logger.info("=" * 80)

    validate_asset(CONFIG["alpaca_symbol"])
    _maybe_start_health_server()

    trader = BollingerBandTrader(CONFIG)

    account_info = trader.check_account_status()
    if not account_info:
        logger.error("Could not connect to Alpaca account. Check API keys. Exiting.")
        sys.exit(1)

    logger.info(f"Account Status: {account_info['status']}")
    logger.info(f"Equity: ${account_info['equity']:.2f}")
    logger.info(f"Buying Power: ${account_info['buying_power']:.2f}")

    trader.config["capital"] = account_info["equity"]
    logger.info(f"Position sizing capital set to live equity: ${trader.config['capital']:.2f}")

    # Pick up any position that already exists on the broker (e.g. left open
    # by a previous run before this container restarted). Retry a few times
    # since a transient API error here must NOT be treated as "no position".
    for attempt in range(1, 4):
        try:
            trader.sync_position_with_broker()
            break
        except APIError as e:
            logger.warning(f"Position sync attempt {attempt}/3 failed: {e}")
            if attempt == 3:
                logger.error("Could not confirm position state after 3 attempts. Exiting so Railway can restart cleanly.")
                sys.exit(1)
            time.sleep(5 * attempt)

    logger.info("Fetching initial market data...")
    trader.fetch_recent_data()

    iteration = 0
    last_full_update = time.time()
    consecutive_errors = 0

    while not shutdown_event.is_set():
        try:
            iteration += 1
            current_time = datetime.now()

            if time.time() - last_full_update > DATA_REFRESH_SECONDS:
                trader.fetch_recent_data()
                last_full_update = time.time()
                logger.info(f"[{current_time.strftime('%H:%M:%S')}] Data refreshed. Latest bars: {len(trader.df_cache)}")

            if trader.position:
                trader.check_exit_conditions()

            if not trader.position:
                signal_result = trader.get_current_signal()
                if signal_result:
                    if signal_result["signal"] == "BUY":
                        trader.place_buy_signal(signal_result["entry_price"], signal_result["row"])
                    elif signal_result["signal"] == "SELL":
                        trader.place_sell_signal(signal_result["entry_price"], signal_result["row"])

            if iteration % 10 == 0:
                pos_status = (
                    f"Position: {trader.position['side'].upper()} @ {trader.position['entry_price']:.5f}"
                    if trader.position
                    else "No position"
                )
                data_status = (
                    f"WARNING: {trader.consecutive_empty_fetches} consecutive empty data fetches - no signals possible"
                    if trader.consecutive_empty_fetches > 0
                    else f"{len(trader.df_cache)} bars cached"
                )
                logger.info(f"[{current_time.strftime('%H:%M:%S')}] {pos_status} | {data_status}")

            consecutive_errors = 0  # reset backoff after a clean iteration
            shutdown_event.wait(POLL_SECONDS)

        except Exception as e:
            consecutive_errors += 1
            backoff = min(POLL_SECONDS * (2 ** consecutive_errors), MAX_BACKOFF_SECONDS)
            logger.error(f"Error in main loop (attempt {consecutive_errors}): {e}", exc_info=True)
            logger.info(f"Backing off {backoff}s before retrying...")
            shutdown_event.wait(backoff)

    logger.info("Shutdown signal received. Exiting loop.")
    logger.info(
        "Note: any open position is left as-is (not force-closed) so it keeps "
        "being managed by the next deploy via sync_position_with_broker()."
    )
    logger.info("Trader stopped gracefully.")


if __name__ == "__main__":
    run()
