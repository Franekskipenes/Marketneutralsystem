import os
import time
import json
import requests
import pandas as pd
import numpy as np
import math
from typing import Optional, Tuple

from eth_account import Account
from eth_account.signers.local import LocalAccount
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

from trading_system import generate_leg_signal_and_returns


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

# Pair to trade using CMMA intermarket signal
BASE_ASSET = "BTC"          # BTC leg coin on Hyperliquid
ALT_ASSET = "ETH"           # ALT leg coin on Hyperliquid
TRADE_LEG = "alt"           # "alt" to trade ALT, "btc" to trade BTC

# Indicator params
LOOKBACK = 96
ATR_LOOKBACK = 168
THRESHOLD = 0.5
EXIT_LEVEL = 0.0
TP_PCT = None               # e.g. 0.05 for 5% take-profit; None to disable

# Execution params
TRADE_SIZE_USD = 50.0
TIMEFRAME = "1h"            # Binance kline interval used for signal
LOOP_INTERVAL_SECONDS = 300  # 5 minutes

# Private key (testnet) – set HL_TESTNET_PRIVATE_KEY in your environment
PRIVATE_KEY_ENV = "HL_TESTNET_PRIVATE_KEY"


# ─────────────────────────────────────────────────────────────
# Local position ledger (persistent)
# ─────────────────────────────────────────────────────────────

LEDGER_FILE = os.path.join(os.path.dirname(__file__), "local_positions.json")
POSITION_LEDGER = {}


def _ledger_key(asset: str) -> str:
    try:
        return str(asset).upper()
    except Exception:
        return str(asset)


def load_position_ledger() -> dict:
    global POSITION_LEDGER
    try:
        if os.path.exists(LEDGER_FILE):
            with open(LEDGER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    POSITION_LEDGER = {str(k).upper(): float(v)
                                       for k, v in data.items()}
                else:
                    POSITION_LEDGER = {}
        else:
            POSITION_LEDGER = {}
    except Exception:
        POSITION_LEDGER = {}
    return POSITION_LEDGER


def save_position_ledger() -> None:
    try:
        with open(LEDGER_FILE, "w", encoding="utf-8") as f:
            json.dump(POSITION_LEDGER, f)
    except Exception:
        pass


def get_local_position(asset: str) -> Tuple[bool, float]:
    key = _ledger_key(asset)
    szi = float(POSITION_LEDGER.get(key, 0.0))
    return (abs(szi) > 0.0, szi)


def set_local_position(asset: str, signed_size: float) -> None:
    key = _ledger_key(asset)
    POSITION_LEDGER[key] = float(signed_size)
    save_position_ledger()


def update_local_after_close(asset: str) -> None:
    set_local_position(asset, 0.0)


def update_local_after_open(asset: str, is_buy: bool, size_coin: float) -> None:
    signed = float(size_coin) if is_buy else -float(size_coin)
    set_local_position(asset, signed)


# ─────────────────────────────────────────────────────────────
# Helpers: Data fetching
# ─────────────────────────────────────────────────────────────

BINANCE_INTERVAL = TIMEFRAME


def _binance_symbol(asset: str) -> str:
    return f"{asset}USDT"


def fetch_binance_klines(asset: str, interval: str = BINANCE_INTERVAL, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch OHLC klines from Binance for asset/USDT and compute log returns.
    Returns DataFrame indexed by datetime with columns: open, high, low, close, diff, next_return.
    """
    symbol = _binance_symbol(asset)
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or len(data) == 0:
        # empty
        return pd.DataFrame(columns=["open", "high", "low", "close", "diff", "next_return"])

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = df[["date", "open", "high", "low", "close"]
             ].dropna().set_index("date").sort_index()
    # compute per-bar log return and next bar log return
    out["diff"] = np.log(out["close"]).diff()
    out["next_return"] = out["diff"].shift(-1)
    return out


# ─────────────────────────────────────────────────────────────
# Helpers: Hyperliquid
# ─────────────────────────────────────────────────────────────

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
}


def get_price_from_coingecko(asset: str) -> Optional[float]:
    try:
        cid = COINGECKO_IDS.get(asset)
        if cid is None:
            return None
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={cid}&vs_currencies=usd"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return float(resp.json()[cid]["usd"])
    except Exception:
        return None


def get_mid_price(info: Info, asset: str) -> Optional[float]:
    try:
        # Try SDK mids if available
        mids = info.all_mids()
        if isinstance(mids, dict) and asset in mids:
            return float(mids[asset])
    except Exception:
        pass
    # Fallback to CoinGecko
    return get_price_from_coingecko(asset)


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _iter_positions_from_state(user_state: dict):
    """
    Yield (coin, signed_size) tuples from various possible user_state shapes.
    """
    if not isinstance(user_state, dict):
        return

    containers = []
    for key in ("assetPositions", "positions", "asset_positions"):
        if key in user_state:
            containers.append(user_state.get(key))

    # Some SDK variants may return a list directly
    if not containers and isinstance(user_state, list):
        containers.append(user_state)

    for container in containers:
        if isinstance(container, dict):
            items = container.values()
        elif isinstance(container, list):
            items = container
        else:
            items = []

        for item in items:
            if isinstance(item, dict):
                pos = item.get("position", item)
                coin = (
                    pos.get("coin")
                    or pos.get("name")
                    or pos.get("asset")
                    or item.get("coin")
                    or item.get("name")
                    or item.get("asset")
                )
                szi = (
                    pos.get("szi")
                    if isinstance(pos, dict) and "szi" in pos
                    else pos.get("sz") if isinstance(pos, dict) and "sz" in pos
                    else item.get("szi") if isinstance(item, dict) and "szi" in item
                    else item.get("sz") if isinstance(item, dict) and "sz" in item
                    else 0
                )
                if coin is not None:
                    yield str(coin), _safe_float(szi)


def list_open_positions(info: Info, address: str):
    try:
        us = info.user_state(address)
    except Exception as e:
        print(f"WARN: user_state fetch failed: {e}")
        return []

    out = []
    try:
        for coin, szi in _iter_positions_from_state(us):
            if abs(szi) > 0:
                out.append({"coin": coin, "size": szi})
    except Exception as e:
        print(f"WARN: Failed to parse positions: {e}")
    return out


def get_position(info: Info, address: str, asset: str) -> Tuple[bool, float]:
    """
    Returns (has_position, signed_size). signed_size > 0 for long, < 0 for short, 0 for flat.
    Robust to API response variations.
    """
    try:
        open_positions = list_open_positions(info, address)
        asset_upper = (asset or "").upper()
        for p in open_positions:
            coin = str(p.get("coin", "")).upper()
            if coin == asset_upper:
                szi = _safe_float(p.get("size", 0.0))
                return (abs(szi) > 0, szi)
        # If positions exist but asset not found, provide one-time log of what we see
        if open_positions:
            try:
                summary = ", ".join(
                    [f"{pp['coin']}={pp['size']}" for pp in open_positions])
                print(f"DEBUG: Open positions seen: {summary}")
            except Exception:
                pass
    except Exception as e:
        print(f"WARN: get_position failed: {e}")
    return (False, 0.0)


def place_limit(exchange: Exchange, asset: str, is_buy: bool, size: float, price: float):
    # Deprecated: using market orders only
    raise NotImplementedError("Limit orders are disabled; using market only.")


def place_market_open(exchange: Exchange, asset: str, is_buy: bool, size: float):
    """Open a position at market for the given asset and side."""
    try:
        sz = float(size)
        if sz <= 0:
            print(f"ERROR: Invalid market size {size}")
            return None
        # market_open(asset, is_buy, size, optional_price=None, slippage_tolerance)
        res = exchange.market_open(asset, is_buy, sz, None, 0.01)
        # On success, update local ledger
        try:
            update_local_after_open(asset, is_buy, sz)
        except Exception:
            pass
        return res
    except Exception as e:
        print(f"ERROR: Failed market_open: {e}")
        return None


def place_market_close(exchange: Exchange, asset: str):
    """Close any open position for the asset at market."""
    try:
        res = exchange.market_close(asset)
        try:
            update_local_after_close(asset)
        except Exception:
            pass
        return res
    except Exception as e:
        print(f"ERROR: Failed market_close: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Helpers: Signal/Position reconciliation
# ─────────────────────────────────────────────────────────────


def signal_mode(sig: pd.Series) -> int:
    """
    Map last signal value to mode: -1 (short), 0 (flat), 1 (long).
    Returns 0 if series is empty/NaN.
    """
    try:
        if sig is None or len(sig) == 0:
            return 0
        v = int(sig.iloc[-1])
        return 1 if v > 0 else (-1 if v < 0 else 0)
    except Exception:
        return 0


def ensure_target_position(info: Info,
                           exchange: Exchange,
                           address: str,
                           asset: str,
                           target_mode: int,
                           price: float,
                           trade_size_usd: float) -> bool:
    """
    Ensure live position matches desired target_mode (-1 short, 0 flat, 1 long).
    Returns True if any order was sent, False if already aligned.
    """
    # Use local ledger for position state
    has_pos, szi = get_local_position(asset)
    in_long = has_pos and szi > 0
    in_short = has_pos and szi < 0

    # No valid price -> cannot act
    if price is None or price <= 0:
        print("WARN: No valid price available; skipping trade sync.")
        return False

    # Determine allowed size precision and quantize the order size to avoid rounding errors
    try:
        meta = info.meta()
        universe = meta.get("universe", []) if isinstance(meta, dict) else []
        sz_decimals = 3
        for a in universe:
            if a.get("name") == asset:
                if a.get("szDecimals") is not None:
                    sz_decimals = int(a.get("szDecimals"))
                break
    except Exception:
        sz_decimals = 3

    size_step = 10 ** (-sz_decimals)
    raw_size_coin = trade_size_usd / price
    # Floor to the nearest lot step; ensure at least one step
    size_coin = max(size_step, math.floor(
        raw_size_coin / size_step) * size_step)
    # No price quantization needed for market execution

    if target_mode == 1:  # want long
        if in_short:
            res_close = place_market_close(exchange, asset)
            print(f"ACTION: Close SHORT {asset}: {json.dumps(res_close)}")
            time.sleep(0.5)
        if not in_long:
            res = place_market_open(exchange, asset, True, size_coin)
            print(
                f"ACTION: Enter LONG {asset} sz={size_coin:.6f} @mkt: {json.dumps(res)}")
            return True
        return False

    if target_mode == -1:  # want short
        if in_long:
            res_close = place_market_close(exchange, asset)
            print(f"ACTION: Close LONG {asset}: {json.dumps(res_close)}")
            time.sleep(0.5)
        if not in_short:
            res = place_market_open(exchange, asset, False, size_coin)
            print(
                f"ACTION: Enter SHORT {asset} sz={size_coin:.6f} @mkt: {json.dumps(res)}")
            return True
        return False

    # target_mode == 0 -> want flat
    if has_pos:
        res = place_market_close(exchange, asset)
        print(f"ACTION: Exit to FLAT {asset}: {json.dumps(res)}")
        return True
    return False


# ─────────────────────────────────────────────────────────────
# Main Bot
# ─────────────────────────────────────────────────────────────


def run_bot():
    pk = os.getenv(PRIVATE_KEY_ENV)
    if not pk:
        raise RuntimeError(f"Missing {PRIVATE_KEY_ENV} environment variable")
    if not pk.startswith("0x"):
        pk = "0x" + pk

    account: LocalAccount = Account.from_key(pk)
    # Load local position ledger on startup
    load_position_ledger()
    # Use mainnet API URL (fallback to official URL if constant missing)
    api_url = getattr(constants, "MAINNET_API_URL",
                      "https://api.hyperliquid.xyz")
    info = Info(api_url, skip_ws=True)
    exchange = Exchange(account, api_url)

    # Connectivity check (no explicit login needed in current SDK)
    try:
        _ = info.user_state(account.address)
        print("LOG: Connected to Hyperliquid Mainnet API.")
    except Exception as e:
        print(f"WARN: Could not fetch user state initially: {e}")

    print(
        f"Bot starting on Hyperliquid Mainnet. Pair: {BASE_ASSET}/{ALT_ASSET}, leg: {TRADE_LEG}")

    while True:
        try:
            # 1) Fetch OHLC for both assets
            btc_df = fetch_binance_klines(
                BASE_ASSET, interval=BINANCE_INTERVAL, limit=1000)
            alt_df = fetch_binance_klines(
                ALT_ASSET, interval=BINANCE_INTERVAL, limit=1000)

            # 2) Build signal and bar-level returns for the chosen leg
            sig, raw, net = generate_leg_signal_and_returns(
                btc_df, alt_df,
                lookback=LOOKBACK,
                threshold=THRESHOLD,
                atr_lookback=ATR_LOOKBACK,
                leg=TRADE_LEG,
                fee_rate=0.0,
                exit_level=EXIT_LEVEL,
                tp_pct=TP_PCT,
            )

            if len(sig) < 1:
                print("LOG: Not enough bars to act yet.")
                time.sleep(LOOP_INTERVAL_SECONDS)
                continue

            mode = signal_mode(sig)  # -1 short, 0 flat, 1 long

            # Asset we trade on HL for this leg
            trade_asset = ALT_ASSET if TRADE_LEG == "alt" else BASE_ASSET

            # 3) Determine current price for sizing
            price = get_mid_price(info, trade_asset)
            if price is None:
                # as a fallback, use the last close from the fetched series
                series_df = alt_df if trade_asset == ALT_ASSET else btc_df
                price = float(series_df["close"].iloc[-1])

            # Also get prices for both legs to support hedged execution
            alt_price = get_mid_price(info, ALT_ASSET)
            if alt_price is None:
                alt_price = float(alt_df["close"].iloc[-1])
            btc_price = get_mid_price(info, BASE_ASSET)
            if btc_price is None:
                btc_price = float(btc_df["close"].iloc[-1])

            # Map pair targets from mode: +1 => ALT long / BTC short, -1 => ALT short / BTC long, 0 => FLAT both
            if mode == 1:
                target_alt = 1
                target_btc = -1
            elif mode == -1:
                target_alt = -1
                target_btc = 1
            else:
                target_alt = 0
                target_btc = 0

            # Resolve primary (configured leg) and secondary (hedge) targets/prices
            if trade_asset == ALT_ASSET:
                primary_asset = ALT_ASSET
                primary_target = target_alt
                primary_price = alt_price
                secondary_asset = BASE_ASSET
                secondary_target = target_btc
                secondary_price = btc_price
            else:
                primary_asset = BASE_ASSET
                primary_target = target_btc
                primary_price = btc_price
                secondary_asset = ALT_ASSET
                secondary_target = target_alt
                secondary_price = alt_price

            # 4) Check position and reconcile to mode (use local ledger)
            has_pos, szi = get_local_position(trade_asset)
            state = "LONG" if (has_pos and szi > 0) else (
                "SHORT" if (has_pos and szi < 0) else "FLAT")
            mode_desc = "LONG" if mode == 1 else (
                "FLAT" if mode == 0 else "SHORT")
            print(f"MODE: {mode_desc} | STATE: {state} | price={price:.2f}")

            # 4a) Log current open positions for both legs (local + parsed)
            has_btc, szi_btc = get_local_position(BASE_ASSET)
            has_alt, szi_alt = get_local_position(ALT_ASSET)
            open_pos_list = [
                {"coin": BASE_ASSET, "size": szi_btc},
                {"coin": ALT_ASSET, "size": szi_alt},
            ]
            try:
                summary = ", ".join(
                    [f"{p['coin']}={p['size']:.6f}" for p in open_pos_list]) or "NONE"
                print(
                    f"POS: {BASE_ASSET}={szi_btc:.6f} | {ALT_ASSET}={szi_alt:.6f} | ALL: {summary}")
            except Exception:
                print(
                    f"POS: {BASE_ASSET}={szi_btc} | {ALT_ASSET}={szi_alt} | ALL: {open_pos_list}")

            # 4b) Reconcile both legs every cycle
            acted_primary = ensure_target_position(
                info=info,
                exchange=exchange,
                address=account.address,
                asset=primary_asset,
                target_mode=primary_target,
                price=primary_price,
                trade_size_usd=TRADE_SIZE_USD,
            )
            acted_secondary = ensure_target_position(
                info=info,
                exchange=exchange,
                address=account.address,
                asset=secondary_asset,
                target_mode=secondary_target,
                price=secondary_price,
                trade_size_usd=TRADE_SIZE_USD,
            )
            if not acted_primary and not acted_secondary:
                print("LOG: Both legs already aligned; no action.")

        except Exception as e:
            print(f"ERROR: {e}")

        print(f"--- Sleeping {LOOP_INTERVAL_SECONDS}s ---")
        time.sleep(LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_bot()
