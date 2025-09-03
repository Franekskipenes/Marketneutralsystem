# Hyperliquid CMMA Bot

A cryptocurrency trading bot that uses intermarket strategy on Hyperliquid.

## Features

- **CMMA Strategy**: Uses the difference between ALT and BTC CMMA indicators to generate trading signals
- **Intermarket Trading**: Can trade either the ALT leg or BTC leg based on relative strength
- **Take Profit**: Optional take-profit functionality to lock in gains
- **Real-time Data**: Fetches live data from Binance for signal generation
- **Automated Trading**: Executes trades automatically on Hyperliquid testnet
- **Precision Handling**: Automatic rounding and minimum size validation for Hyperliquid API compatibility
- **Automatic Size Adjustment**: Automatically retries with larger sizes if orders are rejected due to size requirements
- **Correct API Format**: Sends size and price as strings as required by Hyperliquid API

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

You need to set your Hyperliquid private key:

**Windows:**
```cmd
set HL_PRIVATE_KEY=your_private_key_here
```

**Linux/Mac:**
```bash
export HL_PRIVATE_KEY=your_private_key_here
```

**Or create a .env file:**
```
HL_PRIVATE_KEY=your_private_key_here
```

### 3. Configuration

Edit the configuration section in `hyperliquid_cmma_bot.py`:

```python
# Pair to trade using CMMA intermarket signal
BASE_ASSET = "BTC"          # BTC leg coin on Hyperliquid
ALT_ASSET = "ETH"           # ALT leg coin on Hyperliquid
TRADE_LEG = "alt"           # "alt" to trade ALT, "btc" to trade BTC

# Indicator params
LOOKBACK = 96               # CMMA lookback period
ATR_LOOKBACK = 168         # ATR lookback period
THRESHOLD = 0.5            # Signal threshold
EXIT_LEVEL = 0.0           # Exit level (0 = exit at 0)
TP_PCT = None              # Take profit percentage (None to disable)

# Execution params
TRADE_SIZE_USD = 100.0     # Trade size in USD (reasonable for testnet)
TIMEFRAME = "1h"           # Binance kline interval
LOOP_INTERVAL_SECONDS = 300 # Bot loop interval (5 minutes)

# Precision settings for Hyperliquid
MIN_SIZE_BTC = 0.001       # Minimum BTC order size (reasonable default)
MIN_SIZE_ALT = 0.01        # Minimum ALT order size (reasonable default)
PRICE_PRECISION = 2        # Price decimal places
SIZE_PRECISION = 6         # Size decimal places


```

## Usage

### Run the Bot

```bash
python hyperliquid_cmma_bot.py
```

### How It Works

1. **Data Fetching**: Bot fetches OHLC data from Binance for both BTC and ALT assets
2. **Signal Generation**: Calculates CMMA indicators and generates intermarket signals
3. **Position Management**: Checks current Hyperliquid positions and manages entries/exits
4. **Trade Execution**: Places limit orders on Hyperliquid based on signals

### Trading Logic

- **Long Signal (1)**: Enter long position when indicator > threshold
- **Short Signal (-1)**: Enter short position when indicator < -threshold  
- **Flat Signal (0)**: Close positions when indicator crosses exit level
- **Take Profit**: Optional exit when cumulative profit reaches specified percentage

### Minimum Trade Sizes

The bot enforces minimum trade sizes to comply with Hyperliquid's requirements:

- **Configurable Minimums**: Set reasonable defaults (0.001 BTC, 0.01 ALT)
- **Size Validation**: Automatically checks if trade sizes meet requirements
- **Clear Feedback**: Provides warnings when sizes are insufficient

The default `TRADE_SIZE_USD = 100.0` is suitable for testnet accounts with limited funds.

### API Format Requirements

Hyperliquid requires specific formatting for order parameters:

- **Size and Price**: Must be sent as strings, not floats
- **Trailing Zeros**: Automatically removed (e.g., "0.1000" → "0.1")
- **Precision**: Respects asset-specific decimal rules
- **Example**: `"s": "0.021836", "p": "4579.60"`

The bot automatically handles this formatting to ensure compatibility.

## Safety Features

- **Testnet Only**: Currently configured for Hyperliquid testnet
- **Error Handling**: Graceful error handling with retry logic
- **Position Checks**: Verifies current positions before placing new orders
- **Logging**: Comprehensive logging of all actions and errors

## Disclaimer

This bot is for educational and testing purposes only. Trading cryptocurrencies involves significant risk. Always test thoroughly on testnet before using real funds.

## Requirements

- Python 3.8+
- Hyperliquid account
- Private key for authentication
- Internet connection for data fetching and trading


