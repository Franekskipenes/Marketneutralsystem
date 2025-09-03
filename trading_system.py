import pandas as pd
import numpy as np
import pandas_ta as ta


def prepare_data(file_path: str,
                 start_date: pd.Timestamp | str | None = None,
                 end_date: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """
    Load OHLCV CSV with a 'date' column, set it as index, and add log-return columns.
    Returns a DataFrame with at least: ['open','high','low','close','volume','diff','next_return'].
    Optional date filtering via start_date/end_date.
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    if start_date is not None or end_date is not None:
        start_ts = pd.to_datetime(start_date) if start_date is not None else None
        end_ts = pd.to_datetime(end_date) if end_date is not None else None
        if start_ts is not None and end_ts is not None:
            df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        elif start_ts is not None:
            df = df.loc[start_ts:]
        elif end_ts is not None:
            df = df.loc[:end_ts]

    df = df.dropna()
    df['diff'] = np.log(df['close']).diff()
    df['next_return'] = df['diff'].shift(-1)
    return df


def cmma(ohlc: pd.DataFrame, lookback: int, atr_lookback: int = 168) -> pd.Series:
    """
    CMMA indicator: (close - SMA(close, lookback)) / (ATR(atr_lookback) * sqrt(lookback)).
    Falls back to manual ATR estimate if ta.atr returns None.
    """
    high = ohlc['high'].astype(float)
    low = ohlc['low'].astype(float)
    close = ohlc['close'].astype(float)

    atr = ta.atr(high, low, close, atr_lookback)
    if atr is None:
        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        atr = tr.rolling(atr_lookback).mean()

    ma = close.rolling(lookback).mean()
    ind = (close - ma) / (atr * lookback ** 0.5)
    return ind


def threshold_revert_signal(ind: pd.Series,
                            threshold: float,
                            log_returns: pd.Series | None = None,
                            tp_pct: float | None = None,
                            exit_level: float = 0.0) -> np.ndarray:
    """
    Reversion signal on an indicator series.
    Enter long if ind > threshold, short if ind < -threshold.
    Exit when the indicator crosses back through +/- exit_level or when a take-profit
    threshold in arithmetic pct is reached (tp_pct, e.g., 0.05 = 5%).

    Parameters
    - ind: indicator series
    - threshold: entry threshold (symmetric for long/short)
    - log_returns: per-bar LOG returns aligned to `ind` (required if tp_pct is set)
    - tp_pct: optional arithmetic take-profit level per trade
    - exit_level: indicator level to close the position (default 0.0)

    Returns
    - numpy int array of positions per bar: +1 long, -1 short, 0 flat
    """
    if ind.empty:
        return np.zeros(0, dtype=int)

    vals = ind.to_numpy()
    n = len(vals)
    sig = np.zeros(n, dtype=int)

    use_tp = tp_pct is not None and tp_pct > 0
    if use_tp:
        if log_returns is None:
            raise ValueError("tp_pct specified but log_returns is None.")
        lr = log_returns.reindex(ind.index).fillna(0.0).to_numpy()
        lr_cum = np.cumsum(lr)
        tp_log = np.log1p(tp_pct)

    pos = 0
    entry_i: int | None = None
    for i, x in enumerate(vals):
        if pos == 0:
            if x > threshold:
                pos = 1
                entry_i = i
            elif x < -threshold:
                pos = -1
                entry_i = i
        elif pos == 1:
            if use_tp and i > entry_i:
                cum = lr_cum[i] - lr_cum[entry_i]
                if cum >= tp_log:
                    pos = 0
                    entry_i = None
            if pos == 1 and x <= exit_level:
                pos = 0
                entry_i = None
        elif pos == -1:
            if use_tp and i > entry_i:
                cum = -(lr_cum[i] - lr_cum[entry_i])
                if cum >= tp_log:
                    pos = 0
                    entry_i = None
            if pos == -1 and x >= -exit_level:
                pos = 0
                entry_i = None
        sig[i] = pos

    return sig


def apply_fees(sig: pd.Series, raw_log_ret: pd.Series, fee_rate: float = 0.0) -> pd.Series:
    """
    Apply per-side trading fees (entry + exit) to a series of raw log returns.
    Fee applied on bars where a position is entered or exited.
    """
    sig = pd.Series(sig, index=raw_log_ret.index).astype(float)
    prev = sig.shift(1).fillna(0)
    entry = (prev == 0) & (sig != 0)
    exit_ = (prev != 0) & (sig == 0)
    fee_log = np.log(1 - fee_rate)
    fees = (entry | exit_).astype(float) * fee_log
    return raw_log_ret + fees


def generate_leg_signal_and_returns(btc_df: pd.DataFrame,
                                    alt_df: pd.DataFrame,
                                    lookback: int,
                                    threshold: float,
                                    atr_lookback: int = 168,
                                    leg: str = "alt",
                                    fee_rate: float = 0.0,
                                    exit_level: float = 0.0,
                                    tp_pct: float | None = None) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Build the intermarket CMMA-diff signal for a single trading leg and compute
    raw and fee-adjusted BAR-LEVEL log returns.

    leg = 'alt' → trade ALT using diff = CMMA(ALT) - CMMA(BTC), returns = ALT next_return
    leg = 'btc' → trade BTC using diff = CMMA(BTC) - CMMA(ALT), returns = BTC next_return

    Returns (signal, raw_log_returns, net_log_returns), all aligned to the signal index.
    """
    idx = btc_df.index.intersection(alt_df.index).sort_values()
    btc = btc_df.loc[idx]
    alt = alt_df.loc[idx]

    btc_c = cmma(btc, lookback, atr_lookback)
    alt_c = cmma(alt, lookback, atr_lookback)

    if leg == "alt":
        diff = (alt_c - btc_c).dropna()
        sig = pd.Series(
            threshold_revert_signal(
                diff,
                threshold,
                log_returns=alt['diff'],
                tp_pct=tp_pct,
                exit_level=exit_level,
            ),
            index=diff.index,
            name="signal_alt",
        )
        lr = alt['next_return'].reindex(diff.index).fillna(0.0)
    elif leg == "btc":
        diff = (btc_c - alt_c).dropna()
        sig = pd.Series(
            threshold_revert_signal(
                diff,
                threshold,
                log_returns=btc['diff'],
                tp_pct=tp_pct,
                exit_level=exit_level,
            ),
            index=diff.index,
            name="signal_btc",
        )
        lr = btc['next_return'].reindex(diff.index).fillna(0.0)
    else:
        raise ValueError("leg must be 'alt' or 'btc'")

    raw = (sig * lr).rename("raw_log_returns")
    net = apply_fees(sig, raw, fee_rate=fee_rate).rename("net_log_returns")
    return sig, raw, net


def generate_pair_trading_system(btc_df: pd.DataFrame,
                                 alt_df: pd.DataFrame,
                                 lookback: int,
                                 threshold: float,
                                 atr_lookback: int = 168,
                                 fee_rate: float = 0.0,
                                 exit_level: float = 0.0,
                                 tp_pct: float | None = None) -> dict:
    """
    Build both legs (ALT and BTC) of the intermarket CMMA-diff trading system.

    Returns a dict with keys:
      - 'alt': {'signal', 'raw', 'net'}
      - 'btc': {'signal', 'raw', 'net'}
      - 'combined_net': fee-adjusted combined BAR-LEVEL returns (average of ALT and BTC legs on their intersection)
    """
    sig_a, raw_a, net_a = generate_leg_signal_and_returns(
        btc_df, alt_df,
        lookback=lookback,
        threshold=threshold,
        atr_lookback=atr_lookback,
        leg="alt",
        fee_rate=fee_rate,
        exit_level=exit_level,
        tp_pct=tp_pct,
    )

    sig_b, raw_b, net_b = generate_leg_signal_and_returns(
        btc_df, alt_df,
        lookback=lookback,
        threshold=threshold,
        atr_lookback=atr_lookback,
        leg="btc",
        fee_rate=fee_rate,
        exit_level=exit_level,
        tp_pct=tp_pct,
    )

    ix = net_a.index.intersection(net_b.index)
    combined_net = ((net_a.loc[ix] + net_b.loc[ix]) / 2.0).rename("combined_net")

    return {
        "alt": {"signal": sig_a, "raw": raw_a, "net": net_a},
        "btc": {"signal": sig_b, "raw": raw_b, "net": net_b},
        "combined_net": combined_net,
    }


