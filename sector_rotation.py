#!/usr/bin/env python3
"""
Sector Rotation Tracker (Simplified, Tradeable Output)
------------------------------------------------------

A CLI tool that helps you find swing trade setups by analyzing where money is flowing across market sectors
and ranking your watchlist by actionable signals.

Designed for:
- Swing trading options (days to weeks)
- Price action + levels (support/resistance)
- RS divergence edge (accumulation before breakout)
- Limited screen time (day job)

Key outputs (simplified):
- Dashboard (default): one-screen market + top/bottom sectors + top 5 actionable names
- Focus list (--focus): slightly more detail for trade planning
- Single ticker (--ticker): compact quick-check with levels + verdict
- AI-ready markdown report (--md rot_report.md)
- Copy/paste prompts for AI trade plans (--print-prompt daily|weekly)

Data: yfinance (3mo, 1d)

Notes:
- No per-ticker yfinance calls in default path (faster, fewer rate-limit issues).
"""

import argparse
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

SECTORS = {
    "Tech": ["XLK", "QQQ"],
    "Energy": ["XLE"],
    "China": ["FXI", "KWEB"],
    "Industrials": ["XLI"],
    "Consumer Disc": ["XLY"],
    "Financials": ["XLF"],
    "Healthcare": ["XLV"],
    "Utilities": ["XLU"],
    "Staples": ["XLP"],
    "Real Estate": ["XLRE"],
    "Materials": ["XLB"],
}

PRIMARY_ETFS = {
    "Tech": "XLK",
    "Energy": "XLE",
    "China": "FXI",
    "Industrials": "XLI",
    "Consumer Disc": "XLY",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Utilities": "XLU",
    "Staples": "XLP",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}

BENCHMARK = "SPY"

DEFAULT_WATCHLIST = [
    "IONQ",
    "RGTI",
    "GOOGL",
    "MSFT",
    "AAPL",
    "AMZN",
    "META",
    "NFLX",
    "ORCL",
    "SNOW",
    "DDOG",
    "ARM",
    "QCOM",
    "IREN",
    "CORZ",
    "HIVE",
    "TSLA",
    "UBER",
    "RDDT",
    "HIMS",
    "LULU",
    "COST",
    "OUST",
    "PL",
    "RDW",
    "SMCI",
    "NBIS",
    "ZETA",
    "TEM",
    "BABA",
    "ABT",
    "UNH",
    "GS",
    "JPM",
    "EOSE",
    "CRVW",
    "XYZ",
]

DEFAULT_WATCHLIST_SECTORS = {
    "IONQ": "Quantum",
    "RGTI": "Quantum",
    "GOOGL": "Tech",
    "MSFT": "Tech",
    "AAPL": "Tech",
    "AMZN": "Tech",
    "META": "Tech",
    "NFLX": "Tech",
    "ORCL": "Tech",
    "SNOW": "Tech",
    "DDOG": "Tech",
    "ARM": "Tech",
    "QCOM": "Tech",
    "IREN": "Crypto/Mining",
    "CORZ": "Crypto/Mining",
    "HIVE": "Crypto/Mining",
    "TSLA": "EV/Growth",
    "UBER": "EV/Growth",
    "RDDT": "Growth",
    "HIMS": "Growth",
    "LULU": "Retail",
    "COST": "Retail",
    "OUST": "LIDAR/Hardware",
    "PL": "Space",
    "RDW": "Space",
    "SMCI": "Hardware",
    "NBIS": "AI Infra",
    "ZETA": "AdTech",
    "TEM": "AI/Health",
    "BABA": "China",
    "ABT": "Healthcare",
    "UNH": "Healthcare",
    "GS": "Financials",
    "JPM": "Financials",
    "EOSE": "Energy Storage",
    "CRVW": "Industrials",
    "XYZ": "Other",
}

WATCHLIST: List[str] = []
WATCHLIST_SECTORS: Dict[str, str] = {}

SHORT_PERIOD = 8
LONG_PERIOD = 21

ET = pytz.timezone("US/Eastern")


# =============================================================================
# WATCHLIST LOADING
# =============================================================================


def load_watchlist_quiet(csv_path: str = None) -> Tuple[List[str], Dict[str, str]]:
    """Load watchlist from CSV without printing. Falls back to defaults."""
    search_paths = [
        csv_path,
        "watchlist.csv",
        os.path.join(os.path.dirname(__file__), "watchlist.csv"),
        os.path.expanduser("~/watchlist.csv"),
        "/app/watchlist.csv",
    ]

    csv_file = None
    for path in search_paths:
        if path and os.path.exists(path):
            csv_file = path
            break

    if csv_file is None:
        return DEFAULT_WATCHLIST.copy(), DEFAULT_WATCHLIST_SECTORS.copy()

    try:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.lower().str.strip()

        if "ticker" not in df.columns:
            return DEFAULT_WATCHLIST.copy(), DEFAULT_WATCHLIST_SECTORS.copy()

        watchlist: List[str] = []
        watchlist_sectors: Dict[str, str] = {}

        for _, row in df.iterrows():
            ticker = str(row["ticker"]).strip().upper()
            if ticker and ticker != "NAN":
                watchlist.append(ticker)
                sector = "Unknown"
                if "sector" in df.columns and pd.notna(row.get("sector")):
                    sector = str(row.get("sector")).strip()
                watchlist_sectors[ticker] = sector

        if not watchlist:
            return DEFAULT_WATCHLIST.copy(), DEFAULT_WATCHLIST_SECTORS.copy()

        return watchlist, watchlist_sectors

    except Exception:
        return DEFAULT_WATCHLIST.copy(), DEFAULT_WATCHLIST_SECTORS.copy()


# =============================================================================
# DATA FETCHING
# =============================================================================


def get_all_tickers() -> List[str]:
    tickers = [BENCHMARK]
    for etfs in SECTORS.values():
        tickers.extend(etfs)
    tickers.extend(WATCHLIST)
    return list(set(tickers))


def fetch_data(period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    tickers = get_all_tickers()
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    return data


# =============================================================================
# CALCULATIONS
# =============================================================================


def calculate_roc(prices: pd.Series, period: int) -> float:
    if len(prices) < period + 1:
        return np.nan
    current = prices.iloc[-1]
    past = prices.iloc[-(period + 1)]
    return ((current - past) / past) * 100


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    return prices.ewm(span=period, adjust=False).mean()


def calculate_relative_strength(ticker_change: float, benchmark_change: float) -> float:
    return ticker_change - benchmark_change


def calculate_volume_weight(current_volume: float, avg_volume: float) -> float:
    if avg_volume == 0:
        return 1.0
    return current_volume / avg_volume


def calculate_rsi(prices: pd.Series, period: int = 14) -> dict:
    if len(prices) < period + 1:
        return {"valid": False}

    delta = prices.diff()
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)

    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    current_rsi = float(rsi.iloc[-1])
    if current_rsi >= 70:
        condition = "OVERBOUGHT"
        emoji = "🔴"
    elif current_rsi >= 60:
        condition = "BULLISH"
        emoji = "🟢"
    elif current_rsi <= 30:
        condition = "OVERSOLD"
        emoji = "🟢"
    elif current_rsi <= 40:
        condition = "BEARISH"
        emoji = "🔴"
    else:
        condition = "NEUTRAL"
        emoji = "⚪"

    rsi_5_ago = float(rsi.iloc[-5]) if len(rsi) >= 5 else current_rsi
    rsi_trend = current_rsi - rsi_5_ago

    return {
        "valid": True,
        "rsi": current_rsi,
        "condition": condition,
        "emoji": emoji,
        "rsi_trend": rsi_trend,
    }


def get_ema_structure(prices: pd.Series) -> dict:
    if len(prices) < LONG_PERIOD + 5:
        return {"valid": False}

    ema_short = calculate_ema(prices, SHORT_PERIOD)
    ema_long = calculate_ema(prices, LONG_PERIOD)

    current_price = float(prices.iloc[-1])
    current_ema_short = float(ema_short.iloc[-1])
    current_ema_long = float(ema_long.iloc[-1])

    price_vs_short = ((current_price - current_ema_short) / current_ema_short) * 100
    price_vs_long = ((current_price - current_ema_long) / current_ema_long) * 100
    ema_spread = ((current_ema_short - current_ema_long) / current_ema_long) * 100

    if current_price > current_ema_short > current_ema_long:
        trend = "BULLISH"
        trend_emoji = "🟢"
    elif current_price < current_ema_short < current_ema_long:
        trend = "BEARISH"
        trend_emoji = "🔴"
    elif current_ema_short > current_ema_long:
        trend = "WEAKENING BULL"
        trend_emoji = "🟡"
    else:
        trend = "WEAKENING BEAR"
        trend_emoji = "🟠"

    return {
        "valid": True,
        "ema_short": current_ema_short,
        "ema_long": current_ema_long,
        "price_vs_short_pct": float(price_vs_short),
        "price_vs_long_pct": float(price_vs_long),
        "ema_spread_pct": float(ema_spread),
        "trend": trend,
        "trend_emoji": trend_emoji,
    }


def calculate_consolidation_score(prices: pd.Series, period: int = 10) -> dict:
    if len(prices) < 50:
        return {"valid": False}

    recent_prices = prices.tail(period)
    high_low_range = float(recent_prices.max() - recent_prices.min())
    avg_range = float(prices.tail(20).std() * 2)

    range_ratio = high_low_range / avg_range if avg_range > 0 else 1.0

    def calc_bb_width(price_slice: pd.Series):
        if len(price_slice) < 20:
            return None
        sma = float(price_slice.mean())
        std = float(price_slice.std())
        if sma > 0 and std > 0:
            return (std * 2) / sma * 100
        return None

    current_bb_width = calc_bb_width(prices.tail(20))
    if current_bb_width is None:
        return {"valid": False}

    historical_bb_widths: List[float] = []
    for i in range(50, len(prices) + 1):
        width = calc_bb_width(prices.iloc[i - 20 : i])
        if width is not None:
            historical_bb_widths.append(float(width))

    if historical_bb_widths:
        readings_below = sum(1 for w in historical_bb_widths if w < current_bb_width)
        bb_percentile = (readings_below / len(historical_bb_widths)) * 100
    else:
        bb_percentile = 50.0

    is_squeezing = bb_percentile < 30

    return {
        "valid": True,
        "range_ratio": float(range_ratio),
        "bb_width": float(current_bb_width),
        "bb_percentile": float(bb_percentile),
        "is_squeezing": bool(is_squeezing),
    }


def calculate_key_levels(prices: pd.Series, highs: pd.Series, lows: pd.Series) -> dict:
    if len(prices) < 5:
        return {"valid": False}

    current_price = float(prices.iloc[-1])
    week_high = float(highs.tail(5).max())
    week_low = float(lows.tail(5).min())

    year_high = float(highs.tail(252).max()) if len(highs) >= 252 else float(highs.max())
    year_low = float(lows.tail(252).min()) if len(lows) >= 252 else float(lows.min())

    dist_to_week_high = ((week_high - current_price) / current_price) * 100
    dist_to_week_low = ((current_price - week_low) / current_price) * 100
    dist_to_52w_high = ((year_high - current_price) / current_price) * 100
    dist_to_52w_low = ((current_price - year_low) / current_price) * 100

    swing_low = float(lows.tail(10).min())
    stop_distance = ((current_price - swing_low) / current_price) * 100

    return {
        "valid": True,
        "price": current_price,
        "week_high": week_high,
        "week_low": week_low,
        "year_high": year_high,
        "year_low": year_low,
        "dist_to_week_high": float(dist_to_week_high),
        "dist_to_week_low": float(dist_to_week_low),
        "dist_to_52w_high": float(dist_to_52w_high),
        "dist_to_52w_low": float(dist_to_52w_low),
        "swing_low": swing_low,
        "stop_distance": float(stop_distance),
    }


# =============================================================================
# ANALYSIS
# =============================================================================


def analyze_sector(sector_name: str, data: pd.DataFrame, benchmark_prices: pd.Series) -> dict:
    primary_etf = PRIMARY_ETFS[sector_name]
    try:
        if isinstance(data.columns, pd.MultiIndex):
            prices = data[primary_etf]["Close"].dropna()
            volume = data[primary_etf]["Volume"].dropna()
        else:
            prices = data["Close"].dropna()
            volume = data["Volume"].dropna()

        if len(prices) < LONG_PERIOD + 5:
            return {"valid": False, "sector": sector_name, "etf": primary_etf}

        roc_short = calculate_roc(prices, SHORT_PERIOD)
        roc_long = calculate_roc(prices, LONG_PERIOD)

        ema_data = get_ema_structure(prices)

        spy_roc_short = calculate_roc(benchmark_prices, SHORT_PERIOD)
        spy_roc_long = calculate_roc(benchmark_prices, LONG_PERIOD)

        rs_short = calculate_relative_strength(roc_short, spy_roc_short)
        rs_long = calculate_relative_strength(roc_long, spy_roc_long)

        avg_volume = float(volume.tail(20).mean())
        current_volume = float(volume.iloc[-1])
        volume_ratio = calculate_volume_weight(current_volume, avg_volume)

        today_change = (
            ((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2]) * 100
            if len(prices) >= 2
            else 0
        )

        conviction = float(today_change) * min(float(volume_ratio), 3.0)

        return {
            "valid": True,
            "sector": sector_name,
            "etf": primary_etf,
            "price": float(prices.iloc[-1]),
            "today_pct": float(today_change),
            "roc_8": float(roc_short),
            "roc_21": float(roc_long),
            "rs_8": float(rs_short),
            "rs_21": float(rs_long),
            "ema_structure": ema_data,
            "volume_ratio": float(volume_ratio),
            "conviction": float(conviction),
        }
    except Exception as e:
        return {"valid": False, "sector": sector_name, "etf": primary_etf, "error": str(e)}


def analyze_individual_stock(ticker: str, data: pd.DataFrame, benchmark_prices: pd.Series) -> dict:
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker not in data.columns.get_level_values(0):
                return {"valid": False, "ticker": ticker, "error": "No data"}
            prices = data[ticker]["Close"].dropna()
            volume = data[ticker]["Volume"].dropna()
            highs = data[ticker]["High"].dropna()
            lows = data[ticker]["Low"].dropna()
        else:
            prices = data["Close"].dropna()
            volume = data["Volume"].dropna()
            highs = data["High"].dropna()
            lows = data["Low"].dropna()

        if len(prices) < LONG_PERIOD + 10:
            return {"valid": False, "ticker": ticker, "error": "Insufficient data"}

        current_price = float(prices.iloc[-1])

        roc_short = calculate_roc(prices, SHORT_PERIOD)
        roc_long = calculate_roc(prices, LONG_PERIOD)

        ema_data = get_ema_structure(prices)

        spy_roc_short = calculate_roc(benchmark_prices, SHORT_PERIOD)
        spy_roc_long = calculate_roc(benchmark_prices, LONG_PERIOD)

        rs_short = calculate_relative_strength(roc_short, spy_roc_short)
        rs_long = calculate_relative_strength(roc_long, spy_roc_long)

        rs_trend = float(rs_short - rs_long)
        price_trend = float(roc_short - roc_long)

        divergence = None
        if rs_trend > 1.5 and price_trend < 0:
            divergence = "BULLISH"
        elif rs_trend < -1.5 and price_trend > 0:
            divergence = "BEARISH"

        avg_volume = float(volume.tail(20).mean())
        current_volume = float(volume.iloc[-1])
        volume_ratio = float(calculate_volume_weight(current_volume, avg_volume))

        today_change = (
            ((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2]) * 100
            if len(prices) >= 2
            else 0
        )
        week_change = (
            ((prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]) * 100
            if len(prices) >= 5
            else 0
        )

        consolidation = calculate_consolidation_score(prices)
        rsi_data = calculate_rsi(prices, period=14)
        levels = calculate_key_levels(prices, highs, lows)

        sector = WATCHLIST_SECTORS.get(ticker, "Unknown")

        # Setup score (your edge)
        setup_score = 0.0
        if divergence == "BULLISH":
            setup_score += 30
        if consolidation.get("is_squeezing"):
            setup_score += 25
        if rs_short > 0:
            setup_score += min(rs_short * 5, 20)
        if ema_data.get("trend") in ["BULLISH", "WEAKENING BULL"]:
            setup_score += 15
        if volume_ratio > 1.2:
            setup_score += 10
        if rsi_data.get("valid"):
            if rsi_data["rsi"] <= 35 and rs_short > 0:
                setup_score += 20
            elif rsi_data["rsi"] >= 70:
                setup_score -= 10

        return {
            "valid": True,
            "ticker": ticker,
            "sector": sector,
            "price": current_price,
            "today_pct": float(today_change),
            "week_pct": float(week_change),
            "roc_8": float(roc_short),
            "roc_21": float(roc_long),
            "rs_8": float(rs_short),
            "rs_21": float(rs_long),
            "rs_trend": float(rs_trend),
            "divergence": divergence,
            "ema_structure": ema_data,
            "volume_ratio": float(volume_ratio),
            "consolidation": consolidation,
            "rsi": rsi_data,
            "levels": levels,
            "setup_score": float(setup_score),
        }

    except Exception as e:
        return {"valid": False, "ticker": ticker, "error": str(e)}


def rank_sectors(analyses: List[dict], metric: str = "rs_8") -> List[dict]:
    valid = [a for a in analyses if a.get("valid", False)]
    return sorted(valid, key=lambda x: x.get(metric, 0), reverse=True)


# =============================================================================
# OUTPUT HELPERS (Simplified)
# =============================================================================


def fmt_pct(x: float) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.2f}%"


def fmt_float(x: float, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.{digits}f}"


def now_et() -> datetime:
    return datetime.now(ET)


def determine_report_type(force_weekend: bool = False) -> str:
    n = now_et()
    if force_weekend or n.weekday() >= 5:
        return "WEEKEND"
    if n.hour < 12:
        return "MORNING"
    if n.hour < 14:
        return "MIDDAY"
    return "CLOSE"


def market_regime_label(sector_analyses: List[dict]) -> str:
    """Simple risk-on/off heuristic using sector RS vs defensives."""
    ranked = rank_sectors(sector_analyses, "rs_8")
    if not ranked:
        return "MIXED"

    defensive = {"Utilities", "Staples", "Healthcare"}
    risk_on = {"Tech", "Consumer Disc", "Financials"}

    def_rs = np.mean([a["rs_8"] for a in ranked if a["sector"] in defensive]) if any(
        a["sector"] in defensive for a in ranked
    ) else 0.0
    risk_rs = np.mean([a["rs_8"] for a in ranked if a["sector"] in risk_on]) if any(
        a["sector"] in risk_on for a in ranked
    ) else 0.0

    if risk_rs > def_rs + 1:
        return "RISK-ON"
    if def_rs > risk_rs + 1:
        return "RISK-OFF"
    return "MIXED"


def build_flags(stock: dict) -> str:
    flags: List[str] = []
    consol = stock.get("consolidation", {})
    rsi = stock.get("rsi", {})
    ema = stock.get("ema_structure", {})

    if consol.get("is_squeezing"):
        flags.append("🔸Squeeze")

    div = stock.get("divergence")
    if div == "BULLISH":
        flags.append("⚡Accum")
    elif div == "BEARISH":
        flags.append("⚠️Distrib")

    if rsi.get("valid"):
        if rsi.get("rsi", 50) <= 35:
            flags.append("🟢Oversold")
        elif rsi.get("rsi", 50) >= 70:
            flags.append("🔴Overbought")

    # "Extended" heuristic: > +6% above 8EMA
    if ema.get("valid") and ema.get("price_vs_short_pct", 0) > 6:
        flags.append("⚠️Extended")

    return " ".join(flags) if flags else "-"


def breakout_trigger(stock: dict) -> str:
    lv = stock.get("levels", {})
    if lv.get("valid"):
        return f"> {lv['week_high']:.2f}"
    return "N/A"


def pullback_trigger(stock: dict) -> str:
    ema = stock.get("ema_structure", {})
    lv = stock.get("levels", {})
    parts: List[str] = []
    if ema.get("valid"):
        parts.append(f"reclaim 8EMA~{ema['ema_short']:.2f}")
        parts.append(f"/ 21EMA~{ema['ema_long']:.2f}")
    if lv.get("valid"):
        parts.append(f"or hold {lv['week_low']:.2f}")
    return " ".join(parts) if parts else "N/A"


def stop_level(stock: dict) -> Tuple[str, str]:
    lv = stock.get("levels", {})
    if lv.get("valid"):
        stop = f"< {lv['swing_low']:.2f}"
        risk = f"{lv['stop_distance']:.1f}%"
        return stop, risk
    return "N/A", "N/A"


def select_top_bottom_sectors(sector_analyses: List[dict], n: int = 3) -> Tuple[List[dict], List[dict]]:
    ranked = rank_sectors(sector_analyses, "rs_8")
    if len(ranked) <= n:
        return ranked, []
    return ranked[:n], ranked[-n:]


def apply_sector_bonus(stock_analyses: List[dict], sector_analyses: List[dict]) -> None:
    """Add a sector bonus + total_score per stock (simple, deterministic)."""
    sector_rs = {a["sector"]: a.get("rs_8", 0.0) for a in sector_analyses if a.get("valid")}
    for s in stock_analyses:
        if not s.get("valid"):
            continue
        bonus = 0.0
        stock_sector = str(s.get("sector", "Unknown")).strip()

        # Exact match for core sectors, otherwise try a few common mappings
        mapping = {
            "Semis": "Tech",
            "AI Infra": "Tech",
            "Hardware": "Tech",
            "Crypto/Mining": "Tech",
            "EV/Growth": "Consumer Disc",
            "Retail": "Consumer Disc",
            "Growth": "Consumer Disc",
            "Energy Storage": "Energy",
            "China": "China",
        }

        if stock_sector in sector_rs:
            bonus = max(0.0, sector_rs.get(stock_sector, 0.0) * 2)
        else:
            mapped = mapping.get(stock_sector)
            if mapped and mapped in sector_rs:
                bonus = max(0.0, sector_rs.get(mapped, 0.0) * 2)

        s["sector_bonus"] = float(bonus)
        s["total_score"] = float(s.get("setup_score", 0.0) + bonus)


def top_setups(stock_analyses: List[dict], n: int = 5) -> List[dict]:
    valid = [s for s in stock_analyses if s.get("valid")]
    return sorted(valid, key=lambda x: x.get("total_score", x.get("setup_score", 0)), reverse=True)[:n]


# =============================================================================
# MARKDOWN (AI-READY REPORT)
# =============================================================================


def render_markdown_dashboard(
    report_type: str,
    benchmark: dict,
    sector_analyses: List[dict],
    stock_analyses: List[dict],
    top_n: int = 5,
) -> str:
    ts = now_et().strftime("%Y-%m-%d %H:%M:%S %Z")
    regime = market_regime_label(sector_analyses)
    inflows, outflows = select_top_bottom_sectors(sector_analyses, 3)
    picks = top_setups(stock_analyses, top_n)

    spy_ema = benchmark.get("ema_structure", {})
    spy_trend = f"{spy_ema.get('trend_emoji','')} {spy_ema.get('trend','N/A')}".strip()

    def sector_rows(items: List[dict]) -> str:
        lines = []
        for i, s in enumerate(items, 1):
            ema = s.get("ema_structure", {})
            trend = f"{ema.get('trend_emoji','')} {ema.get('trend','N/A')}".strip()
            lines.append(
                f"| {i} | {s['sector']} | {s['etf']} | {fmt_pct(s.get('rs_8'))} | {trend} | {fmt_float(s.get('volume_ratio'),1)}x |"
            )
        return "\n".join(lines) if lines else "| - | - | - | - | - | - |"

    stock_lines = []
    for i, s in enumerate(picks, 1):
        ema = s.get("ema_structure", {})
        trend = f"{ema.get('trend_emoji','')} {ema.get('trend','N/A')}".strip()
        flags = build_flags(s)
        bo = breakout_trigger(s)
        pb = pullback_trigger(s)
        st, risk = stop_level(s)
        lv = s.get("levels", {})
        stop_num = lv.get("swing_low", None)
        risk_num = lv.get("stop_distance", None)
        stock_lines.append(
            f"| {i} | {s['ticker']} | {s.get('sector','Unknown')} | {int(round(s.get('total_score', s.get('setup_score',0))))} | {fmt_pct(s.get('rs_8'))} | {trend} | {flags} | {bo} | {pb} | {st} | {risk} |"
        )

    stock_table = "\n".join(stock_lines) if stock_lines else "| - | - | - | - | - | - | - | - | - | - | - |"

    md = f"""# Sector Rotation Dashboard — {ts} ({report_type})

## 1) Market Regime
- SPY Price: {benchmark.get('price', float('nan')):.2f}
- SPY Trend (8/21): {spy_trend}
- SPY ROC(8): {fmt_pct(benchmark.get('roc_8'))} | ROC(21): {fmt_pct(benchmark.get('roc_21'))}
- Regime: {regime}

## 2) Sector Flow (Top 3 / Bottom 3)

### Inflows (leaders)
| Rank | Sector | ETF | RS(8) | Trend | Vol |
|---:|---|---|---:|---|---:|
{sector_rows(inflows)}

### Outflows (laggards)
| Rank | Sector | ETF | RS(8) | Trend | Vol |
|---:|---|---|---:|---|---:|
{sector_rows(outflows)}

## 3) Action List — Top {top_n} Setups (tradeable)
> Each line includes triggers + stop so an AI can turn it into a plan.

| # | Ticker | Sector | Score | RS(8) | Trend | Flags | Breakout Trigger | Pullback Trigger | Stop (swing low) | Risk% |
|---:|---|---|---:|---:|---|---|---|---|---|---:|
{stock_table}

## 4) Notes (optional, keep short)
- Rotation signals: (use sector RS diverging from trend in your AI prompt if needed)
- Caution: names with 🔴Overbought or ⚠️Extended are usually “wait for pullback”

## 5) Data Window / Assumptions
- Lookback: 3mo daily
- RS: ROC vs SPY (8/21)
- Squeeze: BB width percentile < 30
- Levels: week H/L (5d), swing low (10d)
"""
    return md


def write_markdown_report(md_text: str, out_path: str = "rot_report.md") -> str:
    path = Path(out_path)
    path.write_text(md_text, encoding="utf-8")
    return str(path)


# =============================================================================
# TERMINAL OUTPUT (DASHBOARD + FOCUS + TICKER)
# =============================================================================


def print_dashboard(report_type: str, benchmark: dict, sector_analyses: List[dict], stock_analyses: List[dict], top_n: int):
    ts = now_et().strftime("%Y-%m-%d %H:%M:%S %Z")
    regime = market_regime_label(sector_analyses)
    inflows, outflows = select_top_bottom_sectors(sector_analyses, 3)
    picks = top_setups(stock_analyses, top_n)

    spy_ema = benchmark.get("ema_structure", {})
    spy_trend = f"{spy_ema.get('trend_emoji','')} {spy_ema.get('trend','N/A')}".strip()

    print()
    print("╔" + "═" * 70 + "╗")
    print(f"║  📊 ROTATION DASHBOARD — {report_type:<10} {ts:<42}║")
    print("╚" + "═" * 70 + "╝")
    print(f"SPY ${benchmark['price']:.2f} | Trend: {spy_trend} | ROC8 {fmt_pct(benchmark.get('roc_8'))} | Regime: {regime}")
    print()

    print("SECTOR FLOW (Top 3 / Bottom 3)")
    def _sec_line(s):
        ema = s.get("ema_structure", {})
        trend = f"{ema.get('trend_emoji','')} {ema.get('trend','N/A')}".strip()
        return f"- {s['sector']:<14} {s['etf']:<5} RS8 {fmt_pct(s.get('rs_8')):<8} Vol {fmt_float(s.get('volume_ratio'),1)}x  {trend}"

    print(" Inflows:")
    for s in inflows:
        print(" ", _sec_line(s))
    print(" Outflows:")
    for s in outflows:
        print(" ", _sec_line(s))
    print()

    print(f"ACTION LIST (Top {top_n}) — triggers + stops")
    print(f"{'#':<2} {'Ticker':<7} {'Score':<5} {'RS8':<8} {'Trend':<16} {'Flags':<22} {'BO':<10} {'PB':<28} {'Stop':<10} {'Risk':<6}")
    print("-" * 110)
    for i, s in enumerate(picks, 1):
        ema = s.get("ema_structure", {})
        trend = f"{ema.get('trend_emoji','')} {ema.get('trend','N/A')}".strip()
        flags = build_flags(s)
        bo = breakout_trigger(s)
        pb = pullback_trigger(s)[:27] + ("…" if len(pullback_trigger(s)) > 27 else "")
        st, risk = stop_level(s)
        print(
            f"{i:<2} {s['ticker']:<7} {int(round(s.get('total_score', s.get('setup_score',0)))):<5} "
            f"{fmt_pct(s.get('rs_8')):<8} {trend:<16} {flags:<22} {bo:<10} {pb:<28} {st:<10} {risk:<6}"
        )
    print()


def print_focus(report_type: str, benchmark: dict, sector_analyses: List[dict], stock_analyses: List[dict], top_n: int):
    ts = now_et().strftime("%Y-%m-%d %H:%M:%S %Z")
    regime = market_regime_label(sector_analyses)
    inflows, outflows = select_top_bottom_sectors(sector_analyses, 3)
    picks = top_setups(stock_analyses, top_n)

    spy_ema = benchmark.get("ema_structure", {})
    spy_trend = f"{spy_ema.get('trend_emoji','')} {spy_ema.get('trend','N/A')}".strip()

    print()
    print("╔" + "═" * 70 + "╗")
    print(f"║  🎯 WEEK/DAILY FOCUS LIST — {report_type:<10} {ts:<40}║")
    print("╚" + "═" * 70 + "╝")
    print(f"Market: SPY {spy_trend} | ROC8 {fmt_pct(benchmark.get('roc_8'))} | Regime: {regime}")
    print()

    print("Sector leaders:", ", ".join([f"{s['sector']}({fmt_pct(s['rs_8'])})" for s in inflows]) or "N/A")
    print("Sector laggards:", ", ".join([f"{s['sector']}({fmt_pct(s['rs_8'])})" for s in outflows]) or "N/A")
    print()

    for s in picks:
        lv = s.get("levels", {})
        ema = s.get("ema_structure", {})
        rsi = s.get("rsi", {})
        consol = s.get("consolidation", {})

        trend = f"{ema.get('trend_emoji','')} {ema.get('trend','N/A')}".strip()
        flags = build_flags(s)

        print(f"- {s['ticker']} @ ${s['price']:.2f} | Score {int(round(s.get('total_score', s.get('setup_score',0))))} | RS8 {fmt_pct(s.get('rs_8'))} | {trend}")
        if rsi.get("valid"):
            print(f"  RSI {rsi['rsi']:.0f} {rsi.get('emoji','')} | Vol {fmt_float(s.get('volume_ratio'),1)}x | BB%ile {fmt_float(consol.get('bb_percentile'),0)}")
        print(f"  Flags: {flags}")
        print(f"  Breakout: {breakout_trigger(s)} (week high)")
        print(f"  Pullback: {pullback_trigger(s)}")
        st, risk = stop_level(s)
        print(f"  Stop: {st} | Risk: {risk}")
        if lv.get("valid"):
            print(f"  Week H/L: {lv['week_high']:.2f}/{lv['week_low']:.2f} | 52w High dist: {lv['dist_to_52w_high']:+.1f}%")
        print()


def run_single_ticker_analysis(ticker: str, data: pd.DataFrame, benchmark_prices: pd.Series, sector_analyses: List[dict]):
    stock = analyze_individual_stock(ticker, data, benchmark_prices)
    if not stock.get("valid"):
        print(f"❌ Could not analyze {ticker}: {stock.get('error', 'unknown error')}")
        return

    # Find relevant sector (simple mapping)
    stock_sector = str(stock.get("sector", "Unknown"))
    relevant_sector = None
    for s in sector_analyses:
        if s.get("valid") and s.get("sector") and s["sector"].lower() in stock_sector.lower():
            relevant_sector = s
            break

    spy_ema = get_ema_structure(benchmark_prices)
    spy_roc = calculate_roc(benchmark_prices, SHORT_PERIOD)

    ts = now_et().strftime("%Y-%m-%d %H:%M:%S %Z")
    print()
    print("╔" + "═" * 60 + "╗")
    print(f"║  {ticker} — Quick Check  {ts:<35}║")
    print("╚" + "═" * 60 + "╝")

    print(f"Market: SPY {spy_ema.get('trend_emoji','')} {spy_ema.get('trend','N/A')} | ROC8 {fmt_pct(spy_roc)}")
    if relevant_sector:
        sec_ema = relevant_sector.get("ema_structure", {})
        print(f"Sector: {relevant_sector['sector']} {sec_ema.get('trend_emoji','')} RS8 {fmt_pct(relevant_sector.get('rs_8'))}")

    ema = stock.get("ema_structure", {})
    rsi = stock.get("rsi", {})
    lv = stock.get("levels", {})
    flags = build_flags(stock)

    trend = f"{ema.get('trend_emoji','')} {ema.get('trend','N/A')}".strip()
    print()
    print(f"{ticker} @ ${stock['price']:.2f}")
    print(f"Score: {int(round(stock.get('total_score', stock.get('setup_score',0))))} | RS8 {fmt_pct(stock.get('rs_8'))} | Trend {trend} | Vol {fmt_float(stock.get('volume_ratio'),1)}x")
    if rsi.get("valid"):
        print(f"RSI: {rsi['rsi']:.0f} {rsi.get('emoji','')}")
    print(f"Flags: {flags}")
    print()

    if lv.get("valid"):
        print("Levels:")
        print(f"  Week High: {lv['week_high']:.2f} | Week Low: {lv['week_low']:.2f}")
        print(f"  Swing Low (stop): {lv['swing_low']:.2f} | Risk: {lv['stop_distance']:.1f}%")
        print(f"  52w High: {lv['year_high']:.2f} (dist {lv['dist_to_52w_high']:+.1f}%)")
        print()

    score = stock.get("total_score", stock.get("setup_score", 0))
    rs = stock.get("rs_8", 0)
    rsi_val = rsi.get("rsi", 50) if rsi.get("valid") else 50
    tr = ema.get("trend", "")

    if score >= 50 and rs > 0 and "BULLISH" in tr:
        verdict = "✅ LOOKS GOOD — RS positive, trend supportive"
    elif score >= 50 and rs > 0:
        verdict = "🟡 OKAY — RS positive but trend weakening"
    elif rs > 2 and rsi_val >= 70:
        verdict = "⚠️ EXTENDED — strong RS but overbought/extended"
    elif rs < -2:
        verdict = "❌ AVOID — weak RS, fighting the tape"
    elif rsi_val <= 35 and rs > 0:
        verdict = "⚡ WATCH — oversold + positive RS (accumulation)"
    elif rsi_val <= 35 and rs < 0:
        verdict = "🔻 FALLING KNIFE — oversold + weak RS"
    else:
        verdict = "😐 NEUTRAL — no clear edge"

    print(verdict)
    print()


# =============================================================================
# AI PROMPTS (COPY/PASTE)
# =============================================================================


def prompt_daily() -> str:
    return """You are my trading assistant. My style: swing trade options (days to weeks), price action + levels, relative strength vs SPY, RS divergence (accumulation before breakout), squeeze breakouts, and I can’t watch screens all day.

Task: Create my TRADE PLAN FOR TODAY using:
1) the Sector Rotation report (markdown below)
2) the News/Events bullets (below)

Rules:
- Focus only on the Top 5 candidates from the report unless news creates a clear higher-priority catalyst for a name already on my watchlist.
- Prefer names in strong-inflow sectors; de-prioritize names in strong-outflow sectors unless they’re a short/put setup.
- Generate BOTH entry styles for each candidate:
  A) Breakout entry: trigger = break/close above Week High
  B) Pullback entry: trigger = reclaim 8EMA / hold 21EMA or week low support
- Each setup must include:
  - Thesis in 1–2 sentences (tie to money flow + trend + news)
  - Trigger, invalidation (stop), and first target
  - Position sizing suggestion based on stop distance (risk-based: tight/medium/wide)
  - Options structure suggestion (ATM/ITM calls, debit spread, put, put spread) with DTE guidance (14–45 DTE)
  - What would make me cancel the trade today (news reversal, gap too large, sector flow flips, etc.)
- Also include:
  - Market regime summary (SPY trend/ROC + risk-on/off)
  - “If-Then” plan for: (1) market gaps up, (2) market gaps down
  - A short “Do Not Trade” list: conditions where I should do nothing

Output format:
1) Market context (5 bullets)
2) Today’s top 3 priority trades (detailed)
3) Remaining 2 candidates (short form)
4) Alerts to set (levels)
5) Do-not-trade checklist

--- SECTOR ROTATION REPORT (PASTE BELOW) ---
[PASTE rot_report.md CONTENT HERE]

--- NEWS / EVENTS (PASTE BELOW) ---
[PASTE TODAY’S NEWS BULLETS HERE]
"""


def prompt_weekly() -> str:
    return """You are my weekly trading planner. My style: swing trade options, RS vs SPY, divergence + squeeze, levels-based entries, and limited screen time.

Task: Create my TRADE PLAN FOR THE WEEK using:
1) the Sector Rotation report (markdown below)
2) the Upcoming Week Catalyst list (below: earnings, macro events, known headlines)

Rules:
- Identify the market regime (risk-on/off/mixed) and the 2–3 sectors most likely to lead this week.
- Produce a ranked Focus List of 5 tickers with:
  - Setup type: Breakout / Pullback / Mean-reversion / Short
  - Key levels: week high/low, swing low stop, major resistance, “line in the sand”
  - Conditions required to take the trade (e.g., “sector must remain top-3 RS”, “SPY must stay above 21EMA”)
  - Options structure guidance + DTE guidance (14–60 DTE depending on setup)
  - A plan for what to do if it gaps through the trigger
- Also include:
  - A “Do Nothing Week” filter: conditions that make me stay in cash
  - A watchlist maintenance list: names to drop/add based on sector flow + weak setups

Output format:
1) Weekly Market Thesis (bullets)
2) Sector Flow Outlook (top 3 / bottom 3, with what to watch)
3) Focus List (5 names, detailed plans)
4) Alerts to set (levels across all 5)
5) Risk Management rules for the week

--- SECTOR ROTATION REPORT (PASTE BELOW) ---
[PASTE rot_report.md CONTENT HERE]

--- UPCOMING CATALYSTS (PASTE BELOW) ---
[PASTE EARNINGS + MACRO CALENDAR + KNOWN HEADLINES HERE]
"""


# =============================================================================
# MAIN
# =============================================================================


def run_analysis(
    top_n: int = 5,
    watchlist_csv: str = None,
    force_weekend: bool = False,
    output_json: bool = False,
    md_path: str = None,
    focus: bool = False,
    ticker: str = None,
):
    global WATCHLIST, WATCHLIST_SECTORS
    WATCHLIST, WATCHLIST_SECTORS = load_watchlist_quiet(watchlist_csv)

    data = fetch_data(period="3mo", interval="1d")

    if isinstance(data.columns, pd.MultiIndex):
        benchmark_prices = data[BENCHMARK]["Close"].dropna()
        benchmark_volume = data[BENCHMARK]["Volume"].dropna()
    else:
        benchmark_prices = data["Close"].dropna()
        benchmark_volume = data["Volume"].dropna()

    benchmark = {
        "valid": True,
        "price": float(benchmark_prices.iloc[-1]),
        "today_pct": float(((benchmark_prices.iloc[-1] - benchmark_prices.iloc[-2]) / benchmark_prices.iloc[-2]) * 100)
        if len(benchmark_prices) >= 2
        else 0.0,
        "roc_8": float(calculate_roc(benchmark_prices, SHORT_PERIOD)),
        "roc_21": float(calculate_roc(benchmark_prices, LONG_PERIOD)),
        "ema_structure": get_ema_structure(benchmark_prices),
        "volume_ratio": float(benchmark_volume.iloc[-1] / benchmark_volume.tail(20).mean())
        if len(benchmark_volume) >= 20
        else 1.0,
    }

    sector_analyses: List[dict] = []
    for sector_name in SECTORS.keys():
        sector_analyses.append(analyze_sector(sector_name, data, benchmark_prices))

    stock_analyses: List[dict] = []
    if ticker:
        # Ensure ticker is included in fetch list. If it wasn't, we still fetched full set; but
        # if user ticker isn't in WATCHLIST and thus not in data, add best-effort analysis:
        if isinstance(data.columns, pd.MultiIndex) and ticker not in data.columns.get_level_values(0):
            # Fetch just this ticker + SPY quickly (fallback)
            extra = yf.download([BENCHMARK, ticker], period="3mo", interval="1d", group_by="ticker", progress=False)
            data = extra
            benchmark_prices = data[BENCHMARK]["Close"].dropna() if isinstance(data.columns, pd.MultiIndex) else data["Close"].dropna()
        run_single_ticker_analysis(ticker, data, benchmark_prices, sector_analyses)
        return

    for t in WATCHLIST:
        stock_analyses.append(analyze_individual_stock(t, data, benchmark_prices))

    apply_sector_bonus(stock_analyses, sector_analyses)

    report_type = determine_report_type(force_weekend)

    if output_json:
        payload = {
            "timestamp": now_et().isoformat(),
            "report_type": report_type,
            "benchmark": benchmark,
            "sectors": sector_analyses,
            "watchlist": stock_analyses,
        }
        print(json.dumps(payload, indent=2, default=str))
        return

    # Terminal output
    if focus:
        print_focus(report_type, benchmark, sector_analyses, stock_analyses, top_n)
    else:
        print_dashboard(report_type, benchmark, sector_analyses, stock_analyses, top_n)

    # Markdown output (AI-ready)
    if md_path:
        md = render_markdown_dashboard(report_type, benchmark, sector_analyses, stock_analyses, top_n=top_n)
        out = write_markdown_report(md, md_path)
        print(f"📝 Wrote AI-ready report: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Sector Rotation Tracker (Simplified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rot.py                          # one-screen dashboard
  python rot.py --focus                  # focus list for planning
  python rot.py --md rot_report.md       # write AI-ready markdown report
  python rot.py --ticker AAPL            # single ticker quick-check
  python rot.py --weekend --focus        # weekend mode (same format)
  python rot.py --print-prompt daily     # print daily AI prompt
  python rot.py --print-prompt weekly    # print weekly AI prompt
""",
    )

    parser.add_argument("--csv", type=str, default=None, help="Path to watchlist CSV")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top setups (default: 5)")
    parser.add_argument("--weekend", action="store_true", help="Force weekend mode labeling")
    parser.add_argument("--json", action="store_true", help="Output JSON (raw data)")
    parser.add_argument("--md", type=str, default=None, help="Write AI-ready markdown report to this path")
    parser.add_argument("--focus", action="store_true", help="Print the focus list (planning view)")
    parser.add_argument("--ticker", "-t", type=str, default=None, help="Single ticker quick-check (e.g., -t AAPL)")
    parser.add_argument(
        "--print-prompt",
        choices=["daily", "weekly"],
        default=None,
        help="Print a copy/paste prompt to generate a trade plan from the markdown + news",
    )

    args = parser.parse_args()

    if args.print_prompt:
        print(prompt_daily() if args.print_prompt == "daily" else prompt_weekly())
        return

    run_analysis(
        top_n=args.top_n,
        watchlist_csv=args.csv,
        force_weekend=args.weekend,
        output_json=args.json,
        md_path=args.md,
        focus=args.focus,
        ticker=args.ticker.upper() if args.ticker else None,
    )


if __name__ == "__main__":
    main()
