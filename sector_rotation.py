#!/usr/bin/env python3
"""
Sector Rotation Tracker
-----------------------
Tracks money flow between market sectors to identify rotation patterns.
Designed to run 3x daily: morning (9:45 ET), midday (12:30 ET), close (4:05 ET)

Metrics computed:
- Rate of Change (ROC) at 8 and 21 periods
- 8/21 EMA trend structure
- Relative strength vs SPY
- Volume-weighted performance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Tuple
import pytz
import warnings
import argparse
import json

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SECTORS = {
    # Sector name: [ETF symbols]
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

# Primary ETF for each sector (used for main ranking)
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

# Individual names watchlist - loaded from CSV
# Default fallback if CSV not found
DEFAULT_WATCHLIST = [
    "IONQ", "RGTI", "GOOGL", "MSFT", "AAPL", "AMZN", "META", "NFLX", "ORCL", "SNOW", "DDOG", "ARM", "QCOM",
    "IREN", "CORZ", "HIVE",
    "TSLA", "UBER", "RDDT", "HIMS", "LULU", "COST",
    "OUST", "PL", "RDW", "SMCI", "NBIS",
    "ZETA", "TEM", "BABA",
    "ABT", "UNH",
    "GS", "JPM",
    "EOSE", "CRVW", "XYZ",
]

DEFAULT_WATCHLIST_SECTORS = {
    "IONQ": "Quantum", "RGTI": "Quantum",
    "GOOGL": "Tech", "MSFT": "Tech", "AAPL": "Tech", "AMZN": "Tech", "META": "Tech",
    "NFLX": "Tech", "ORCL": "Tech", "SNOW": "Tech", "DDOG": "Tech", "ARM": "Tech", "QCOM": "Tech",
    "IREN": "Crypto/Mining", "CORZ": "Crypto/Mining", "HIVE": "Crypto/Mining",
    "TSLA": "EV/Growth", "UBER": "EV/Growth", "RDDT": "Growth", "HIMS": "Growth", "LULU": "Retail", "COST": "Retail",
    "OUST": "LIDAR/Hardware", "PL": "Space", "RDW": "Space", "SMCI": "Hardware", "NBIS": "AI Infra",
    "ZETA": "AdTech", "TEM": "AI/Health", "BABA": "China",
    "ABT": "Healthcare", "UNH": "Healthcare",
    "GS": "Financials", "JPM": "Financials",
    "EOSE": "Energy Storage", "CRVW": "Industrials", "XYZ": "Other",
}

# These will be populated from CSV or defaults
WATCHLIST = []
WATCHLIST_SECTORS = {}


def load_watchlist(csv_path: str = None) -> Tuple[List[str], Dict[str, str]]:
    """
    Load watchlist from CSV file.
    
    CSV format:
        ticker,sector,notes
        AAPL,Tech,Apple - consumer tech
        NVDA,Semis,NVIDIA - AI chips leader
    
    Args:
        csv_path: Path to CSV file. If None, searches default locations.
    
    Returns:
        Tuple of (watchlist, watchlist_sectors)
    """
    import os
    
    # Default search paths
    search_paths = [
        csv_path,
        'watchlist.csv',
        os.path.join(os.path.dirname(__file__), 'watchlist.csv'),
        os.path.expanduser('~/watchlist.csv'),
        '/app/watchlist.csv',  # Docker container path
    ]
    
    csv_file = None
    for path in search_paths:
        if path and os.path.exists(path):
            csv_file = path
            break
    
    if csv_file is None:
        print("  ℹ️  No watchlist.csv found, using default watchlist")
        return DEFAULT_WATCHLIST.copy(), DEFAULT_WATCHLIST_SECTORS.copy()
    
    print(f"  📄 Loading watchlist from: {csv_file}")
    
    watchlist = []
    watchlist_sectors = {}
    
    try:
        df = pd.read_csv(csv_file)
        
        # Normalize column names (handle different cases/formats)
        df.columns = df.columns.str.lower().str.strip()
        
        if 'ticker' not in df.columns:
            print(f"  ⚠️  CSV missing 'ticker' column, using defaults")
            return DEFAULT_WATCHLIST.copy(), DEFAULT_WATCHLIST_SECTORS.copy()
        
        for _, row in df.iterrows():
            ticker = str(row['ticker']).strip().upper()
            if ticker and ticker != 'NAN':
                watchlist.append(ticker)
                
                # Get sector if available
                sector = "Unknown"
                if 'sector' in df.columns and pd.notna(row.get('sector')):
                    sector = str(row['sector']).strip()
                
                watchlist_sectors[ticker] = sector
        
        print(f"  ✓ Loaded {len(watchlist)} tickers from CSV")
        return watchlist, watchlist_sectors
        
    except Exception as e:
        print(f"  ⚠️  Error reading CSV: {e}, using defaults")
        return DEFAULT_WATCHLIST.copy(), DEFAULT_WATCHLIST_SECTORS.copy()


# Sector mapping for watchlist names (for context in reports)
# This is now a fallback - primary source is CSV

# Lookback periods (matching your 8/21 EMAs)
SHORT_PERIOD = 8
LONG_PERIOD = 21

# Timezone
ET = pytz.timezone('US/Eastern')


# =============================================================================
# DATA FETCHING
# =============================================================================

def get_all_tickers() -> List[str]:
    """Get all tickers we need to fetch."""
    tickers = [BENCHMARK]
    for etfs in SECTORS.values():
        tickers.extend(etfs)
    tickers.extend(WATCHLIST)
    return list(set(tickers))


def fetch_data(period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical data for all tickers.
    
    Args:
        period: How much history to fetch (1mo, 3mo, 6mo, 1y)
        interval: Data interval (1d, 1h, etc.)
    
    Returns:
        DataFrame with OHLCV data for all tickers
    """
    tickers = get_all_tickers()
    print(f"📡 Fetching data for {len(tickers)} tickers...")
    
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        group_by='ticker',
        progress=False,
        threads=True
    )
    
    return data


def get_current_prices(tickers: List[str]) -> Dict[str, dict]:
    """Get current/latest price data for tickers."""
    result = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.fast_info
            hist = stock.history(period="2d")
            
            if len(hist) >= 1:
                latest = hist.iloc[-1]
                prev_close = hist.iloc[-2]['Close'] if len(hist) >= 2 else latest['Close']
                
                result[ticker] = {
                    'price': latest['Close'],
                    'open': latest['Open'],
                    'high': latest['High'],
                    'low': latest['Low'],
                    'volume': latest['Volume'],
                    'prev_close': prev_close,
                    'change_pct': ((latest['Close'] - prev_close) / prev_close) * 100,
                    'intraday_pct': ((latest['Close'] - latest['Open']) / latest['Open']) * 100
                }
        except Exception as e:
            print(f"  ⚠️  Error fetching {ticker}: {e}")
            
    return result


# =============================================================================
# CALCULATIONS
# =============================================================================

def calculate_roc(prices: pd.Series, period: int) -> float:
    """Calculate Rate of Change over N periods."""
    if len(prices) < period + 1:
        return np.nan
    
    current = prices.iloc[-1]
    past = prices.iloc[-(period + 1)]
    
    return ((current - past) / past) * 100


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return prices.ewm(span=period, adjust=False).mean()


def get_ema_structure(prices: pd.Series) -> dict:
    """
    Analyze EMA structure for a price series.
    
    Returns:
        dict with EMA values and structure analysis
    """
    if len(prices) < LONG_PERIOD + 5:
        return {'valid': False}
    
    ema_short = calculate_ema(prices, SHORT_PERIOD)
    ema_long = calculate_ema(prices, LONG_PERIOD)
    
    current_price = prices.iloc[-1]
    current_ema_short = ema_short.iloc[-1]
    current_ema_long = ema_long.iloc[-1]
    
    # Price position relative to EMAs
    price_vs_short = ((current_price - current_ema_short) / current_ema_short) * 100
    price_vs_long = ((current_price - current_ema_long) / current_ema_long) * 100
    
    # EMA spread (short vs long)
    ema_spread = ((current_ema_short - current_ema_long) / current_ema_long) * 100
    
    # Trend determination
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
        'valid': True,
        'ema_short': current_ema_short,
        'ema_long': current_ema_long,
        'price_vs_short_pct': price_vs_short,
        'price_vs_long_pct': price_vs_long,
        'ema_spread_pct': ema_spread,
        'trend': trend,
        'trend_emoji': trend_emoji
    }


def calculate_relative_strength(ticker_change: float, benchmark_change: float) -> float:
    """Calculate relative strength vs benchmark."""
    return ticker_change - benchmark_change


def calculate_volume_weight(current_volume: float, avg_volume: float) -> float:
    """Calculate volume relative to average."""
    if avg_volume == 0:
        return 1.0
    return current_volume / avg_volume


def calculate_consolidation_score(prices: pd.Series, period: int = 10) -> dict:
    """
    Detect consolidation/squeeze conditions.
    
    Returns:
        dict with consolidation metrics
    """
    if len(prices) < 50:  # Need enough history for percentile calc
        return {'valid': False}
    
    recent_prices = prices.tail(period)
    
    # Range compression (current range vs 20-day average range)
    high_low_range = recent_prices.max() - recent_prices.min()
    avg_range = prices.tail(20).std() * 2  # Approximate range using std
    
    range_ratio = high_low_range / avg_range if avg_range > 0 else 1.0
    
    # Bollinger Band Width (squeeze detection)
    # BB Width = (Upper Band - Lower Band) / Middle Band
    # We calculate this as: (2 * std) / sma * 100
    
    def calc_bb_width(price_slice):
        """Calculate BB width for a price slice."""
        if len(price_slice) < 20:
            return None
        sma = price_slice.mean()
        std = price_slice.std()
        if sma > 0 and std > 0:
            return (std * 2) / sma * 100
        return None
    
    # Current BB width (last 20 days)
    current_bb_width = calc_bb_width(prices.tail(20))
    
    if current_bb_width is None:
        return {'valid': False}
    
    # Calculate historical BB widths (rolling)
    # Go back through history and calculate BB width at each point
    historical_bb_widths = []
    for i in range(50, len(prices) + 1):
        width = calc_bb_width(prices.iloc[i-20:i])
        if width is not None:
            historical_bb_widths.append(width)
    
    # Calculate percentile: what % of historical readings are BELOW current width
    # Lower percentile = tighter squeeze relative to history
    if historical_bb_widths:
        readings_below = sum(1 for w in historical_bb_widths if w < current_bb_width)
        bb_percentile = (readings_below / len(historical_bb_widths)) * 100
    else:
        bb_percentile = 50  # Default to middle if no history
    
    # Consolidation score (lower = tighter consolidation)
    is_squeezing = bb_percentile < 30  # BB width in bottom 30% of history
    
    return {
        'valid': True,
        'range_ratio': range_ratio,
        'bb_width': current_bb_width,
        'bb_percentile': bb_percentile,
        'is_squeezing': is_squeezing,
    }


def calculate_key_levels(prices: pd.Series, highs: pd.Series, lows: pd.Series) -> dict:
    """
    Calculate key price levels for charting.
    """
    if len(prices) < 5:
        return {'valid': False}
    
    current_price = prices.iloc[-1]
    
    # Prior week high/low (last 5 trading days)
    week_high = highs.tail(5).max()
    week_low = lows.tail(5).min()
    
    # 52-week high/low
    year_high = highs.tail(252).max() if len(highs) >= 252 else highs.max()
    year_low = lows.tail(252).min() if len(lows) >= 252 else lows.min()
    
    # Distance calculations
    dist_to_week_high = ((week_high - current_price) / current_price) * 100
    dist_to_week_low = ((current_price - week_low) / current_price) * 100
    dist_to_52w_high = ((year_high - current_price) / current_price) * 100
    dist_to_52w_low = ((current_price - year_low) / current_price) * 100
    
    # Recent swing low (potential stop level) - lowest low in last 10 days
    swing_low = lows.tail(10).min()
    stop_distance = ((current_price - swing_low) / current_price) * 100
    
    return {
        'valid': True,
        'price': current_price,
        'week_high': week_high,
        'week_low': week_low,
        'year_high': year_high,
        'year_low': year_low,
        'dist_to_week_high': dist_to_week_high,
        'dist_to_week_low': dist_to_week_low,
        'dist_to_52w_high': dist_to_52w_high,
        'dist_to_52w_low': dist_to_52w_low,
        'swing_low': swing_low,
        'stop_distance': stop_distance,
    }


def analyze_individual_stock(ticker: str, data: pd.DataFrame, benchmark_prices: pd.Series) -> dict:
    """
    Full analysis for an individual stock.
    """
    try:
        # Handle DataFrame structure
        if isinstance(data.columns, pd.MultiIndex):
            if ticker not in data.columns.get_level_values(0):
                return {'valid': False, 'ticker': ticker, 'error': 'No data'}
            prices = data[ticker]['Close'].dropna()
            volume = data[ticker]['Volume'].dropna()
            highs = data[ticker]['High'].dropna()
            lows = data[ticker]['Low'].dropna()
        else:
            prices = data['Close'].dropna()
            volume = data['Volume'].dropna()
            highs = data['High'].dropna()
            lows = data['Low'].dropna()
        
        if len(prices) < LONG_PERIOD + 10:
            return {'valid': False, 'ticker': ticker, 'error': 'Insufficient data'}
        
        # Basic metrics
        current_price = prices.iloc[-1]
        
        # ROC
        roc_short = calculate_roc(prices, SHORT_PERIOD)
        roc_long = calculate_roc(prices, LONG_PERIOD)
        
        # EMA structure
        ema_data = get_ema_structure(prices)
        
        # Relative strength vs SPY
        spy_roc_short = calculate_roc(benchmark_prices, SHORT_PERIOD)
        spy_roc_long = calculate_roc(benchmark_prices, LONG_PERIOD)
        rs_short = calculate_relative_strength(roc_short, spy_roc_short)
        rs_long = calculate_relative_strength(roc_long, spy_roc_long)
        
        # RS divergence detection
        # Positive divergence: RS improving while price flat/down
        # Negative divergence: RS weakening while price flat/up
        rs_trend = rs_short - rs_long  # Positive = RS accelerating
        price_trend = roc_short - roc_long  # Positive = price accelerating
        
        divergence = None
        if rs_trend > 1.5 and price_trend < 0:
            divergence = "BULLISH"  # RS improving, price lagging - accumulation
        elif rs_trend < -1.5 and price_trend > 0:
            divergence = "BEARISH"  # RS weakening, price holding - distribution
        
        # Volume
        avg_volume = volume.tail(20).mean()
        current_volume = volume.iloc[-1]
        volume_ratio = calculate_volume_weight(current_volume, avg_volume)
        
        # Today's change
        today_change = ((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2]) * 100 if len(prices) >= 2 else 0
        
        # Weekly change
        week_change = ((prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]) * 100 if len(prices) >= 5 else 0
        
        # Consolidation
        consolidation = calculate_consolidation_score(prices)
        
        # Key levels
        levels = calculate_key_levels(prices, highs, lows)
        
        # Sector
        sector = WATCHLIST_SECTORS.get(ticker, "Unknown")
        
        # Setup score (for ranking)
        # Higher = better setup for your style (RS divergence + consolidation)
        setup_score = 0
        if divergence == "BULLISH":
            setup_score += 30
        if consolidation.get('is_squeezing'):
            setup_score += 25
        if rs_short > 0:
            setup_score += min(rs_short * 5, 20)  # Cap at 20 points
        if ema_data.get('trend') in ['BULLISH', 'WEAKENING BULL']:
            setup_score += 15
        if volume_ratio > 1.2:
            setup_score += 10
        
        return {
            'valid': True,
            'ticker': ticker,
            'sector': sector,
            'price': current_price,
            'today_pct': today_change,
            'week_pct': week_change,
            'roc_8': roc_short,
            'roc_21': roc_long,
            'rs_8': rs_short,
            'rs_21': rs_long,
            'rs_trend': rs_trend,
            'divergence': divergence,
            'ema_structure': ema_data,
            'volume_ratio': volume_ratio,
            'consolidation': consolidation,
            'levels': levels,
            'setup_score': setup_score,
        }
        
    except Exception as e:
        return {'valid': False, 'ticker': ticker, 'error': str(e)}


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_sector(sector_name: str, data: pd.DataFrame, benchmark_data: pd.Series) -> dict:
    """
    Comprehensive analysis for a sector.
    
    Args:
        sector_name: Name of the sector
        data: Full historical data DataFrame
        benchmark_data: SPY close prices for comparison
    
    Returns:
        dict with all sector metrics
    """
    primary_etf = PRIMARY_ETFS[sector_name]
    
    try:
        # Handle both multi-ticker and single-ticker DataFrame structures
        if isinstance(data.columns, pd.MultiIndex):
            prices = data[primary_etf]['Close'].dropna()
            volume = data[primary_etf]['Volume'].dropna()
        else:
            prices = data['Close'].dropna()
            volume = data['Volume'].dropna()
        
        if len(prices) < LONG_PERIOD + 5:
            return {'valid': False, 'sector': sector_name, 'etf': primary_etf}
        
        # Rate of Change
        roc_short = calculate_roc(prices, SHORT_PERIOD)
        roc_long = calculate_roc(prices, LONG_PERIOD)
        
        # EMA Structure
        ema_data = get_ema_structure(prices)
        
        # Relative Strength vs SPY
        spy_roc_short = calculate_roc(benchmark_data, SHORT_PERIOD)
        spy_roc_long = calculate_roc(benchmark_data, LONG_PERIOD)
        
        rs_short = calculate_relative_strength(roc_short, spy_roc_short)
        rs_long = calculate_relative_strength(roc_long, spy_roc_long)
        
        # Volume analysis
        avg_volume = volume.tail(20).mean()
        current_volume = volume.iloc[-1]
        volume_ratio = calculate_volume_weight(current_volume, avg_volume)
        
        # Today's performance
        if len(prices) >= 2:
            today_change = ((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2]) * 100
        else:
            today_change = 0
        
        # Volume-weighted performance (conviction score)
        conviction = today_change * min(volume_ratio, 3.0)  # Cap at 3x to avoid outliers
        
        return {
            'valid': True,
            'sector': sector_name,
            'etf': primary_etf,
            'price': prices.iloc[-1],
            'today_pct': today_change,
            'roc_8': roc_short,
            'roc_21': roc_long,
            'rs_8': rs_short,
            'rs_21': rs_long,
            'ema_structure': ema_data,
            'volume_ratio': volume_ratio,
            'conviction': conviction,
        }
        
    except Exception as e:
        print(f"  ⚠️  Error analyzing {sector_name}: {e}")
        return {'valid': False, 'sector': sector_name, 'etf': primary_etf, 'error': str(e)}


def rank_sectors(analyses: List[dict], metric: str = 'rs_8') -> List[dict]:
    """Rank sectors by a given metric."""
    valid = [a for a in analyses if a.get('valid', False)]
    return sorted(valid, key=lambda x: x.get(metric, 0), reverse=True)


# =============================================================================
# REPORTING
# =============================================================================

def format_pct(value: float, width: int = 6) -> str:
    """Format percentage with color indicator."""
    if pd.isna(value):
        return "  N/A "
    
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:>{width}.2f}%"


def format_trend_bar(value: float, max_val: float = 10) -> str:
    """Create a simple ASCII bar for the value."""
    if pd.isna(value):
        return ""
    
    normalized = min(abs(value) / max_val, 1.0)
    bar_len = int(normalized * 10)
    
    if value >= 0:
        return "█" * bar_len + "░" * (10 - bar_len)
    else:
        return "░" * (10 - bar_len) + "█" * bar_len


def print_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_watchlist_report(stock_analyses: List[dict], top_n: int = 5, sections: dict = None):
    """Print the individual watchlist analysis."""
    
    if sections is None:
        sections = {k: True for k in ['watchlist', 'top5', 'divergence', 'squeeze', 'levels']}
    
    valid_stocks = [s for s in stock_analyses if s.get('valid', False)]
    
    if not valid_stocks:
        print("\n  ⚠️  No valid stock data available")
        return
    
    # Top N by setup score (dynamic picks)
    if sections.get('top5', True):
        print_header(f"🎯 TOP {top_n} SETUPS THIS WEEK (Dynamic Picks)")
        print("  Ranked by: RS divergence + consolidation + trend alignment\n")
        
        top_setups = sorted(valid_stocks, key=lambda x: x.get('setup_score', 0), reverse=True)[:top_n]
        
        for i, s in enumerate(top_setups, 1):
            ema = s.get('ema_structure', {})
            consol = s.get('consolidation', {})
            levels = s.get('levels', {})
            
            divergence_flag = f"⚡ {s['divergence']} DIV" if s.get('divergence') else ""
            squeeze_flag = "🔸 SQUEEZE" if consol.get('is_squeezing') else ""
            flags = " ".join(filter(None, [divergence_flag, squeeze_flag]))
            
            print(f"  {i}. {s['ticker']:<6} | {s['sector']:<14} | Score: {s['setup_score']:<3.0f}")
            print(f"     Price: ${s['price']:.2f} | Week: {format_pct(s['week_pct'])} | RS(8): {format_pct(s['rs_8'])}")
            print(f"     Trend: {ema.get('trend_emoji', '❓')} {ema.get('trend', 'N/A'):<15} | Vol: {s['volume_ratio']:.1f}x")
            if flags:
                print(f"     Flags: {flags}")
            if levels.get('valid'):
                print(f"     Levels: Week H/L: ${levels['week_high']:.2f}/${levels['week_low']:.2f} | "
                      f"Stop (swing low): ${levels['swing_low']:.2f} ({levels['stop_distance']:.1f}% risk)")
            print()
    
    # RS Divergences (your edge)
    if sections.get('divergence', True):
        print_header("⚡ RS DIVERGENCE SIGNALS (Accumulation/Distribution)")
        
        bullish_div = [s for s in valid_stocks if s.get('divergence') == 'BULLISH']
        bearish_div = [s for s in valid_stocks if s.get('divergence') == 'BEARISH']
        
        print("\n  🟢 BULLISH DIVERGENCE (RS improving, price consolidating):")
        if bullish_div:
            for s in sorted(bullish_div, key=lambda x: x['rs_trend'], reverse=True):
                print(f"     • {s['ticker']:<6} ({s['sector']:<12}) | RS trend: +{s['rs_trend']:.1f}% | "
                      f"Price: ${s['price']:.2f} | Week: {format_pct(s['week_pct'])}")
        else:
            print("     None detected in watchlist")
        
        print("\n  🔴 BEARISH DIVERGENCE (RS weakening, price holding):")
        if bearish_div:
            for s in sorted(bearish_div, key=lambda x: x['rs_trend']):
                print(f"     • {s['ticker']:<6} ({s['sector']:<12}) | RS trend: {s['rs_trend']:.1f}% | "
                      f"Price: ${s['price']:.2f} | Week: {format_pct(s['week_pct'])}")
        else:
            print("     None detected in watchlist")
    
    # Consolidation / Squeeze candidates
    if sections.get('squeeze', True):
        print_header("🔸 CONSOLIDATION / SQUEEZE CANDIDATES")
        print("  (Tight range + low volatility = potential breakout setup)\n")
        
        squeezing = [s for s in valid_stocks if s.get('consolidation', {}).get('is_squeezing')]
        squeezing = sorted(squeezing, key=lambda x: x.get('consolidation', {}).get('bb_percentile', 100))
        
        if squeezing:
            print(f"  {'Ticker':<8} {'Sector':<14} {'BB %ile':<10} {'RS(8)':<10} {'Trend':<18}")
            print("  " + "-" * 64)
            for s in squeezing[:10]:
                ema = s.get('ema_structure', {})
                consol = s.get('consolidation', {})
                print(f"  {s['ticker']:<8} {s['sector']:<14} {consol.get('bb_percentile', 0):>5.0f}%     "
                      f"{format_pct(s['rs_8']):<10} {ema.get('trend_emoji', '')} {ema.get('trend', 'N/A'):<15}")
        else:
            print("  No squeeze setups detected")
    
    # Full watchlist table
    if sections.get('watchlist', True):
        print_header("📋 FULL WATCHLIST STATUS")
        print(f"  {'Ticker':<7} {'Sector':<12} {'Price':<10} {'Week':<9} {'RS(8)':<9} {'RS(21)':<9} {'Trend':<12}")
        print("  " + "-" * 76)
        
        # Sort by RS(8)
        sorted_stocks = sorted(valid_stocks, key=lambda x: x.get('rs_8', 0), reverse=True)
        
        for s in sorted_stocks:
            ema = s.get('ema_structure', {})
            trend_str = f"{ema.get('trend_emoji', '❓')} {ema.get('trend', 'N/A')[:10]}"
            print(f"  {s['ticker']:<7} {s['sector'][:11]:<12} ${s['price']:<9.2f} "
                  f"{format_pct(s['week_pct']):<9} {format_pct(s['rs_8']):<9} "
                  f"{format_pct(s['rs_21']):<9} {trend_str:<12}")
    
    # Key levels summary for top setups
    if sections.get('levels', True):
        top_setups = sorted(valid_stocks, key=lambda x: x.get('setup_score', 0), reverse=True)[:top_n]
        
        print_header("📐 KEY LEVELS FOR TOP SETUPS")
        print("  (Use for entries, stops, and targets)\n")
        
        for s in top_setups:
            levels = s.get('levels', {})
            if not levels.get('valid'):
                continue
            
            print(f"  {s['ticker']} (${s['price']:.2f})")
            print(f"     Week High:    ${levels['week_high']:.2f}  ({levels['dist_to_week_high']:+.1f}% away)")
            print(f"     Week Low:     ${levels['week_low']:.2f}  ({levels['dist_to_week_low']:+.1f}% below)")
            print(f"     52w High:     ${levels['year_high']:.2f}  ({levels['dist_to_52w_high']:+.1f}% away)")
            print(f"     Swing Low:    ${levels['swing_low']:.2f}  (stop level, {levels['stop_distance']:.1f}% risk)")
            print()


def print_report(analyses: List[dict], benchmark_analysis: dict, report_type: str = "DAILY", sections: dict = None):
    """Print the full rotation report."""
    
    if sections is None:
        sections = {k: True for k in ['benchmark', 'sectors', 'money_flow', 'conviction', 'ema_structure']}
    
    now = datetime.now(ET)
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print(f"║  📊 SECTOR ROTATION REPORT - {report_type:<37} ║")
    print(f"║  Generated: {now.strftime('%Y-%m-%d %H:%M:%S %Z'):<44} ║")
    print("╚" + "═" * 68 + "╝")
    
    # Benchmark summary (always show if enabled)
    if sections.get('benchmark', True):
        print_header("📈 BENCHMARK (SPY)")
        if benchmark_analysis.get('valid'):
            ema = benchmark_analysis.get('ema_structure', {})
            print(f"  Price: ${benchmark_analysis['price']:.2f}  |  "
                  f"Today: {format_pct(benchmark_analysis['today_pct'])}  |  "
                  f"Trend: {ema.get('trend_emoji', '❓')} {ema.get('trend', 'N/A')}")
            print(f"  ROC(8): {format_pct(benchmark_analysis['roc_8'])}  |  "
                  f"ROC(21): {format_pct(benchmark_analysis['roc_21'])}  |  "
                  f"Volume: {benchmark_analysis['volume_ratio']:.1f}x avg")
    
    # Skip remaining sections if no sector data
    if not analyses:
        return
    
    # Sector Rankings by Relative Strength (8-day)
    if sections.get('sectors', True):
        print_header("🏆 SECTOR RANKING BY 8-DAY RELATIVE STRENGTH")
        print(f"  {'Rank':<4} {'Sector':<14} {'ETF':<5} {'RS(8)':<9} {'RS(21)':<9} {'Today':<9} {'Trend':<15}")
        print("  " + "-" * 66)
        
        ranked = rank_sectors(analyses, 'rs_8')
        for i, a in enumerate(ranked, 1):
            ema = a.get('ema_structure', {})
            trend_str = f"{ema.get('trend_emoji', '❓')} {ema.get('trend', 'N/A')}"
            print(f"  {i:<4} {a['sector']:<14} {a['etf']:<5} "
                  f"{format_pct(a['rs_8']):<9} {format_pct(a['rs_21']):<9} "
                  f"{format_pct(a['today_pct']):<9} {trend_str:<15}")
    
    ranked = rank_sectors(analyses, 'rs_8')
    
    # Money Flow Analysis
    if sections.get('money_flow', True):
        print_header("💰 MONEY FLOW ANALYSIS")
        
        # Strong inflows (positive RS + bullish structure)
        strong_inflow = [a for a in ranked if a['rs_8'] > 1.0 and 
                        a.get('ema_structure', {}).get('trend') in ['BULLISH', 'WEAKENING BULL']]
        
        # Strong outflows (negative RS + bearish structure)
        strong_outflow = [a for a in ranked if a['rs_8'] < -1.0 and
                         a.get('ema_structure', {}).get('trend') in ['BEARISH', 'WEAKENING BEAR']]
        
        # Rotation signals (RS diverging from trend)
        rotation_signals = [a for a in ranked if 
                           (a['rs_8'] > 1.0 and a.get('ema_structure', {}).get('trend') in ['BEARISH', 'WEAKENING BEAR']) or
                           (a['rs_8'] < -1.0 and a.get('ema_structure', {}).get('trend') in ['BULLISH', 'WEAKENING BULL'])]
        
        print("\n  🟢 MONEY FLOWING IN (RS > +1%, Bullish structure):")
        if strong_inflow:
            for a in strong_inflow:
                print(f"     • {a['sector']} ({a['etf']}): RS(8) = {format_pct(a['rs_8'])}, "
                      f"Vol = {a['volume_ratio']:.1f}x")
        else:
            print("     None detected")
        
        print("\n  🔴 MONEY FLOWING OUT (RS < -1%, Bearish structure):")
        if strong_outflow:
            for a in strong_outflow:
                print(f"     • {a['sector']} ({a['etf']}): RS(8) = {format_pct(a['rs_8'])}, "
                      f"Vol = {a['volume_ratio']:.1f}x")
        else:
            print("     None detected")
        
        print("\n  ⚡ ROTATION SIGNALS (RS diverging from trend):")
        if rotation_signals:
            for a in rotation_signals:
                ema = a.get('ema_structure', {})
                direction = "turning bullish" if a['rs_8'] > 0 else "turning bearish"
                print(f"     • {a['sector']} ({a['etf']}): {direction} - "
                      f"RS(8) = {format_pct(a['rs_8'])} but trend = {ema.get('trend', 'N/A')}")
        else:
            print("     None detected")
    
    # Conviction Ranking (Volume-Weighted)
    if sections.get('conviction', True):
        print_header("📊 CONVICTION RANKING (Volume-Weighted Performance)")
        conviction_ranked = sorted(ranked, key=lambda x: abs(x.get('conviction', 0)), reverse=True)[:5]
        
        for a in conviction_ranked:
            direction = "↑" if a['conviction'] > 0 else "↓"
            print(f"  {direction} {a['sector']:<14} | Today: {format_pct(a['today_pct'])} | "
                  f"Volume: {a['volume_ratio']:.1f}x | Conviction: {a['conviction']:+.2f}")
    
    # EMA Structure Summary
    if sections.get('ema_structure', True):
        print_header("📐 EMA STRUCTURE SUMMARY (8/21 EMA)")
        print(f"  {'Sector':<14} {'Price vs 8':<12} {'Price vs 21':<12} {'EMA Spread':<12} {'Structure':<15}")
        print("  " + "-" * 66)
        
        for a in ranked:
            ema = a.get('ema_structure', {})
            if ema.get('valid'):
                print(f"  {a['sector']:<14} "
                      f"{format_pct(ema['price_vs_short_pct']):<12} "
                      f"{format_pct(ema['price_vs_long_pct']):<12} "
                      f"{format_pct(ema['ema_spread_pct']):<12} "
                      f"{ema.get('trend_emoji', '')} {ema.get('trend', 'N/A'):<13}")
    
    # Key Takeaways (always show if we have sector data)
    if any([sections.get('sectors'), sections.get('money_flow')]):
        print_header("🎯 KEY TAKEAWAYS")
        
        strong_inflow = [a for a in ranked if a['rs_8'] > 1.0 and 
                        a.get('ema_structure', {}).get('trend') in ['BULLISH', 'WEAKENING BULL']]
        strong_outflow = [a for a in ranked if a['rs_8'] < -1.0 and
                         a.get('ema_structure', {}).get('trend') in ['BEARISH', 'WEAKENING BEAR']]
        rotation_signals = [a for a in ranked if 
                           (a['rs_8'] > 1.0 and a.get('ema_structure', {}).get('trend') in ['BEARISH', 'WEAKENING BEAR']) or
                           (a['rs_8'] < -1.0 and a.get('ema_structure', {}).get('trend') in ['BULLISH', 'WEAKENING BULL'])]
        
        if strong_inflow:
            top_inflow = strong_inflow[0]
            print(f"  • Strongest inflow: {top_inflow['sector']} - consider long setups in this sector")
        
        if strong_outflow:
            top_outflow = strong_outflow[0]
            print(f"  • Strongest outflow: {top_outflow['sector']} - be cautious with longs here")
        
        if rotation_signals:
            print(f"  • {len(rotation_signals)} rotation signal(s) detected - watch for trend changes")
        
        # Defensive vs Risk-On
        defensive = ['Utilities', 'Staples', 'Healthcare']
        risk_on = ['Tech', 'Consumer Disc', 'Financials']
        
        defensive_rs = np.mean([a['rs_8'] for a in ranked if a['sector'] in defensive])
        risk_on_rs = np.mean([a['rs_8'] for a in ranked if a['sector'] in risk_on])
        
        if risk_on_rs > defensive_rs + 1:
            print(f"  • Risk-On environment: Growth sectors outperforming defensives by {risk_on_rs - defensive_rs:.1f}%")
        elif defensive_rs > risk_on_rs + 1:
            print(f"  • Risk-Off environment: Defensive sectors outperforming growth by {defensive_rs - risk_on_rs:.1f}%")
        else:
            print(f"  • Mixed environment: No clear risk-on/risk-off signal")
    
    print("\n" + "─" * 70)
    print(f"  Report complete. Next scheduled run based on your cron configuration.")
    print("─" * 70)


# =============================================================================
# MAIN
# =============================================================================

def run_analysis(output_json: bool = False, weekend_mode: bool = False, sections: dict = None, top_n: int = 5, watchlist_csv: str = None):
    """Run the full sector rotation analysis."""
    
    global WATCHLIST, WATCHLIST_SECTORS
    
    # Load watchlist from CSV
    WATCHLIST, WATCHLIST_SECTORS = load_watchlist(watchlist_csv)
    
    # Default sections (all enabled)
    if sections is None:
        sections = {
            'benchmark': True,
            'sectors': True,
            'money_flow': True,
            'conviction': True,
            'ema_structure': True,
            'watchlist': True,
            'top5': True,
            'divergence': True,
            'squeeze': True,
            'levels': True,
            'weekend_summary': weekend_mode,
        }
    
    # Determine if we need sector data
    need_sectors = any([
        sections.get('sectors'),
        sections.get('money_flow'),
        sections.get('conviction'),
        sections.get('ema_structure'),
        sections.get('weekend_summary'),
    ])
    
    # Determine if we need watchlist data
    need_watchlist = any([
        sections.get('watchlist'),
        sections.get('top5'),
        sections.get('divergence'),
        sections.get('squeeze'),
        sections.get('levels'),
        sections.get('weekend_summary'),
    ])
    
    # Fetch data
    data = fetch_data(period="3mo", interval="1d")
    
    # Get benchmark data
    if isinstance(data.columns, pd.MultiIndex):
        benchmark_prices = data[BENCHMARK]['Close'].dropna()
        benchmark_volume = data[BENCHMARK]['Volume'].dropna()
    else:
        # Single ticker case
        benchmark_prices = data['Close'].dropna()
        benchmark_volume = data['Volume'].dropna()
    
    # Analyze benchmark
    benchmark_analysis = {
        'valid': True,
        'sector': 'Benchmark',
        'etf': BENCHMARK,
        'price': benchmark_prices.iloc[-1],
        'today_pct': ((benchmark_prices.iloc[-1] - benchmark_prices.iloc[-2]) / benchmark_prices.iloc[-2]) * 100 if len(benchmark_prices) >= 2 else 0,
        'roc_8': calculate_roc(benchmark_prices, SHORT_PERIOD),
        'roc_21': calculate_roc(benchmark_prices, LONG_PERIOD),
        'ema_structure': get_ema_structure(benchmark_prices),
        'volume_ratio': benchmark_volume.iloc[-1] / benchmark_volume.tail(20).mean() if len(benchmark_volume) >= 20 else 1.0,
    }
    
    # Analyze sectors (if needed)
    analyses = []
    if need_sectors:
        print("📊 Analyzing sectors...")
        for sector_name in SECTORS.keys():
            analysis = analyze_sector(sector_name, data, benchmark_prices)
            analyses.append(analysis)
            if analysis.get('valid'):
                print(f"  ✓ {sector_name}")
            else:
                print(f"  ✗ {sector_name} (insufficient data)")
    
    # Analyze individual watchlist stocks (if needed)
    stock_analyses = []
    if need_watchlist:
        print("\n📈 Analyzing watchlist stocks...")
        for ticker in WATCHLIST:
            stock_analysis = analyze_individual_stock(ticker, data, benchmark_prices)
            stock_analyses.append(stock_analysis)
            if stock_analysis.get('valid'):
                print(f"  ✓ {ticker}")
            else:
                print(f"  ✗ {ticker} ({stock_analysis.get('error', 'unknown error')})")
    
    # Determine report type based on time and mode
    now = datetime.now(ET)
    hour = now.hour
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    
    if weekend_mode or weekday >= 5:  # Weekend
        report_type = "WEEKEND PREP"
        sections['weekend_summary'] = True
    elif hour < 12:
        report_type = "MORNING OPEN"
    elif hour < 14:
        report_type = "MIDDAY UPDATE"
    else:
        report_type = "MARKET CLOSE"
    
    # Output
    if output_json:
        output = {
            'timestamp': now.isoformat(),
            'report_type': report_type,
            'benchmark': benchmark_analysis,
            'sectors': analyses if need_sectors else [],
            'watchlist': stock_analyses if need_watchlist else [],
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_report(analyses, benchmark_analysis, report_type, sections)
        
        if need_watchlist:
            print_watchlist_report(stock_analyses, top_n=top_n, sections=sections)
        
        # Weekend-specific summary
        if sections.get('weekend_summary'):
            print_weekend_summary(analyses, stock_analyses, benchmark_analysis)


def print_weekend_summary(sector_analyses: List[dict], stock_analyses: List[dict], benchmark: dict):
    """Print weekend-specific prep summary."""
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║  📅 WEEKEND PREP SUMMARY - WEEK AHEAD GAMEPLAN                     ║")
    print("╚" + "═" * 68 + "╝")
    
    valid_sectors = [s for s in sector_analyses if s.get('valid')]
    valid_stocks = [s for s in stock_analyses if s.get('valid')]
    
    # Market regime
    print_header("🌍 MARKET REGIME")
    
    ema = benchmark.get('ema_structure', {})
    print(f"  SPY: ${benchmark['price']:.2f} | {ema.get('trend_emoji', '')} {ema.get('trend', 'N/A')}")
    print(f"  Weekly momentum (ROC-8): {format_pct(benchmark.get('roc_8', 0))}")
    
    # Sector leaders and laggards
    ranked_sectors = sorted(valid_sectors, key=lambda x: x.get('rs_8', 0), reverse=True)
    leaders = ranked_sectors[:3]
    laggards = ranked_sectors[-3:]
    
    print("\n  Leading sectors:  ", end="")
    print(", ".join([f"{s['sector']} ({format_pct(s['rs_8']).strip()})" for s in leaders]))
    
    print("  Lagging sectors:  ", end="")
    print(", ".join([f"{s['sector']} ({format_pct(s['rs_8']).strip()})" for s in laggards]))
    
    # Week ahead focus list
    print_header("🎯 WEEK AHEAD FOCUS LIST")
    
    # Best setups
    top_setups = sorted(valid_stocks, key=lambda x: x.get('setup_score', 0), reverse=True)[:5]
    
    print("\n  PRIORITY WATCHLIST (highest setup scores):\n")
    for s in top_setups:
        levels = s.get('levels', {})
        consol = s.get('consolidation', {})
        
        # Build action items
        actions = []
        if s.get('divergence') == 'BULLISH':
            actions.append("accumulation signal")
        if consol.get('is_squeezing'):
            actions.append("squeeze forming")
        if levels.get('valid') and levels.get('dist_to_week_high', 100) < 3:
            actions.append(f"near week high (${levels['week_high']:.2f})")
        
        action_str = " | ".join(actions) if actions else "monitor for entry"
        
        print(f"  • {s['ticker']:<6} ${s['price']:.2f}")
        print(f"    Why: {action_str}")
        if levels.get('valid'):
            print(f"    Entry trigger: Break above ${levels['week_high']:.2f}")
            print(f"    Stop: ${levels['swing_low']:.2f} ({levels['stop_distance']:.1f}% risk)")
        print()
    
    # Names to avoid
    print("  ⚠️  CAUTION LIST (weak RS, bearish divergence, or lagging):\n")
    
    avoid_list = [s for s in valid_stocks if s.get('rs_8', 0) < -3 or s.get('divergence') == 'BEARISH']
    avoid_list = sorted(avoid_list, key=lambda x: x.get('rs_8', 0))[:5]
    
    if avoid_list:
        for s in avoid_list:
            reason = "bearish divergence" if s.get('divergence') == 'BEARISH' else f"weak RS ({format_pct(s['rs_8']).strip()})"
            print(f"  • {s['ticker']:<6} - {reason}")
    else:
        print("  None flagged - watchlist looks healthy")
    
    # Trading checklist
    print_header("✅ PRE-MARKET CHECKLIST")
    print("""
  □ Review top 5 setups on your charts
  □ Mark key levels (week H/L, swing lows)
  □ Check earnings calendar for watchlist names
  □ Note any overnight futures/news
  □ Set alerts at breakout levels
  □ Size positions based on stop distance
    """)
    
    print("\n" + "─" * 70)
    print("  Good luck this week! 🎯")
    print("─" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Sector Rotation Tracker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sector_rotation.py                    # Full report (all sections)
  python sector_rotation.py --weekend          # Force weekend prep mode
  python sector_rotation.py --sectors-only     # Just sector analysis
  python sector_rotation.py --watchlist-only   # Just watchlist analysis
  python sector_rotation.py --top5 --levels    # Top 5 setups with key levels
  python sector_rotation.py --divergence       # Just RS divergence signals
  python sector_rotation.py --squeeze          # Just consolidation/squeeze candidates
  python sector_rotation.py --json             # Full report as JSON
  python sector_rotation.py --csv my_watchlist.csv  # Use custom watchlist
        """
    )
    
    # Output format
    parser.add_argument('--json', action='store_true', 
                        help='Output as JSON')
    
    # Mode flags
    parser.add_argument('--weekend', action='store_true', 
                        help='Force weekend prep mode (adds week ahead summary)')
    
    # Watchlist source
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to watchlist CSV file (default: watchlist.csv)')
    
    # Section flags - sectors
    parser.add_argument('--sectors-only', action='store_true',
                        help='Only show sector rotation analysis (no watchlist)')
    parser.add_argument('--sectors', action='store_true',
                        help='Include sector rotation analysis')
    parser.add_argument('--money-flow', action='store_true',
                        help='Include money flow analysis')
    parser.add_argument('--conviction', action='store_true',
                        help='Include conviction ranking')
    parser.add_argument('--ema-structure', action='store_true',
                        help='Include EMA structure summary')
    
    # Section flags - watchlist
    parser.add_argument('--watchlist-only', action='store_true',
                        help='Only show watchlist analysis (no sectors)')
    parser.add_argument('--watchlist', action='store_true',
                        help='Include full watchlist status table')
    parser.add_argument('--top5', action='store_true',
                        help='Include top 5 setups')
    parser.add_argument('--divergence', action='store_true',
                        help='Include RS divergence signals')
    parser.add_argument('--squeeze', action='store_true',
                        help='Include consolidation/squeeze candidates')
    parser.add_argument('--levels', action='store_true',
                        help='Include key levels for top setups')
    
    # Customization
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top setups to show (default: 5)')
    
    args = parser.parse_args()
    
    # Determine which sections to show
    sections = {
        'benchmark': True,  # Always show benchmark
        'sectors': False,
        'money_flow': False,
        'conviction': False,
        'ema_structure': False,
        'watchlist': False,
        'top5': False,
        'divergence': False,
        'squeeze': False,
        'levels': False,
        'weekend_summary': False,
    }
    
    # If no specific flags provided, show everything
    specific_flags = [
        args.sectors_only, args.watchlist_only,
        args.sectors, args.money_flow, args.conviction, args.ema_structure,
        args.watchlist, args.top5, args.divergence, args.squeeze, args.levels
    ]
    
    if not any(specific_flags):
        # Default: show everything
        for key in sections:
            sections[key] = True
    else:
        # Specific flags provided
        if args.sectors_only:
            sections['sectors'] = True
            sections['money_flow'] = True
            sections['conviction'] = True
            sections['ema_structure'] = True
        
        if args.watchlist_only:
            sections['watchlist'] = True
            sections['top5'] = True
            sections['divergence'] = True
            sections['squeeze'] = True
            sections['levels'] = True
        
        # Individual flags
        if args.sectors:
            sections['sectors'] = True
        if args.money_flow:
            sections['money_flow'] = True
        if args.conviction:
            sections['conviction'] = True
        if args.ema_structure:
            sections['ema_structure'] = True
        if args.watchlist:
            sections['watchlist'] = True
        if args.top5:
            sections['top5'] = True
        if args.divergence:
            sections['divergence'] = True
        if args.squeeze:
            sections['squeeze'] = True
        if args.levels:
            sections['levels'] = True
    
    # Weekend mode
    if args.weekend:
        sections['weekend_summary'] = True
    
    run_analysis(
        output_json=args.json, 
        weekend_mode=args.weekend,
        sections=sections,
        top_n=args.top_n,
        watchlist_csv=args.csv
    )


if __name__ == "__main__":
    main()
