# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sector Rotation Tracker - A CLI tool for swing traders that analyzes money flow across market sectors and ranks watchlist stocks by actionable signals. Designed for options swing trading with limited screen time.

## Commands

### Docker (Recommended - via Makefile)

**IMPORTANT:** After making code changes, rebuild the Docker image first:
```bash
make build           # Rebuild image after code changes
```

Then run commands:
```bash
make run             # Dashboard
make focus           # Focus list
make md              # Generate markdown report to ./reports/
make ticker T=AAPL   # Single ticker
make prompt-daily    # Print daily AI prompt

# History tracking
make snapshot        # Save daily snapshot
make track-events    # Track signal changes
make snapshot-full   # Both snapshot + events
```

### Local Development (using uv)
```bash
# Install dependencies with uv
uv pip install -r requirements.txt

# Run dashboard (default view)
uv run sector_rotation.py

# Focus list (more detail for trade planning)
uv run sector_rotation.py --focus

# Single ticker quick-check
uv run sector_rotation.py --ticker AAPL

# Generate AI-ready markdown report
uv run sector_rotation.py --md rot_report.md

# Print AI prompt templates
uv run sector_rotation.py --print-prompt daily
uv run sector_rotation.py --print-prompt weekly

# History tracking
uv run sector_rotation.py --save-snapshot --track-events     # Both (recommended daily)

# Trade journal
uv run sector_rotation.py --log-trade AAPL --action OPEN --entry 150.00 --quantity 10 --strategy "LONG_CALL" --notes "Breakout play"
uv run sector_rotation.py --log-trade AAPL --action CLOSE --entry 150.00 --exit 155.00 --quantity 10

# Watchlist management
uv run sector_rotation.py --watchlist-add NVDA --reason "Strong RS + sector rotation"
uv run sector_rotation.py --watchlist-remove XYZ --reason "Weak setup, no edge"
```

## Architecture

Single-file Python script (`sector_rotation.py`) with these main sections:

1. **Configuration** (lines 43-163): Sector ETF mappings, default watchlist, EMA periods (8/21)
2. **Data Fetching** (lines 220-241): Bulk yfinance download for all tickers
3. **Calculations** (lines 248-435): ROC, EMA, RSI, relative strength, consolidation detection, key levels
4. **Analysis** (lines 443-604): Sector and individual stock analysis with setup scoring
5. **Output** (lines 608-1020): Terminal dashboards, markdown reports, AI prompts
6. **History Tracking** (lines 1103-1411): Daily snapshots, signal events, trade journal, watchlist changes

### Key Concepts

- **Relative Strength (RS)**: ROC vs SPY benchmark - positive = outperforming market
- **RS Divergence**: RS improving while price consolidates = accumulation (bullish edge)
- **Setup Score**: Composite 0-100 based on divergence, squeeze, RS, trend, volume
- **EMA Structure**: Price position relative to 8/21 EMAs determines trend state

### Watchlist Loading

Searches for `watchlist.csv` in order: CLI arg → current dir → script dir → home dir → /app/
Falls back to hardcoded `DEFAULT_WATCHLIST` if not found.

CSV format: `ticker,sector` columns

## Dependencies

- yfinance (Yahoo Finance data)
- pandas, numpy (data processing)
- pytz (timezone handling for market hours)

## Output Modes

| Flag | Description |
|------|-------------|
| (none) | One-screen terminal dashboard |
| `--focus` | Detailed planning view |
| `--ticker X` | Single stock quick-check |
| `--md path` | AI-ready markdown report |
| `--json` | Raw JSON output |
| `--print-prompt daily\|weekly` | AI prompt templates |
| `--save-snapshot` | Save full state to CSV |
| `--track-events` | Log signal changes to CSV |

## History Tracking

All history data is stored in `history/` directory as CSV files:

### 1. Daily Snapshots (`history/daily_snapshots.csv`)
Captures complete state for all tickers (SPY, sectors, stocks) each run.

**Columns:** date, timestamp, ticker, type, sector, price, roc_8, roc_21, rs_8, rs_21, setup_score, total_score, trend, rsi, volume_ratio, divergence, is_squeezing, bb_percentile

**Use cases:**
- Historical performance review
- Backtesting setup scores
- Tracking how scores evolved over time
- Identifying what worked/didn't work

### 2. Signal Events (`history/signal_events.csv`)
Automatically detects and logs significant changes by comparing to previous snapshot.

**Tracked events:**
- `ADDED_TO_WATCHLIST` - First time ticker appears
- `SETUP_IMPROVED` / `SETUP_WEAKENED` - Score change >20 points
- `DIVERGENCE_BULLISH` / `DIVERGENCE_BEARISH` - New divergence detected
- `ENTERED_SQUEEZE` / `EXITED_SQUEEZE` - BB squeeze state change
- `RS_TURNED_POSITIVE` / `RS_TURNED_NEGATIVE` - RS crosses zero

**Columns:** date, timestamp, ticker, event_type, description, score_before, score_after, rs_8_before, rs_8_after

**Use cases:**
- Alerts for setup changes
- Backtesting signal timing
- Identifying entry/exit patterns

### 3. Trade Journal (`history/trades.csv`)
Manual logging of actual positions taken.

**Columns:** date, timestamp, ticker, action (OPEN/CLOSE/ADJUST), entry_price, exit_price, quantity, strategy, stop_loss, target, pnl, pnl_pct, notes

**Use cases:**
- Performance tracking
- Journal review
- Strategy win rate analysis
- Risk/reward validation

### 4. Watchlist Changes (`history/watchlist_changes.csv`)
Tracks when and why tickers are added/removed from watchlist.

**Columns:** date, timestamp, ticker, action (ADD/REMOVE), reason

**Use cases:**
- Watchlist maintenance history
- Understanding rotation patterns
- Documenting decision-making process
