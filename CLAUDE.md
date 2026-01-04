# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sector Rotation Tracker - A CLI tool for swing traders that analyzes money flow across market sectors and ranks watchlist stocks by actionable signals. Designed for options swing trading with limited screen time.

## Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard (default view)
python sector_rotation.py

# Focus list (more detail for trade planning)
python sector_rotation.py --focus

# Single ticker quick-check
python sector_rotation.py --ticker AAPL

# Generate AI-ready markdown report
python sector_rotation.py --md rot_report.md

# Print AI prompt templates
python sector_rotation.py --print-prompt daily
python sector_rotation.py --print-prompt weekly

# JSON output
python sector_rotation.py --json

# Weekend mode
python sector_rotation.py --weekend
```

### Docker (via Makefile)
```bash
make build           # Build image
make run             # Dashboard
make focus           # Focus list
make md              # Generate markdown report to ./reports/
make ticker T=AAPL   # Single ticker
make prompt-daily    # Print daily AI prompt
```

## Architecture

Single-file Python script (`sector_rotation.py`) with these main sections:

1. **Configuration** (lines 43-163): Sector ETF mappings, default watchlist, EMA periods (8/21)
2. **Data Fetching** (lines 220-241): Bulk yfinance download for all tickers
3. **Calculations** (lines 248-435): ROC, EMA, RSI, relative strength, consolidation detection, key levels
4. **Analysis** (lines 443-604): Sector and individual stock analysis with setup scoring
5. **Output** (lines 608-1020): Terminal dashboards, markdown reports, AI prompts

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
