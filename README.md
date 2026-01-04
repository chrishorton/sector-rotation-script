# Sector Rotation Tracker

Track money flow between market sectors and analyze individual stocks to identify rotation patterns and high-probability setups. Designed for swing traders who want to understand where institutional money is moving.

## What It Tracks

**Sectors (via ETFs):**
- Tech (XLK, QQQ)
- Energy (XLE)
- China (FXI, KWEB)
- Industrials (XLI)
- Consumer Discretionary (XLY)
- Financials (XLF)
- Healthcare (XLV)
- Utilities (XLU)
- Consumer Staples (XLP)
- Real Estate (XLRE)
- Materials (XLB)

**Individual Watchlist (37 names):**
- Quantum/AI: IONQ, RGTI, GOOGL, MSFT, AAPL, AMZN, META, NFLX, ORCL, SNOW, DDOG, ARM, QCOM
- Crypto Miners: IREN, CORZ, HIVE
- Growth: TSLA, UBER, RDDT, HIMS, LULU, COST
- Hardware/Space: OUST, PL, RDW, SMCI, NBIS
- Software/Data: ZETA, TEM, BABA
- Healthcare: ABT, UNH
- Financials: GS, JPM
- Other: EOSE, CRVW, XYZ

**Metrics Computed:**
- **Rate of Change (ROC)** at 8 and 21 periods (matching your EMA setup)
- **8/21 EMA trend structure** - price position relative to EMAs, EMA spread
- **Relative Strength vs SPY** - is the stock/sector beating or lagging the market?
- **RS Divergence Detection** - accumulation (bullish) or distribution (bearish) signals
- **Consolidation/Squeeze Detection** - Bollinger Band width percentile
- **Key Levels** - week high/low, 52w high/low, swing low (stop level)
- **Setup Score** - composite ranking for finding the best trades

## Report Sections

1. **Sector Rotation Analysis** - which sectors are leading/lagging
2. **Money Flow Analysis** - inflows, outflows, rotation signals
3. **Top 5 Setups** - dynamically ranked best opportunities from your watchlist
4. **RS Divergence Signals** - accumulation/distribution detection (your edge)
5. **Consolidation/Squeeze Candidates** - names setting up for breakouts
6. **Full Watchlist Status** - all 37 names ranked by RS
7. **Key Levels** - specific prices for entries, stops, targets
8. **Weekend Prep Summary** (weekends only) - gameplan for the week ahead

## Quick Start

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the report
python sector_rotation.py

# Force weekend prep mode (expanded analysis)
python sector_rotation.py --weekend

# Output as JSON (for piping to other tools)
python sector_rotation.py --json
```

### Run with Docker

```bash
# Build the image
docker build -t sector-rotation:latest .

# Run it
docker run --rm sector-rotation:latest

# Weekend prep mode
docker run --rm sector-rotation:latest --weekend
```

### Deploy to Kubernetes

1. **Update the image registry** in `k8s/cronjob.yaml`:
   ```yaml
   image: your-registry/sector-rotation:latest
   ```

2. **Build and push your image:**
   ```bash
   docker build -t your-registry/sector-rotation:latest .
   docker push your-registry/sector-rotation:latest
   ```

3. **Apply the manifests:**
   ```bash
   kubectl apply -f k8s/cronjob.yaml
   ```

4. **Check the jobs:**
   ```bash
   kubectl get cronjobs -n trading-tools
   kubectl get jobs -n trading-tools
   kubectl logs -n trading-tools job/sector-rotation-morning-<id>
   ```

### Schedule

The k8s CronJobs are configured for:
- **9:45 AM ET** - Morning (after opening volatility settles)
- **12:30 PM ET** - Midday check
- **4:05 PM ET** - Close summary

Only runs Monday-Friday. Automatically detects weekends and shows "WEEKEND PREP" report.

## Understanding the Signals

### Relative Strength (RS)
- **RS > +1%**: Outperforming SPY (money flowing in)
- **RS < -1%**: Underperforming SPY (money flowing out)
- **-1% to +1%**: Neutral / in-line with market

### RS Divergence (Your Edge)
- **Bullish Divergence**: RS improving while price consolidates → accumulation happening
- **Bearish Divergence**: RS weakening while price holds → distribution happening

This is early warning before the move. Look for these on your charts.

### EMA Structure
- **🟢 BULLISH**: Price > 8 EMA > 21 EMA (strong uptrend)
- **🟡 WEAKENING BULL**: 8 EMA > 21 EMA but price pulling back
- **🔴 BEARISH**: Price < 8 EMA < 21 EMA (strong downtrend)
- **🟠 WEAKENING BEAR**: 8 EMA < 21 EMA but price recovering

### Setup Score
Composite score (0-100) based on:
- RS divergence (+30 if bullish)
- Consolidation/squeeze (+25 if BB squeeze)
- Positive RS (+up to 20)
- Trend alignment (+15 if bullish trend)
- Volume confirmation (+10 if elevated)

Higher score = better setup for your swing trading style.

### Consolidation Detection
- **BB Percentile < 30%**: Bollinger Band width in bottom 30% of history = squeeze
- Squeeze + bullish RS = potential breakout setup

## Customization

Edit `sector_rotation.py` to:

- **Modify watchlist**: Update the `WATCHLIST` list
- **Change lookback periods**: Modify `SHORT_PERIOD` and `LONG_PERIOD`
- **Add sectors**: Modify the `SECTORS` dict
- **Adjust scoring**: Tweak the `setup_score` calculation in `analyze_individual_stock()`

## Future Enhancements

- [ ] Discord webhook integration for alerts
- [ ] Options IV percentile data
- [ ] Earnings calendar integration
- [ ] Backtesting RS divergence signals
- [ ] Web dashboard visualization

## Data Source

Uses `yfinance` (Yahoo Finance) - free, no API key required. Data is ~15 minutes delayed for intraday, but sufficient for swing trading analysis.

If you need real-time data, consider upgrading to:
- Polygon.io ($29/mo)
- Alpha Vantage (paid tier)
- Interactive Brokers API (if you have an account)
