# NeuralTrade AI Coding Agent Instructions

## Project Overview
NeuralTrade is a **Docker-based financial prediction system** analyzing **cryptocurrencies, forex, stocks, and bonds** using:
- **Technical Analysis**: TA-Lib indicators (RSI, SMA, Bollinger Bands)
- **LLM Commentary**: Ollama/Llama3 via LangChain for trading recommendations

**Architecture**: 3-stage sequential pipeline
```
fetch_market_data() → analyze_data() → get_ai_commentary() → console output
```

**Critical Design**: Multi-container Docker setup with inter-container communication (`neuraltrade_brain` → `ollama-service:11434`)

## Project Structure
```
NEURALTRADE-CLAUDE/
├── .env                    # OPENBB_PAT token (git-ignored)
├── docker-compose.yml      # 2 services: neuraltrade + ollama-service
├── Dockerfile              # Debian-based Python 3.10, TA-Lib C library build
├── main.py                 # Sequential pipeline orchestrator
├── requirements.txt        # Heavy ML: PyTorch + TensorFlow + TA-Lib
└── modules/
    ├── data_loader.py      # OpenBB multi-asset fetcher (crypto/forex/stocks/bonds)
    ├── technical_analysis.py  # TA-Lib indicator calculations
    └── ai_advisor.py       # LangChain → Ollama with ICT strategy prompt
```

## Critical Knowledge

### Docker Container Communication
- **Never use `localhost`** in [ai_advisor.py](../modules/ai_advisor.py) when running in Docker
- Correct: `base_url="http://ollama-service:11434"` (service name from docker-compose.yml)
- Local dev: `http://localhost:11434` (requires Ollama installed locally)

### TA-Lib Installation Order (CRITICAL)
1. **C library first** (Dockerfile lines 20-27): wget → compile → install
2. **Python wrapper second** (requirements.txt): `pip install TA-Lib`
- Breaking order causes import errors; rebuild entire image if broken

### Data Flow Contract
```python
# data_loader.py returns
df: pd.DataFrame  # columns: ['close', 'open', 'high', 'low', 'volume']

# technical_analysis.py returns
{
    "price": float,
    "rsi": float,
    "sma_50": float,
    "sma_200": float,
    "bb_pos": float,  # 0-1 position in Bollinger Band
    "trend": str      # "YUKSELIS (BULL)" or "DUSUS (BEAR)"
}

# ai_advisor.py returns
str  # 3-sentence trading commentary
```

## Code Conventions (Non-Negotiable)

### Turkish UI Language
- **All user-facing strings in Turkish**: print statements, error messages, comments
- Examples: `"BASLATILIYOR"` (starting), `"icin veri isteniyor"` (requesting data)
- Keep consistency when adding features

### Print Pattern (Docker-Critical)
```python
print(f"🚀 STAGE_NAME: {details}", flush=True)  # flush=True mandatory
```
- Emoji prefixes: 🚀 start, 📡 data, 🧮 analysis, 🤖 AI, ✅ success, ❌ error
- `flush=True` ensures real-time Docker logs (`docker logs -f neuraltrade_brain`)

### Error Handling Template
```python
try:
    result = operation()
    print("✅ SUCCESS_MSG", flush=True)
    return result
except Exception as e:
    print(f"❌ ERROR: {e}", flush=True)
    return None
```

## Development Workflows

### Quick Docker Test Cycle
```bash
# Terminal 1: Build & start containers
docker-compose up --build -d

# Terminal 2: Run analysis
docker exec -it neuraltrade_brain python main.py

# View AI service logs
docker logs -f neuraltrade_ollama

# Stop all
docker-compose down
```

### Local Development (No Docker)
```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Change ai_advisor.py base_url to localhost
# Run directly
python main.py
```
**Required**: Ollama running locally (`ollama serve`)

### Adding New Technical Indicators
1. Edit [technical_analysis.py](../modules/technical_analysis.py):
```python
import talib
close = df['close'].values
new_indicator = talib.NEW_FUNC(close, timeperiod=N)
result["new_key"] = new_indicator[-1]  # Latest value
```
2. Update [ai_advisor.py](../modules/ai_advisor.py) prompt to include `{analysis_data['new_key']}`
3. Test: `docker exec -it neuraltrade_brain python main.py`

### Changing Target Asset
Edit [main.py](../main.py):
```python
SYMBOL = "BTC-USD"    # Crypto
SYMBOL = "EUR-USD"    # Forex
SYMBOL = "AAPL"       # Stock
SYMBOL = "TLT"        # Bond ETF
```
OpenBB auto-detects asset type via `obb.crypto.price.historical()` (also supports `obb.equity`, `obb.forex`)

## Environment & Dependencies

### .env File (Required)
```bash
OPENBB_PAT=your_token_here  # Get from openbb.co/app
FMP_API_KEY=optional        # Alternative data provider
```
**Critical**: Token must be valid; test with `obb.crypto.price.historical()` in Python shell

### GPU Acceleration (Optional)
Uncomment in [docker-compose.yml](../docker-compose.yml):
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```
Requires: NVIDIA Docker runtime + CUDA drivers

## Troubleshooting

### "Connection refused" from AI Module
```bash
# Check Ollama container running
docker ps | grep ollama

# Test connectivity
docker exec neuraltrade_brain ping ollama-service

# Pull model manually
docker exec neuraltrade_ollama ollama pull llama3
```

### "Module 'talib' not found"
- TA-Lib C library build failed → check Dockerfile logs:
```bash
docker-compose build --no-cache neuraltrade
```

### Empty DataFrame
- Invalid OPENBB_PAT token → verify in `.env`
- Symbol not found → test alternative: `SYMBOL = "BTC-USD"`
- Provider error → change `provider="yfinance"` to `provider="fmp"` in [data_loader.py](../modules/data_loader.py)

## File-Specific Notes

### [main.py](../main.py)
- Single-threaded sequential pipeline (no async)
- Change `SYMBOL` variable for different assets
- All prints use `flush=True` for Docker compatibility

### [modules/data_loader.py](../modules/data_loader.py)
- OpenBB provider: `yfinance` (default, free)
- Alternative providers: `fmp`, `polygon` (require API keys)
- Returns pandas DataFrame or None on error

### [modules/technical_analysis.py](../modules/technical_analysis.py)
- All TA-Lib calculations use `close` price array
- Indicators use latest value: `[-1]` indexing
- Trend logic: `price > sma_50` = bullish

### [modules/ai_advisor.py](../modules/ai_advisor.py)
- LangChain wrapper around Ollama (not raw API)
- Default model: `llama3` (pulled during first run)
- Prompt template: ICT Silver Bullet strategy (3-sentence limit)
- **Critical**: `base_url` must match Docker service name
