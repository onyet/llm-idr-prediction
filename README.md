# LLM-EXHANGE — FastAPI starter ✅

A small FastAPI starter project for simple currency forecasting endpoints that use historical data in `data/`.

---

## Quick start 🔧

1. Create a virtual env and install dependencies:

```bash
./setup.sh
```

2. Activate the env:

```bash
source .venv/bin/activate
```

3. Run the dev server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open the interactive API docs: `http://127.0.0.1:8000/docs`

---

## Forecasting endpoints (RAG) 📈

The app provides simple retrieval-augmented forecasting endpoints using historical data found in the `data/` folder (≈5 years):

- `GET /rag/idr-usd?days=N&amount=A` — predict IDR → USD (optionally for an `amount` in IDR)
- `GET /rag/idr-sar?days=N&amount=A` — predict IDR → SAR (optionally for an `amount` in IDR)
- `GET /rag/idr-gold-gram?days=N&amount_gram=A` — predict gold price in IDR per gram (optionally for an `amount_gram` in grams)
- `GET /rag/idr-summary` — comprehensive IDR analysis summary comparing against USD, SAR, and Gold with investment recommendations

Query parameters:

- `days` (int): number of days ahead to predict (default `0` = today). Maximum is **7**. If >7 the API returns HTTP 400 with a short message.
- `amount` (float): nominal amount in IDR to convert (default `1.0`). Response includes `predicted_for_amount` showing the forecast for the provided nominal amount.
- `amount_gram` (float): amount in grams for gold prediction (default `1.0`).

Responses are JSON with the predicted value and uncertainty bounds, for example:

```json
{
  "pair": "idr-usd",
  "method": "prophet",
  "date": "2025-01-03",
  "predicted": 6.01234e-05,
  "predicted_lower": 5.99e-05,
  "predicted_upper": 6.03e-05,
  "trained_until": "2024-12-31"
}
```

The `/rag/idr-summary` endpoint provides a comprehensive analysis:

```json
{
  "summary": "IDR vs USD: TURUN (-0.50%) | IDR vs SAR: TURUN (-0.51%) | Harga Emas: NAIK (+1.27%)",
  "recommendation": {
    "main_recommendation": "INVESTASI EMAS - Emas menunjukkan tren positif...",
    "best_option": "GOLD"
  }
}
```

If the requested date is inside the historical range, the service will return the observed value with a note.

---

## TradingView data endpoints 📊

Fetch historical data for symbols using yfinance (Yahoo Finance API, similar to TradingView):

- `GET /tradingview/get/{kode}?range=R&save=S` — get historical data for symbol {kode} (e.g., USDIDR, automatically adds =X for forex)
- `GET /tradingview/list-code?search=S` — list available ticker codes/symbols with optional search

Query parameters:

- `range` (str): '1y' or '5y' (default '1y')
- `save` (int): 1 to save data to `data/{kode}_{range}.json` (filename sanitized), 0 otherwise (default 0)
- `search` (str): optional search string to filter by code or name (case-insensitive)

Returns JSON with historical price data (Open, High, Low, Close, Volume, etc.) for the specified period. If saved, includes "saved_to" field.

The `/tradingview/list-code` endpoint returns a list of available tickers including Indonesian stocks, forex pairs, commodities, and cryptocurrencies.

---

## Implementation notes ⚠️

- Model: uses `prophet` (FB Prophet) and `pandas` to train on the `data/*.json` files. Models and data are cached in-memory for faster responses.
- Data: historical daily rates are in `data/USDIDR_X.json`, `data/SAR_X.json`, and `data/GC_F.json`. Data is automatically updated daily at 07:00 via cronjob.
- Limits: `days` > 7 is intentionally rejected to keep the forecasts short and reliable.
- Cronjob: The `update_data.py` script runs daily to fetch fresh data for GC=F, SAR=X, and USDIDR=X.

If you plan to use this in production, consider:
- Precomputing forecasts and serving from a cache or DB (cron job or background worker)
- Adding authentication, rate-limiting, and monitoring
- Using a simpler/training-once approach if you need faster startup times

---

## Data Update Script 🔄

The project includes an automated data update script that fetches fresh historical data daily:

- `update_data.py` — fetches 5-year data for GC=F (Gold), SAR=X (Saudi Riyal), and USDIDR=X (USD/IDR) using yfinance
- Cronjob is automatically set up when running `./run.sh` (runs daily at 07:00)
- Data is saved to `data/GC_F.json`, `data/SAR_X.json`, and `data/USDIDR_X.json`

To run manually:

```bash
python update_data.py
```

---

## Tests 🧪

Run the test suite:

```bash
# ensure PYTHONPATH so `app` package can be imported
export PYTHONPATH=.
source .venv/bin/activate
pytest -q
```

The project includes basic tests for the new `/rag` endpoints.

---

## Docker 🐳

Build and run:

```bash
docker build -t llm-exchange:latest .
docker run -p 8000:8000 llm-exchange:latest
```

---

## Server initialization on Ubuntu (easy setup) 🖥️

If you have a fresh Ubuntu VPS and cloned this repo, you can run the provided init script to prepare the machine, install system and Python dependencies, run tests, and create easy run/stop scripts.

Example:

```bash
# from the repository root
sudo bash init_server.sh
```

After the script finishes:

- Start the server: `./run.sh` (runs uvicorn in background, sets up cronjob for data updates, and writes pid to `uvicorn.pid` and logs to `uvicorn.log`)
- Stop the server: `./stop.sh` (stops server and removes cronjob)
- Restart the server: `./restart.sh` (stops if running, then starts)
- A sample systemd unit is available at `systemd/llm-exchange.service` — edit it (WorkingDirectory/ExecStart/User) before copying to `/etc/systemd/system/` and enabling it with systemd.

Notes:
- The init script installs system packages with `apt` (python3, pip, build tools, BLAS/LAPACK libs, etc). It may take several minutes depending on the machine and network.
- Prophet may take extra time to build; check `uvicorn.log` or the pip output if installation fails and install additional OS libs as needed.
- Cronjob for data updates is automatically set up when running `./run.sh` and removed when running `./stop.sh`.

---

If you'd like, I can also add a script that installs and enables the systemd service automatically (requires editing the unit file to set correct `WorkingDirectory` and `User`), or create an Ansible playbook for repeatable deployments.

---

## Author 📝

- **Name**: Dian Mukti Wibowo
- **GitHub**: [@onyet](https://github.com/onyet)
- **Email**: onyetcorp@gmail.com

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

