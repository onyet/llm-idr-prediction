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

Query parameter:

- `days` (int): number of days ahead to predict (default `0` = today). Maximum is **7**. If >7 the API returns HTTP 400 with a short message.
- `amount` (float): nominal amount in IDR to convert (default `1.0`). Response includes `predicted_for_amount` showing the forecast for the provided nominal amount.

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

If the requested date is inside the historical range, the service will return the observed value with a note.

---

## Implementation notes ⚠️

- Model: uses `prophet` (FB Prophet) and `pandas` to train on the `data/*.json` files. Models and data are cached in-memory for faster responses.
- Data: historical daily rates are in `data/idr-usd.json` and `data/idr-sar.json`.
- Limits: `days` > 7 is intentionally rejected to keep the forecasts short and reliable.

If you plan to use this in production, consider:
- Precomputing forecasts and serving from a cache or DB (cron job or background worker)
- Adding authentication, rate-limiting, and monitoring
- Using a simpler/training-once approach if you need faster startup times

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

- Start the server: `./run.sh` (runs uvicorn in background and writes pid to `uvicorn.pid` and logs to `uvicorn.log`)
- Stop the server: `./stop.sh`
- A sample systemd unit is available at `systemd/llm-exchange.service` — edit it (WorkingDirectory/ExecStart/User) before copying to `/etc/systemd/system/` and enabling it with systemd.

Notes:
- The init script installs system packages with `apt` (python3, pip, build tools, BLAS/LAPACK libs, etc). It may take several minutes depending on the machine and network.
- Prophet may take extra time to build; check `uvicorn.log` or the pip output if installation fails and install additional OS libs as needed.

---

If you'd like, I can also add a script that installs and enables the systemd service automatically (requires editing the unit file to set correct `WorkingDirectory` and `User`), or create an Ansible playbook for repeatable deployments.

---

## Author 📝

- **Name**: Dian Mukti Wibowo
- **GitHub**: [@onyet](https://github.com/onyet)
- **Email**: onyetcorp@gmail.com

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

