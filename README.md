# FINANCE — Bank Statement Insights

Streamlit app that:
- Uploads bank statements (`PDF`, `Excel`, `CSV`)
- Cleans data (missing values, date parsing, amount extraction)
- Auto-categorizes using simple keywords (Swiggy/Uber/Amazon)
- Shows a Plotly dashboard (pie, line, bar)
- Tracks category budgets with progress + “near limit” alerts
- Detects simple spending anomalies
- Detects repeated transactions as subscriptions

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Render Deploy Note

If Render build fails while compiling `pandas`, make sure the service uses Python from `runtime.txt`:

```txt
python-3.11.9
```

Then trigger a fresh deploy (clear build cache if needed).

If Render still uses Python 3.14 (logs show `cpython-314`), deploy using `render.yaml` (included) which pins:
- `pythonVersion: 3.11.9`
- Start command binds Streamlit to `$PORT`

