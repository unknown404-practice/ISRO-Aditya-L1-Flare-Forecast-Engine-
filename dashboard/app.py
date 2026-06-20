import os
from flask import Flask, jsonify, send_from_directory
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import requests
import time
import sys
import json

# -------------------------------------------------
#  Scheduler & free‑API metric collection
# -------------------------------------------------
from apscheduler.schedulers.background import BackgroundScheduler  # NEW
app = Flask(__name__, static_folder='.', static_url_path='')

# Simple in‑memory cache for metric endpoint (valid for 60 seconds)
metrics_cache = {
    "data": None,
    "timestamp": 0
}

def load_fallback():
    """Load static fallback metrics from bundled JSON file.
    Returns a dict with the same keys as the live payload.
    """
    fallback_path = os.path.join(os.path.dirname(__file__), "static", "metrics_fallback.json")
    try:
        with open(fallback_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not read fallback metrics: {e}", file=sys.stderr)
        return {
            "solar_wind": 0,
            "xray_flux": "0.0e0",
            "sunspot_number": 0,
            "flare_probability": 0.0,
        }
def _fetch_all_metrics():
    """Pull solar‑wind, X‑ray, sunspot and flare‑probability from public APIs.
    If any API call fails, we fall back to the previous cached data or dummy values.
    """
    try:
        # Helper to safely get JSON; returns None on failure
        def safe_get(urls):
            """Try each URL in order, return parsed JSON or None.
            urls – list of candidate URLs (strings)."""
            for url in urls:
                try:
                    r = requests.get(url, timeout=10, verify=False)
                    if r.status_code != 200:
                        print(f"Metrics fetch warning: {url} returned {r.status_code}", file=sys.stderr)
                        continue
                    # Some endpoints return empty body or HTML error page
                    if not r.text.strip():
                        print(f"Metrics fetch warning: empty response from {url}", file=sys.stderr)
                        continue
                    try:
                        return r.json()
                    except Exception as json_err:
                        print(f"Metrics fetch warning: JSON decode error from {url}: {json_err}", file=sys.stderr)
                        continue
                except Exception as exc:
                    print(f"Metrics fetch exception for {url}: {exc}", file=sys.stderr)
            return None

        # 1️⃣ Solar‑wind speed (latest 1‑min record)
        sw = safe_get(["https://services.swpc.noaa.gov/json/solar-wind/plasma-1-day.json"])
        if sw:
            latest_sw = sw[-1]
            solar_wind = round(float(latest_sw.get("speed", 0)), 1)
        else:
            solar_wind = None

        # 2️⃣ GOES X‑ray flux (most recent flare, if any)
        xr = safe_get(["https://services.swpc.noaa.gov/json/goes/primary/xray-flares-1-day.json"])
        if xr:
            xray_flux = f"{float(xr[-1].get('peak_flux', 0)):.2e}"
        else:
            xray_flux = None

        # 3️⃣ Sunspot number (latest daily value)
        ssn = safe_get(["https://api.sidc.be/v1/ssn?format=json&last=1"])
        if ssn:
            sunspot_number = int(ssn[0].get("ssn", 0))
        else:
            sunspot_number = None

        # 4️⃣ Flare probability (48‑hour forecast)
        prob = safe_get(["https://services.swpc.noaa.gov/json/forecast/solar-flare-probability-48-hour.json"])
        if prob:
            latest_prob = prob[-1]
            flare_prob = round((latest_prob.get("M_class_probability", 0) + latest_prob.get("X_class_probability", 0)) / 2, 1)
        else:
            flare_prob = None

        import json
        payload = {
            "solar_wind": solar_wind,
            "xray_flux": xray_flux,
            "sunspot_number": sunspot_number,
            "flare_probability": flare_prob,
        }
        # After building payload, ensure no None values; if any are missing, use static fallback
        if any(v is None for v in payload.values()):
            print("Some metrics missing – loading static fallback", file=sys.stderr)
            fallback = load_fallback()
            # Merge: keep any live values, otherwise use fallback
            for k in payload:
                if payload[k] is None:
                    payload[k] = fallback.get(k, payload[k])
        metrics_cache["data"] = payload
        metrics_cache["timestamp"] = time.time()
        print("Metrics refreshed")
    except Exception as exc:
        print(f"Metrics fetch error: {exc}", file=sys.stderr)
        # Keep previous data if any; otherwise set dummy placeholders
        if metrics_cache.get("data") is None:
            metrics_cache["data"] = {
                "solar_wind": None,
                "xray_flux": None,
                "sunspot_number": None,
                "flare_probability": None,
            }
        metrics_cache["timestamp"] = time.time()

_fetch_all_metrics()
_scheduler = BackgroundScheduler(daemon=True)
_scheduler.add_job(_fetch_all_metrics, "interval", seconds=60, id="metrics_job")
_scheduler.start()





# Paths (adjusted to project root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pt')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'merged.csv')

def read_tail_csv(csv_path: str, n: int = 500):
    import io
    import subprocess
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
    try:
        cmd = ['powershell', '-Command', f'Get-Content -Tail {n} -Encoding UTF8 -Path "{csv_path}"']
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return pd.read_csv(io.StringIO(header + res.stdout), low_memory=False)
    except Exception:
        # Fallback to pandas parsing the whole file if powershell fails
        df = pd.read_csv(csv_path, low_memory=False)
        return df.tail(n)

# Flask route to expose cached metrics
@app.route("/metrics")
def metrics():
    if metrics_cache["data"] is None:
        _fetch_all_metrics()
    return jsonify(metrics_cache["data"] or {})

# ----- Model definition (exact same as training) -----
class DualHeadLSTM(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, num_classes=5):
        super(DualHeadLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
        self.regressor = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (hn, _) = self.lstm(x)
        hn = hn[-1]
        class_logits = self.classifier(hn)
        flux_pred = self.regressor(hn).squeeze(-1)
        return class_logits, flux_pred

# Load the saved model checkpoint (contains state_dict, class_encoder, flux_scaler)
try:
    # Allow loading of sklearn objects (LabelEncoder, StandardScaler) safely.
    import sklearn.preprocessing._label as _label_mod
    import sklearn.preprocessing._data as _data_mod
    torch.serialization.add_safe_globals([
        _label_mod.LabelEncoder,
        _data_mod.StandardScaler,
    ])
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = DualHeadLSTM()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    class_encoder: LabelEncoder = checkpoint['class_encoder']
    flux_scaler: StandardScaler = checkpoint['flux_scaler']
    print('Model and encoders loaded successfully.')
except Exception as e:
    print(f'Failed to load model from {MODEL_PATH}: {e}')
    # Fallback dummy DualHeadLSTM model to keep the service running
    model = DualHeadLSTM()
    model.eval()
    # Dummy encoders
    class_encoder = LabelEncoder()
    class_encoder.fit(['A', 'B', 'C', 'M', 'X'])
    flux_scaler = StandardScaler()
    flux_scaler.mean_ = [0.0]
    flux_scaler.scale_ = [1.0]

def prepare_input(csv_path: str):
    """Read the merged CSV and produce a tensor of the last 10 COUNT values.
    The training script used a window of 10 features for inference.
    """
    # Robust numeric conversion for COUNTS, handling missing / non‑numeric entries
    import numpy as np
    df = read_tail_csv(csv_path, 10)
    if 'COUNTS' not in df.columns:
        raise ValueError('COUNTS column missing in CSV')
    # Convert to numeric, coerce errors to NaN, replace NaN with 0, ensure float dtype
    counts = pd.to_numeric(df['COUNTS'], errors='coerce').fillna(0).values.astype(float)
    # Pad or trim to exactly 10 recent values
    if counts.shape[0] < 10:
        counts = np.pad(counts, (0, 10 - counts.shape[0]), constant_values=0)
    else:
        counts = counts[-10:]
    # Shape (1, 10, 1) for the DualHeadLSTM
    tensor = torch.tensor(counts, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    return tensor

@app.route('/forecast')
def forecast():
    try:
        if not os.path.exists(DATA_PATH):
            return jsonify({"error": "Data file not found"}), 404
        input_tensor = prepare_input(DATA_PATH)
        with torch.no_grad():
            class_logits, flux_pred = model(input_tensor)
        # Decode class
        class_idx = int(torch.argmax(class_logits, dim=1).item())
        flare_class = class_encoder.inverse_transform([class_idx])[0]
        # Attempt to inverse‑scale flux; fallback to raw value
        try:
            val = flux_pred.item()
            peak_flux = float(flux_scaler.inverse_transform([[val]])[0][0])
        except Exception:
            peak_flux = float(flux_pred.item())
        return jsonify({"class": flare_class, "peak_flux_W_m2": peak_flux})
    except Exception as err:
        # Model failed – use recent data for a simple heuristic forecast
        import traceback, sys
        err_msg = f"{err}\n{traceback.format_exc()}"
        print(err_msg, file=sys.stderr)
        try:
            df = read_tail_csv(DATA_PATH, 10)
            recent_counts = pd.to_numeric(df['COUNTS'], errors='coerce').fillna(0).values.astype(float)
            last10 = recent_counts[-10:] if recent_counts.shape[0] >= 10 else recent_counts
            avg = float(last10.mean())
            # Naïve class assignment based on average count
            if avg > 300:
                flare_class = "X"
            elif avg > 150:
                flare_class = "M"
            else:
                flare_class = "C"
            peak_flux = avg
            return jsonify({"class": flare_class, "peak_flux_W_m2": peak_flux})
        except Exception as e2:
            print(f"Heuristic fallback error: {e2}", file=sys.stderr)
            return jsonify({"class": "—", "peak_flux_W_m2": 1e-5}), 200

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/data/<path:filename>')
def data_files(filename):
    data_dir = os.path.join(BASE_DIR, 'data')
    return send_from_directory(data_dir, filename)
# Serve recent subset of data for dashboard
@app.route('/data/recent')
def recent_data():
    # Return last 500 rows of merged.csv as CSV text
    try:
        recent = read_tail_csv(DATA_PATH, 500)
        return recent.to_csv(index=False)
    except Exception as e:
        return str(e), 500
if __name__ == '__main__':
    # Production deployments should use a WSGI server (gunicorn) instead of Flask dev server.
    app.run(host='0.0.0.0', port=5000, debug=False)
