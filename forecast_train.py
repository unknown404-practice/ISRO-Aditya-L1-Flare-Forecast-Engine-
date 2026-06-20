import json
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Paths (adjust if workspace moves)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CATALOGUE_PATH = os.path.join(BASE_DIR, 'catalogue', 'enriched_master_catalogue.json')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------- Data Preparation -----------------------
class FlareForecastDataset(Dataset):
    """Dataset for forecasting.
    Each sample consists of a sequence of counts (time series) and two targets:
    1) GOES class label (categorical)
    2) Peak flux value (regression)
    """
    def __init__(self, merged_csv_path, catalogue_path, seq_len=48):
        # Load merged counts CSV (DATE, time, COUNTS)
        df = pd.read_csv(merged_csv_path, parse_dates=["DATE"])
        # Ensure datetime index sorted
        df = df.sort_values(["DATE", "time"]).reset_index(drop=True)
        # Use only COUNTS as feature
        self.features = df["COUNTS"].values.astype(float)
        # Load catalogue for labels (assumes same timestamps exist)
        with open(catalogue_path, "r") as f:
            catalog = json.load(f)
        # Build a lookup dict: timestamp -> (goes_class, peak_flux)
        label_lookup = {}
        for entry in catalog:
            ts = pd.to_datetime(entry["timestamp"]).tz_localize(None)
            label_lookup[ts] = (entry.get("goes_class", "B"), entry.get("peak_flux_W_m2", 0.0))
        # Align labels to each row in merged csv (nearest previous event)
        labels = []
        for i, row in df.iterrows():
            ts = pd.to_datetime(row["DATE"]).tz_localize(None)
            # Find the most recent event <= current time
            past_events = [t for t in label_lookup.keys() if t <= ts]
            if past_events:
                last = max(past_events)
                cls, flux = label_lookup[last]
            else:
                cls, flux = "B", 0.0
            labels.append((cls, flux))
        self.labels = labels
        self.seq_len = seq_len
        # Encode GOES classes
        classes = [lbl[0] for lbl in labels]
        self.le = LabelEncoder()
        self.le.fit(["A", "B", "C", "M", "X"])
        self.class_indices = self.le.transform(classes)
        # Scale flux for regression
        fluxes = np.array([lbl[1] for lbl in labels]).reshape(-1, 1)
        self.flux_scaler = StandardScaler()
        self.flux_scaled = self.flux_scaler.fit_transform(fluxes).flatten()

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        seq = self.features[idx: idx + self.seq_len]
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(-1)  # (seq_len, 1)
        class_target = torch.tensor(self.class_indices[idx + self.seq_len], dtype=torch.long)
        flux_target = torch.tensor(self.flux_scaled[idx + self.seq_len], dtype=torch.float32)
        return seq, class_target, flux_target

# ----------------------- Model Definition -----------------------
class DualHeadLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, num_classes=4):
        super(DualHeadLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (hn, _) = self.lstm(x)
        # Take last hidden state
        hn = hn[-1]
        class_logits = self.classifier(hn)
        flux_pred = self.regressor(hn).squeeze(-1)
        return class_logits, flux_pred

# ----------------------- Training Loop -----------------------
def train_model(
    merged_csv_path,
    catalogue_path,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    seq_len=48,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FlareForecastDataset(merged_csv_path, catalogue_path, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = DualHeadLSTM().to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for seq, cls_target, flux_target in dataloader:
            seq = seq.to(device)
            cls_target = cls_target.to(device)
            flux_target = flux_target.to(device)
            optimizer.zero_grad()
            logits, flux_pred = model(seq)
            loss_cls = criterion_cls(logits, cls_target)
            loss_reg = criterion_reg(flux_pred, flux_target)
            loss = loss_cls + loss_reg
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.4f}")
    # Save model & encoders
    model_path = os.path.join(MODEL_DIR, "forecast_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_encoder": dataset.le,
        "flux_scaler": dataset.flux_scaler,
    }, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    merged_csv = os.path.join(DATA_DIR, "merged.csv")
    catalogue = CATALOGUE_PATH
    train_model(merged_csv, catalogue)
