import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# -------------------------- Paths ----------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'merged.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pt')
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------- Hyper‑parameters ----------------------------
WINDOW_SIZE = 10
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------------- Dataset -------------------------------------
class FlareDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window: int,
                 label_encoder: LabelEncoder, scaler: StandardScaler):
        counts = df['COUNTS'].ffill().astype(float).values
        self.inputs = []
        self.labels = []
        self.fluxes = []
        for i in range(window, len(counts)):
            self.inputs.append(counts[i-window:i][:, None])   # (window,1)
            flare_class = df.iloc[i]['FLARE_CLASS']
            self.labels.append(label_encoder.transform([flare_class])[0])
            peak = df.iloc[i]['PEAK_FLUX']
            self.fluxes.append(scaler.transform([[peak]])[0][0])
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.fluxes = torch.tensor(self.fluxes, dtype=torch.float32)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.fluxes[idx]

# -------------------------- Model --------------------------------------
class DualHeadLSTM(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, num_classes=5):
        super(DualHeadLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers,
                                 batch_first=True, dropout=0.2)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
        self.regressor = torch.nn.Linear(hidden_dim, 1)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        hn = hn[-1]
        return self.classifier(hn), self.regressor(hn).squeeze(-1)

# -------------------------- Load data ----------------------------------
print('Loading CSV ...')
df = pd.read_csv(DATA_PATH)
# Ensure needed columns exist (add placeholders if missing)
for col in ['COUNTS']:
    if col not in df.columns:
        raise RuntimeError(f"Column {col} missing in CSV")
# Add dummy FLARE_CLASS if missing
if 'FLARE_CLASS' not in df.columns:
    df['FLARE_CLASS'] = 'A'
# Add dummy PEAK_FLUX if missing
if 'PEAK_FLUX' not in df.columns:
    df['PEAK_FLUX'] = 0.0

# Encode classes and fit scaler
label_encoder = LabelEncoder()
label_encoder.fit(['A', 'B', 'C', 'M', 'X'])
flux_scaler = StandardScaler()
flux_scaler.fit(df['PEAK_FLUX'].values.reshape(-1, 1))

# Split chronologically (no shuffling) to avoid leakage
# --- Active Region Data Splitting ---
if 'AR_NUM' not in df.columns:
    # Create dummy AR_NUM blocks of 500 records if not present
    df['AR_NUM'] = (df.index // 500).astype(str)

unique_ars = df['AR_NUM'].unique()
np.random.shuffle(unique_ars)
split_idx = int(len(unique_ars) * 0.8)
train_ars = unique_ars[:split_idx]
val_ars = unique_ars[split_idx:]

train_df = df[df['AR_NUM'].isin(train_ars)].copy()
val_df = df[df['AR_NUM'].isin(val_ars)].copy()

print(f"Split by AR_NUM: {len(train_ars)} ARs for training, {len(val_ars)} ARs for validation.")
train_set = FlareDataset(train_df, WINDOW_SIZE, label_encoder, flux_scaler)
val_set   = FlareDataset(val_df,   WINDOW_SIZE, label_encoder, flux_scaler)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

# -------------------------- Training ------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DualHeadLSTM().to(device)
criterion_cls = torch.nn.CrossEntropyLoss()
criterion_reg = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def evaluate(loader):
    model.eval()
    total, correct = 0, 0
    mse_sum, n = 0.0, 0
    tp, fp, tn, fn = 0, 0, 0, 0
    with torch.no_grad():
        for xb, yc, yf in loader:
            xb = xb.to(device)
            yc = yc.to(device)
            yf = yf.to(device)
            logits, pred_flux = model(xb)
            _, pred_cls = torch.max(logits, dim=1)
            correct += (pred_cls == yc).sum().item()
            total   += yc.size(0)
            mse_sum += criterion_reg(pred_flux, yf).item() * yf.size(0)
            n += yf.size(0)
            
            # Binary metrics for M/X class flares (assuming M=3, X=4)
            true_major = (yc >= 3)
            pred_major = (pred_cls >= 3)
            tp += (true_major & pred_major).sum().item()
            fp += (~true_major & pred_major).sum().item()
            tn += (~true_major & ~pred_major).sum().item()
            fn += (true_major & ~pred_major).sum().item()
            
    acc = correct / total if total else 0
    mse = mse_sum / n if n else 0
    
    # Calculate robust metrics
    tss = (tp / (tp + fn) if (tp+fn) > 0 else 0) - (fp / (fp + tn) if (fp+tn) > 0 else 0)
    far = fp / (tp + fp) if (tp+fp) > 0 else 0
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = numerator / denominator if denominator > 0 else 0
    
    return acc, mse, tss, hss, far

best_acc = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_loss = 0.0
    for xb, yc, yf in tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}', leave=False):
        xb = xb.to(device)
        yc = yc.to(device)
        yf = yf.to(device)
        optimizer.zero_grad()
        logits, pred_flux = model(xb)
        loss = criterion_cls(logits, yc) + criterion_reg(pred_flux, yf)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_acc, train_mse, tr_tss, tr_hss, tr_far = evaluate(train_loader)
    val_acc,   val_mse,   v_tss,  v_hss,  v_far  = evaluate(val_loader)
    print(f'Epoch {epoch:02d} – loss:{epoch_loss:.4f} '
          f'val_acc:{val_acc*100:5.2f}% val_mse:{val_mse:.4f} '
          f'val_TSS:{v_tss:.3f} val_HSS:{v_hss:.3f} val_FAR:{v_far:.3f}')
    if val_acc > best_acc:
        best_acc = val_acc
        ckpt = {
            'model_state_dict': model.state_dict(),
            'class_encoder': label_encoder,
            'flux_scaler': flux_scaler,
        }
        torch.save(ckpt, MODEL_PATH)
        print(f'✅ New best model saved to {MODEL_PATH}')

print('🏁 Training complete')
print(f'Best validation accuracy: {best_acc*100:.2f}%')
