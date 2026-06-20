import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Simple model matching DualHeadLSTM input/output dimensions
model = torch.nn.Linear(10, 2)
state_dict = model.state_dict()

# Dummy encoders
le = LabelEncoder()
le.fit(['A', 'B', 'C', 'M', 'X'])

sc = StandardScaler()
sc.mean_ = np.array([0.0])
sc.scale_ = np.array([1.0])

ckpt = {
    'model_state_dict': state_dict,
    'class_encoder': le,
    'flux_scaler': sc
}

torch.save(ckpt, r'C:\\Users\\RANADEEP\\Desktop\\Gemini\\aditya_l1_flare_pipeline\\models\\model.pt')
print('Dummy model checkpoint created')
