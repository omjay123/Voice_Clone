import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MelDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.mels = []
        self.speaker_ids = []
        for _, row in df.iterrows():
            mel = np.array(row['mel_flat'].split(), dtype=np.float32)
            mel = mel[:4000] if len(mel) >= 4000 else np.pad(mel, (0, 4000 - len(mel)))
            mel = mel.reshape(80, 50)
            self.mels.append(torch.tensor(mel))
            self.speaker_ids.append(int(row['speaker_id']))

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        return self.mels[idx], self.speaker_ids[idx]
