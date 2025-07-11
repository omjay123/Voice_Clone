import torch
import pandas as pd
import numpy as np
from model import VoiceCloner

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VoiceCloner()
model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
model.eval()

df = pd.read_csv("mel_reference.csv")
predictions = []

for _, row in df.iterrows():
    speaker_id = row['speaker_id']
    mel = np.array(row['mel_flat'].split(), dtype=np.float32)
    mel = mel[:4000] if len(mel) >= 4000 else np.pad(mel, (0, 4000 - len(mel)))
    mel_tensor = torch.tensor(mel.reshape(80, 50)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mel = model(mel_tensor).squeeze(0).cpu().numpy().flatten()

    pred_str = " ".join(map(str, pred_mel[:4000]))
    predictions.append({"speaker_id": speaker_id, "predicted_mel_flat": pred_str})

pd.DataFrame(predictions).to_csv("cloned_mel_predictions.csv", index=False)
print("Saved to cloned_mel_predictions.csv")
