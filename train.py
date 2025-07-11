import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt

from model import VoiceCloner
from data_loader import MelDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load train and validation datasets
train_dataset = MelDataset("mel_train.csv")
val_dataset = MelDataset("mel_val.csv")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Initialize model
model = VoiceCloner().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Track loss for plotting
train_losses = []
val_losses = []

# Training loop
for epoch in range(50):
    model.train()
    train_loss = 0.0
    for mel, _ in train_loader:
        mel = mel.to(device)
        pred = model(mel)
        loss = loss_fn(pred, mel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss_avg = train_loss / len(train_loader)
    train_losses.append(train_loss_avg)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for mel, _ in val_loader:
            mel = mel.to(device)
            pred = model(mel)
            val_loss += loss_fn(pred, mel).item()

    val_loss_avg = val_loss / len(val_loader)
    val_losses.append(val_loss_avg)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")

# Save model
torch.save(model.state_dict(), "model1.pt")
print("Model saved to model1.pt")

# Plot training + validation loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss", marker='o')
plt.plot(val_losses, label="Val Loss", marker='x')
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_curve.png")
print("Saved training_curve.png")
