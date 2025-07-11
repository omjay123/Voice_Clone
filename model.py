import torch
import torch.nn as nn

class SpeakerEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(80, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(256, embedding_dim)

    def forward(self, x): 
        x = self.conv(x)
        x = self.pool(x).squeeze(2)  
        return self.linear(x) 

class MelDecoder(nn.Module):
    def __init__(self, embedding_dim=128, output_dim=80, time_steps=50):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim * time_steps)
        )
        self.output_dim = output_dim
        self.time_steps = time_steps

    def forward(self, embedding): 
        out = self.fc(embedding)
        return out.view(-1, self.output_dim, self.time_steps)

class VoiceCloner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpeakerEncoder()
        self.decoder = MelDecoder()

    def forward(self, mel):
        embedding = self.encoder(mel)
        return self.decoder(embedding)
