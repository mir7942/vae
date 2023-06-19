import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):    
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.2):
        super(Encoder, self).__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # 2nd hidden layer
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # output layer
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def reparameterization(self, mean, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        h = self.fc2(self.fc1(x))

        mean = F.relu(self.mu(h))
        logvar = F.relu(self.logvar(h))

        z = self.reparameterization(mean, logvar)
        return z, mean, logvar
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, dropout=0.2):
        super(Decoder, self).__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # 2nd hidden layer
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # output layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.fc2(self.fc1(z))
        x_reconst = F.sigmoid(self.fc3(h))
        return x_reconst
    
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def forward(self, x):
        z, mean, log_var = self.Encoder(x)
        x_reconst = self.Decoder(z)
        
        return x_reconst, mean, log_var