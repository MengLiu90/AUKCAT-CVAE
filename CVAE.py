## condition on substrate + EC, with regressor head
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

df = pd.read_csv('/Datasets/example_data.csv')

mol2vec_cols = [f'mol2vec_{i}' for i in range(300)]
node2vec_cols = [f'Embedding_{i+1}' for i in range(128)]
ec2vec_cols = [f'ec2vec_{i}' for i in range(1024)]

x_cols = node2vec_cols  # species embedding (input)
c_cols = mol2vec_cols + ec2vec_cols  # substrate + EC embedding (condition)
X = df[x_cols].values.astype(np.float32)
C = df[c_cols].values.astype(np.float32)
y_o = df["kcat"].values.astype(np.float32)
y = np.log10(np.clip(y_o, 1e-6, None))  # log(kcat)

# ----- Train/Test Split -----
X_train, X_test, C_train, C_test, y_train, y_test = train_test_split(X, C, y, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(C_train), torch.tensor(y_train)), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(C_test), torch.tensor(y_test)), batch_size=64, shuffle=False)

save_dir = 'Trained_model'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
save_path = os.path.join(save_dir, "cvae_model.pth")

# ----- CVAE with Auxiliary Regressor -----
class CVAEWithRegressor(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        self.kcat_head = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

    def predict_kcat(self, z, c):
        return self.kcat_head(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        y_pred = self.predict_kcat(z, c)
        return x_recon, mu, logvar, y_pred

def cvae_loss(x_recon, x, mu, logvar, y_pred, y_true, beta=0.001, gamma=7.0):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    reg_loss = nn.functional.mse_loss(y_pred.squeeze(), y_true, reduction='sum')
    total_loss = recon_loss + beta * kl_loss + gamma * reg_loss
    return total_loss, recon_loss, kl_loss, reg_loss

input_dim = X.shape[1]
cond_dim = C.shape[1]
latent_dim = 64
# ----- Training -----
model = CVAEWithRegressor(input_dim=X.shape[1], cond_dim=C.shape[1], latent_dim=64)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 100

train_losses, test_losses, pearsons = [], [], []
best_pearson = 0.68
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    recon_train, kl_train, reg_train = 0, 0, 0

    for x_batch, c_batch, y_batch in train_loader:
        x_recon, mu, logvar, y_pred = model(x_batch, c_batch)
        loss, recon_loss, kl_loss, reg_loss = cvae_loss(x_recon, x_batch, mu, logvar, y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        recon_train += recon_loss.item()
        kl_train += kl_loss.item()
        reg_train += reg_loss.item()

    train_losses.append(total_train_loss / len(X_train))

    # ---- Evaluation ----
    model.eval()
    preds, targets = [], []
    total_test_loss = 0
    recon_test, kl_test, reg_test = 0, 0, 0

    with torch.no_grad():
        for x_batch, c_batch, y_batch in test_loader:
            x_recon, mu, logvar, y_pred = model(x_batch, c_batch)
            loss, recon_loss, kl_loss, reg_loss = cvae_loss(x_recon, x_batch, mu, logvar, y_pred, y_batch)

            total_test_loss += loss.item()
            recon_test += recon_loss.item()
            kl_test += kl_loss.item()
            reg_test += reg_loss.item()

            preds.append(y_pred.squeeze().numpy())
            targets.append(y_batch.numpy())

    test_losses.append(total_test_loss / len(X_test))

    y_pred_all = np.concatenate(preds)
    y_true_all = np.concatenate(targets)
    pearson = pearsonr(y_pred_all, y_true_all)[0]
    pearsons.append(pearson)

    if pearson >= best_pearson:
        print('model_saved')
        best_pearson = pearson  # Update best Pearson
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'condition_dim': cond_dim,
            'latent_dim': latent_dim
        }, save_path)



    # print(f"Epoch {epoch+1:03d} | "
    #       f"Train Loss: {train_losses[-1]:.4f} (Recon: {recon_train/len(X_train):.2f}, KL: {kl_train/len(X_train):.2f}, Reg: {reg_train/len(X_train):.2f}) | "
    #       f"Test Loss: {test_losses[-1]:.4f} (Recon: {recon_test/len(X_test):.2f}, KL: {kl_test/len(X_test):.2f}, Reg: {reg_test/len(X_test):.2f}) | "
    #       f"Pearson: {pearson:.4f}")
    print(
        f"Epoch {epoch + 1:03d} | Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f} | Pearson: {pearson:.4f}")

# ----- Plot loss curves -----
plt.plot(range(2, len(train_losses) + 1), train_losses[1:], label='Train Loss')
plt.plot(range(2, len(test_losses) + 1), test_losses[1:], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Conditional VAE Training and Testing Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "cvae_loss_curve.png"), dpi=300)
plt.close()


# ----- Plot -----
plt.plot(pearsons, label='Pearson on Test')
plt.xlabel("Epoch")
plt.ylabel("Pearson r (log kcat)")
plt.title("CVAE w/ kcat Regressor")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "cvae_pearson.png"), dpi=300)
plt.close()
