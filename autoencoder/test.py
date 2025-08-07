import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ---- 1. Autoencoder Definition ----
class JointAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(),
            nn.Linear(32, 2)  # 2D latent space
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# ---- 2. Synthetic Joint Space Data ----
def generate_joint_sequence(offset=0.0, scale=1.0, noise=0.0):
    t = np.linspace(0, 2 * np.pi, 100)
    hip = scale * np.sin(t + offset) + noise * np.random.randn(len(t))
    knee = scale * np.sin(t + np.pi/4 + offset) + noise * np.random.randn(len(t))
    ankle = scale * np.sin(t + np.pi/2 + offset) + noise * np.random.randn(len(t))
    return np.stack([hip, knee, ankle, hip, knee, ankle], axis=1)

# 데이터 생성
normal = generate_joint_sequence()
hemi = generate_joint_sequence(offset=0.2, scale=0.8, noise=0.01)

# 정규화
mean, std = normal.mean(0), normal.std(0)
normal = (normal - mean) / std
hemi = (hemi - mean) / std

# Tensor 변환
x_normal = torch.tensor(normal, dtype=torch.float32)
x_hemi = torch.tensor(hemi, dtype=torch.float32)

# ---- 3. Train AE with Δz → Δx supervision ----
model = JointAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
lambda_delta = 1.0  # Δx 학습 가중치

for epoch in range(500):
    model.train()

    # 기본 재구성 학습 (normal → normal)
    x_hat, z_normal = model(x_normal)
    recon_loss = loss_fn(x_hat, x_normal)

    # Δz → Δx 보상 학습
    z_hemi = model.encoder(x_hemi)  # 따로 detach 안 함: 같이 학습 가능
    delta_z = z_hemi - z_normal
    delta_x_hat = model.decoder(delta_z)
    delta_x_true = x_hemi - x_normal
    delta_loss = loss_fn(delta_x_hat, delta_x_true)

    total_loss = recon_loss + lambda_delta * delta_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"[Epoch {epoch}] Recon: {recon_loss.item():.6f}, Δx loss: {delta_loss.item():.6f}")

# ---- 4. Evaluate latent + decoded Δx ----
model.eval()
with torch.no_grad():
    _, z_normal = model(x_normal)
    _, z_hemi = model(x_hemi)
    delta_z = z_hemi - z_normal
    delta_x_hat = model.decoder(delta_z)
    delta_x_true = x_hemi - x_normal

    delta_x_hat_np = delta_x_hat.detach().numpy()
    delta_x_np = delta_x_true.detach().numpy()

# ---- 5. Plot: Raw Δx vs Decoded Δz ----
joint_names = ['Hip_L', 'Knee_L', 'Ankle_L', 'Hip_R', 'Knee_R', 'Ankle_R']
plt.figure(figsize=(12, 8))

for i in range(6):
    plt.subplot(3, 2, i+1)
    plt.plot(delta_x_np[:, i], label='Raw Δx', linestyle='--')
    plt.plot(delta_x_hat_np[:, i], label='Decoded Δz', linestyle='-')
    plt.title(joint_names[i])
    plt.xlabel("Time step")
    plt.ylabel("Δ Joint (standardized)")
    plt.legend()
    plt.grid(True)

plt.suptitle("Comparison: Raw Joint Difference vs. Decoded Δz")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
