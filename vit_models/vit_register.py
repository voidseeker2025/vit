import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.cuda.amp import autocast, GradScaler  
import csv
import time
from datetime import datetime


# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper ---
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# --- ViT Components ---
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.,
                 num_registers=4):  # <-- new arg: number of registers
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Positional embeddings now account for CLS + registers + patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1 + num_registers, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # 🚀 Register tokens
        self.register_tokens = nn.Parameter(torch.randn(1, num_registers, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)


    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        register_tokens = repeat(self.register_tokens, '1 r d -> b r d', b=b)

        # Concatenate CLS + registers + patches
        x = torch.cat((cls_tokens, register_tokens, x), dim=1)

        # Add position embeddings
        x += self.pos_embedding[:, : (n + 1 + self.register_tokens.size(1))]
        x = self.dropout(x)

        x = self.transformer(x)

        # Pooling: keep CLS only (registers are for representation, not classification head)
        x = x[:, 0] if self.pool == 'cls' else x.mean(dim=1)

        x = self.to_latent(x)
        return self.mlp_head(x)


# --- Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# --- Model ---
model = ViT(
    image_size=32,
    patch_size=4,
    num_classes=10,
    dim=256,
    depth=12,
    heads=3,
    mlp_dim=512,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

# --- Optimizer, Scaler, Loss ---
optimizer = optim.Adam(model.parameters(), lr=3e-4)
scaler = GradScaler()  # Mixed precision
epochs = 20 # best model around  ep12

train_losses, test_losses, train_accs, test_accs = [], [], [], []

# --- Training Loop ---
epoch_times = []

for epoch in range(epochs):
    start_time = time.time()

    model.train()
    train_loss, correct, total = 0, 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(): 
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(loss=loss.item())

    train_acc = correct / total
    train_losses.append(train_loss / len(train_loader))
    train_accs.append(train_acc)

    # --- Evaluation ---
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(): 
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
            test_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    test_losses.append(test_loss / len(test_loader))
    test_accs.append(test_acc)
    end_time = time.time()
    epoch_duration = end_time - start_time
    epoch_times.append(epoch_duration)

    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# --- Plot Results ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Accuracy")
plt.plot(test_accs, label="Test Accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()
plt.show()

# --- Save Training Data to CSV ---
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_path = os.path.join("vit", "vit_result")
csv_filename = os.path.join(folder_path, "vit_overlap.csv")
total_time = sum(epoch_times)

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Header
    writer.writerow(["Epoch", "Train Loss", "Test Loss", "Train Acc", "Test Acc","Epoch Time (s)"])
    writer.writerow([])
    writer.writerow(["Total Training Time (s)", total_time])

    # Epoch-wise data
    for epoch in range(epochs):
        writer.writerow([
            epoch + 1,
            train_losses[epoch],
            test_losses[epoch],
            train_accs[epoch],
            test_accs[epoch],
            epoch_times[epoch]
        ])


print(f"\nTraining log saved to {csv_filename}")

