import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torchinfo import summary

# -----------------------------
# Create output folder
# -----------------------------
os.makedirs("test", exist_ok=True)

# -----------------------------
# Dummy inputs (for ONNX export)
# -----------------------------
dummy_img = torch.randn(1, 1, 28, 28)
dummy_flat = torch.randn(1, 28 * 28)
dummy_img_24 = torch.randn(1, 24, 256, 256)


# -----------------------------
# Model 1: Simple CNN
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 13 * 13, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


# -----------------------------
# Model 2: Deep CNN
# -----------------------------
class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


# -----------------------------
# Model 3: MLP
# -----------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# -----------------------------
# Model 4: CNN + MLP
# -----------------------------
class CNN_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4 * 13 * 13, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# -----------------------------
# Model 5: Tiny
# -----------------------------
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.fc(x)


# -----------------------------
# Model 6: TinyCNN
# -----------------------------
class TinyCNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv2d(24, 32, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.cnn(x)
        return x


# -----------------------------
# Instantiate models
# -----------------------------
models = {
    "model1": (SimpleCNN(), dummy_img),
    "model2": (DeepCNN(), dummy_img),
    "model3": (MLP(), dummy_flat),
    "model4": (CNN_MLP(), dummy_img),
    "model5": (TinyNet(), dummy_flat),
    "model6": (TinyCNNNet(), dummy_img_24)
}

# -----------------------------
# Save BOTH .pt and .onnx
# -----------------------------
for name, (model, dummy) in models.items():
    model.eval()

    pt_path = f"test/{name}.pt"
    onnx_path = f"test/{name}.onnx"

    # -------------------------
    # Save PyTorch model
    # -------------------------
    torch.save(
        {"state_dict": model.state_dict(), "model_class": model.__class__.__name__},
        pt_path,
    )

    # -------------------------
    # Export ONNX model
    # -------------------------
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
    )

    print(f"✅ Saved {pt_path} and {onnx_path}")

# -----------------------------
# Summary
# -----------------------------
summary(SimpleCNN(), input_size=(1, 1, 28, 28))

print("🎉 All models saved in .pt and .onnx formats")
