import torch

from model import SimpleSuperResolutionCNN

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)

model = SimpleSuperResolutionCNN().to(device)

print(model)

x = torch.randn(
    1,
    3,
    64,
    64
).to(device)


with torch.no_grad():
    y = model(x)

print("\n Input shape : ", x.shape)
print("\n Output shape : ", y.shape)