import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# ---- Model ----
class BigCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BigCNN, self).__init__()
        # Input: (1, 28, 28) for MNIST
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),   # (64, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (128, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # (128, 14, 14)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),# (256, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# (256, 14, 14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # (256, 7, 7)

            nn.Conv2d(256, 512, kernel_size=3, padding=1),# (512, 7, 7)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# (512, 7, 7)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),                 # (512, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ---- Training Loop ----
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# ---- Evaluation ----
def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    acc = correct / len(test_loader.dataset)
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    # Device setup
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("[WARN] CUDA not available, using CPU")

    # Data
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model, optimizer, loss
    model = BigCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Track memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Training
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion)
        acc = evaluate(model, device, test_loader)
        print(f"Epoch {epoch}, Test Accuracy: {acc*100:.2f}%")

    # Memory usage
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"[INFO] Peak GPU memory usage: {peak_mem:.2f} MB")

if __name__ == "__main__":
    main()