import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import SimpleCNN, CLASSES

EPOCHS     = 20
BATCH_SIZE = 64
LR         = 0.001
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Training on: {DEVICE.upper()}")

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

train_data = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_transform)
test_data  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=(DEVICE=="cuda"))
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=256,        shuffle=False, num_workers=2, pin_memory=(DEVICE=="cuda"))

print(f"Train: {len(train_data):,}  |  Test: {len(test_data):,}")

model     = SimpleCNN().to(DEVICE)
loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")


def run_epoch(loader, training):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images)
            loss  = loss_fn(preds, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct    += (preds.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


train_losses, test_losses   = [], []
train_accs,   test_accs     = [], []
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    train_loss, train_acc = run_epoch(train_loader, training=True)
    test_loss,  test_acc  = run_epoch(test_loader,  training=False)
    scheduler.step()

    train_losses.append(train_loss); test_losses.append(test_loss)
    train_accs.append(train_acc);   test_accs.append(test_acc)

    print(f"Epoch {epoch:2d}/{EPOCHS} | Train {train_acc:.1f}% | Test {test_acc:.1f}% | Loss {test_loss:.3f} | {time.time()-t0:.0f}s")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "best_model.pt")
        print(f"           ✓ Saved (best={best_acc:.1f}%)")

print(f"\nDone. Best test accuracy: {best_acc:.1f}%")

epochs_range = range(1, EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs_range, train_accs,   label="Train"); ax1.plot(epochs_range, test_accs,   label="Test")
ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("%"); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(epochs_range, train_losses, label="Train"); ax2.plot(epochs_range, test_losses, label="Test")
ax2.set_title("Loss");     ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("Curves saved → training_curves.png")
