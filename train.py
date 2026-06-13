from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = MNISTCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

writer = SummaryWriter(log_dir="logs/mnist_run1")

total_step = 0
epoch_num = 3
for epoch in range(epoch_num):
    model.train()
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        writer.add_scalar("Train/Loss", loss.item(), total_step)
        total_step += 1

    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = model(imgs)
            test_loss += criterion(outputs, labels).item()
            _, pred = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    acc = correct / total
    avg_test_loss = test_loss / len(test_loader)

    writer.add_scalar("Test/Loss", avg_test_loss, epoch)
    writer.add_scalar("Test/Accuracy", acc, epoch)

images, _ = next(iter(train_loader))
writer.add_graph(model, images)
writer.add_images("Train/Samples", images[:16])

writer.close()
print(f"训练完成，启动可视化命令：tensorboard --logdir=logs")
