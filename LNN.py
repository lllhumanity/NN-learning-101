import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch.nn as nn
import random
from tqdm import tqdm
import os

# 通用流式数据集（支持训练/验证集），一次 mmap 读取 + 输出 CPU 张量
class ChunkIterableDataset(IterableDataset):
    def __init__(self, feature_chunks, label_chunks, shuffle=False):
        self.shuffle = shuffle
        assert len(feature_chunks) == len(label_chunks), "特征与标签chunk数量应一致"

        self.data = []
        for fx_path, fy_path in zip(feature_chunks, label_chunks):
            fx_data = np.load(fx_path, mmap_mode='r')
            fy_data = np.load(fy_path, mmap_mode='r')
            self.data.append((fx_data, fy_data))

    def __iter__(self):
        chunk_indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(chunk_indices)

        for idx in chunk_indices:
            fx_data, fy_data = self.data[idx]
            sample_indices = list(range(len(fx_data)))
            if self.shuffle:
                random.shuffle(sample_indices)

            for i in sample_indices:
                fx = fx_data[i].flatten()
                fy = fy_data[i]
                yield torch.tensor(fx, dtype=torch.float32), torch.tensor(fy, dtype=torch.float32)


class TemporalClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


torch.backends.cudnn.benchmark = True

def train_model(train_feature_chunks, train_label_chunks,
                test_feature_chunks, test_label_chunks,
                num_epochs=100, batch_size=100000,
                checkpoint_path="checkpoint.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备信息] 当前使用设备: {device}")
    if device.type == 'cuda':
        print(f"[GPU名称] {torch.cuda.get_device_name(device)}")

    train_dataset = ChunkIterableDataset(train_feature_chunks, train_label_chunks, shuffle=True)
    test_dataset = ChunkIterableDataset(test_feature_chunks, test_label_chunks, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    sample_input = next(iter(train_loader))[0].to(device)
    model = TemporalClassifier(input_size=sample_input.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    best_accuracy = 0.0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f"[恢复训练] 从第 {start_epoch} 轮继续，最佳准确率为 {best_accuracy:.4f}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        with tqdm(desc=f"Epoch {epoch+1:03d} [Train]", unit="batch") as pbar:
            for batch_idx, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(X).squeeze()
                    loss = criterion(outputs, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * X.size(0)
                pbar.update(1)

        model.eval()
        test_loss, correct = 0.0, 0
        with torch.no_grad():
            with tqdm(desc=f"Epoch {epoch+1:03d} [Test]", unit="batch") as pbar_test:
                for batch_idx, (X, y) in enumerate(test_loader):
                    X, y = X.to(device), y.to(device)
                    with torch.cuda.amp.autocast():
                        outputs = model(X).squeeze()
                        test_loss += criterion(outputs, y).item() * X.size(0)
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        correct += (preds == y).sum().item()
                    pbar_test.update(1)

        train_loss = train_loss / ((batch_idx + 1) * batch_size)
        test_loss = test_loss / ((batch_idx + 1) * batch_size)
        accuracy = correct / ((batch_idx + 1) * batch_size)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "LNN_best_model.pth")
            print(f"[保存最佳模型] Epoch {epoch+1:03d} 准确率提升至 {best_accuracy:.4f}")

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_accuracy': best_accuracy
        }, checkpoint_path)

    return model


if __name__ == "__main__":
    train_features = [
        "train_data/chunk_0.fX.npy",
        "train_data/chunk_1.fX.npy"
    ]
    train_labels = [
        "train_data/chunk_0.fy.npy",
        "train_data/chunk_1.fy.npy"
    ]
    test_features = [
        "validate_data/chunk_0.fX.npy",
        "validate_data/chunk_1.fX.npy"
    ]
    test_labels = [
        "validate_data/chunk_0.fy.npy",
        "validate_data/chunk_1.fy.npy"
    ]

    model = train_model(
        train_feature_chunks=train_features,
        train_label_chunks=train_labels,
        test_feature_chunks=test_features,
        test_label_chunks=test_labels,
        num_epochs=100,
        batch_size=100000
    )

    torch.save(model.state_dict(), "LNN_classifier.pth")
