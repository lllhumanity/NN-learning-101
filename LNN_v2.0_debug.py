import toml
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import gc
import sys

batch_size = 256
train_data_dir = "train_data"
test_data_dir = "validate_data"
checkpoint_path = "checkpoint.pth"

class classificationDataset(Dataset):
    def __init__(self, feature_chunks, label_chunks):
        self.feature_chunks = feature_chunks
        self.label_chunks = label_chunks
        assert len(self.feature_chunks) == len(self.label_chunks)

    def __getitem__(self, index):
        return self.feature_chunks[index], self.label_chunks[index]

    def __len__(self):
        return len(self.feature_chunks)

def load_data(outdir: str, batch_size: int, ndp: int):
    fX = np.zeros((ndp, 4, 257), dtype=np.float32)
    fy = np.zeros((ndp,), dtype=np.float32)
    chunk_idx = 0
    feature_chunk_file = os.path.join(outdir, f"chunk_{chunk_idx}.fX.npy")
    label_chunk_file = os.path.join(outdir, f"chunk_{chunk_idx}.fy.npy")
    fX_chunk = np.load(feature_chunk_file)
    fy_chunk = np.load(label_chunk_file)
    fX[0:ndp, :, :] = fX_chunk
    fy[0:ndp] = fy_chunk
    print(f'[Info] load features finished! features shape: {fX.shape}, labels shape: {fy.shape}')

    dataset = classificationDataset(fX, fy)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader

class LNNmodelv2(nn.Module):
    def __init__(self, device, input_size, hidden_size=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)

def initialize_model(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)

def train_model(num_epochs=10, batch_size=256, checkpoint_path="checkpoint.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device info] using device: {device}")
    if device.type == 'cuda':
        print(f"[GPU name] {torch.cuda.get_device_name(device)}")

    model = LNNmodelv2(device=device, input_size=1028, hidden_size=128, dropout=0.1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.apply(initialize_model)
    start_epoch = 0
    best_accuracy = 0.0

    # load data
    print("loading validate data")
    val_dataloader = load_data("validate_data", batch_size=batch_size, ndp=840484)
    print("loading train data")
    train_dataloader = load_data("train_data", batch_size=batch_size, ndp=842057)

    # load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f"[checkpoint loaded] from epoch {start_epoch}, best acc: {best_accuracy:.4f}")

    # training
    model.to(device)
    print("[info] start training")
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as profiler:
        for epoch in range(start_epoch, num_epochs):
            print(f"Epoch {epoch+1} start")
            pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (X, y) in enumerate(pbar):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type):
                    outputs = model(X).squeeze()
                    loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({"loss": loss.item()})
            profiler.step()

    # ==== SAFE SAVING ====
    model.to("cpu")  # move off GPU
    torch.save(model.state_dict(), "LNN_final_model.pth")

    # clear memory
    del model, train_dataloader, val_dataloader, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()
    print("[info] model saved and resources cleaned up")

    sys.exit(0)

if __name__ == "__main__":
    train_model(
        num_epochs=10,
        batch_size=256,
        checkpoint_path="checkpoint.pth"
    )
