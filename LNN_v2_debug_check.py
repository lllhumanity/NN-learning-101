import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from memory_profiler import profile

batch_size = 25600
train_data_dir = "train_data"
test_data_dir = "validate_data"
checkpoint_path = "checkpoint.pth"


class classificationDataset(Dataset):
    '''
    this class is a dataset for the classification task, it will load the data from the feature and label chunks, and return the data and label in the form of tensor.
    '''

    def __init__(self, feature_chunks, label_chunks, device):
        self.feature_chunks = torch.tensor(feature_chunks, device=device, dtype=torch.float32)
        self.label_chunks = torch.tensor(label_chunks, device=device, dtype=torch.float32)
        assert len(self.feature_chunks) == len(
            self.label_chunks), "feature should have the same number of samples as label"

    def __getitem__(self, index):
        return self.feature_chunks[index], self.label_chunks[index]

    def __len__(self):
        return len(self.feature_chunks)


def load_data(outdir: str, batch_size: int, ndp: int, device):
    '''
    this function is used to load the data from the feature and label chunks, and return the data and label in the form of tensor.
    '''
    # read metadata file
    # metadata_file = os.path.join(outdir, "metadata.toml")
    # metadata = toml.load(metadata_file)
    # feature_chunk_sizes = metadata["feature_chunk_sizes"]

    # ndp = np.sum(feature_chunk_sizes)
    # n_feature_chunks = len(feature_chunk_sizes)

    # initialize output
    fX = np.zeros((ndp, 4, 257), dtype=np.float32)
    fy = np.zeros((ndp,), dtype=np.float32)
    chunk_idx = 0
    # start_idx = 0
    # for chunk_idx in range(n_feature_chunks):
    #    feature_chunk_file = os.path.join(outdir, f"chunk_{chunk_idx}.fX.npy")
    #    label_chunk_file = os.path.join(outdir, f"chunk_{chunk_idx}.fy.npy")
    #    fX_chunk = np.load(feature_chunk_file, mmap_mode='r')
    #    fy_chunk = np.load(label_chunk_file, mmap_mode='r')
    #    end_idx = start_idx + len(fy_chunk)
    #    fX[start_idx:end_idx, :, :] = fX_chunk
    #    fy[start_idx:end_idx] = fy_chunk
    #    start_idx = end_idx
    feature_chunk_file = os.path.join(outdir, f"chunk_{chunk_idx}.fX.npy")
    fX_chunk = np.load(feature_chunk_file)
    label_chunk_file = os.path.join(outdir, f"chunk_{chunk_idx}.fy.npy")
    fy_chunk = np.load(label_chunk_file)
    fX[0:ndp, :, :] = fX_chunk
    fy[0:ndp, ] = fy_chunk
    print(f'[Info] load features finished! features shape: {fX.shape}, labels shape: {fy.shape}')

    # construct dataset and dataloader
    dataset = classificationDataset(fX, fy, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type != 'cuda'))
    return dataloader


class LNNmodelv2(nn.Module):
    '''
    this class defines the LNN v2 model, a most simple LSTM model with 2 layers and 128 hidden units.
    '''

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
    '''
    this function is meant for linear layers' weight initialization
    '''
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)


@profile(precision=4, stream=open('check_log.txt', 'w'))
def train_model(num_epochs=10, batch_size=256, checkpoint_path="checkpoint.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device info] current using device: {device}")
    if device.type == 'cuda':
        print(f"[GPU name] {torch.cuda.get_device_name(device)}")
        torch.cuda.reset_peak_memory_stats(device)
        print(f"Initial GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"Initial GPU Memory Reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

    # Model initialization
    model = LNNmodelv2(device=device, input_size=1028, hidden_size=128, dropout=0.1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.apply(initialize_model)
    model.to(device)
    if device.type == 'cuda':
        print(f"After Model Init - GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"After Model Init - GPU Memory Reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

    # Load data
    print(f"loading validate data")
    val_dataloader = load_data("validate_data/", batch_size=batch_size, ndp=840484, device=device)
    print(f"loading train data")
    train_dataloader = load_data("train_data/", batch_size=batch_size, ndp=842057, device=device)
    if device.type == 'cuda':
        print(f"After Data Load - GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"After Data Load - GPU Memory Reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")


    model.to(device)
    best_accuracy = 0.0

    # 4. torch profiler and training
    print(f"[info]start training")
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
    ) as profiler:
        # Training loop
        for epoch in range(num_epochs):
            print(f"epoch {epoch + 1} start")
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch_idx, (X, y) in enumerate(pbar):
                model.train()
                # X, y = X.to(device), y.to(device)
                if device.type == 'cuda' and batch_idx % 1000 == 0:  # Log every 10 batches
                    print(f"model on: {next(model.parameters()).device}")
                    print(f"data on: {X.device}")
                    print(f"Batch {batch_idx} - GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
                    print(f"Batch {batch_idx} - GPU Memory Reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
                # Add forward and backward pass for completeness
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs.squeeze(), y)
                loss.backward()
                optimizer.step()

            # model.eval()
            # test_loss = 0.0
            # correct = 0
            # with torch.no_grad():
            #     pbar_test = tqdm(val_dataloader, desc=f"Validating Epoch {epoch + 1}/{num_epochs}")
            #     for batch_idx, (X, y) in enumerate(pbar_test):
            #         X, y = X.to(device), y.to(device)
            #         with torch.autocast(device_type=device.type):
            #             outputs = model(X).squeeze()
            #             test_loss += criterion(outputs, y).item() * X.size(0)
            #             preds = (torch.sigmoid(outputs) > 0.5).float()
            #             correct += (preds == y).sum().item()
            #         pbar_test.set_postfix({"loss": test_loss / ((batch_idx + 1) * batch_size)})
            #     pbar_test.close()

            # # train_loss = train_loss / ((batch_idx + 1) * batch_size)
            # test_loss = test_loss / ((batch_idx + 1) * batch_size)
            # accuracy = correct / ((batch_idx + 1) * batch_size)

            # print(f"Epoch {epoch + 1:03d} | " 
            #       f"Test Loss: {test_loss:.4f} | "
            #       f"Accuracy: {accuracy:.4f}")

            # if accuracy > best_accuracy:
            #     best_accuracy = accuracy
            #     torch.save(model.state_dict(), "LNN_best_model.pth")
            #     print(f"[saving] Epoch {epoch + 1:03d} accuracy to {best_accuracy:.4f}")
            # torch.save({
            #     'epoch': epoch,
            #     'model': model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            #     'best_accuracy': best_accuracy
            # }, checkpoint_path)

            pbar.close()
            if device.type == 'cuda':
                print(f"End of Epoch {epoch + 1} - Peak GPU Memory Allocated: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
                print(f"End of Epoch {epoch + 1} - Peak GPU Memory Reserved: {torch.cuda.max_memory_reserved(device) / 1024**2:.2f} MB")

    return model


if __name__ == "__main__":
    model = train_model(
        num_epochs=10,
        batch_size=256,
        checkpoint_path="checkpoint.pth"
    )

    torch.save(model.state_dict(), "LNN_final_model.pth")


