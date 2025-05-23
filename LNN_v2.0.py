import toml
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

batch_size = 256
train_data_dir = "train_data"
test_data_dir = "validate_data"
checkpoint_path = "checkpoint.pth"

class classificationDataset(Dataset):
    '''
    this class is a dataset for the classification task, it will load the data from the feature and label chunks, and return the data and label in the form of tensor.
    '''
    def __init__(self, feature_chunks, label_chunks):
        self.feature_chunks = feature_chunks
        self.label_chunks = label_chunks
        assert len(self.feature_chunks) == len(self.label_chunks), "feature should have the same number of samples as label"

    def __getitem__(self, index):
        return self.feature_chunks[index], self.label_chunks[index]

    def __len__(self):
        return len(self.feature_chunks)

def load_data(outdir: str, batch_size: int):
    '''
    this function is used to load the data from the feature and label chunks, and return the data and label in the form of tensor.
    '''
    #read metadata file
    metadata_file = os.path.join(outdir, "metadata.toml")
    metadata = toml.load(metadata_file)
    feature_chunk_sizes = metadata["feature_chunk_sizes"]
    
    ndp = np.sum(feature_chunk_sizes)
    n_feature_chunks = len(feature_chunk_sizes)
    
    #initialize output
    fX = np.zeros((ndp, 4, 257), dtype=np.uint8)
    fy = np.zeros((ndp,), dtype=np.uint8)
    start_idx = 0
    for chunk_idx in range(n_feature_chunks):
        feature_chunk_file = os.path.join(outdir, f"chunk_{chunk_idx}.fX.npy")
        label_chunk_file = os.path.join(outdir, f"chunk_{chunk_idx}.fy.npy")
        fX_chunk = np.load(feature_chunk_file, mmap_mode='r')
        fy_chunk = np.load(label_chunk_file, mmap_mode='r')
        end_idx = start_idx + len(fy_chunk)
        fX[start_idx:end_idx, :, :] = fX_chunk
        fy[start_idx:end_idx] = fy_chunk
        start_idx = end_idx
    print(f'[Info] load features finished! features shape: {fX.shape}, labels shape: {fy.shape}')
    
    #construct dataset and dataloader
    dataset = classificationDataset(fX, fy)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader
    


#loading train and test data
print(f"loading train data")
train_dataloader = DataLoader(
    classificationDataset(
        feature_chunks=["train_data/chunk_0.fX.npy", "train_data/chunk_1.fX.npy"],
        label_chunks=["train_data/chunk_0.fy.npy", "train_data/chunk_1.fy.npy"]
    ),
    batch_size=batch_size,
    num_workers=4,
    shuffle=True
)   

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

def train_model(
    num_epochs=100,
    batch_size = 256,
    checkpoint_path = "checkpoint.pth"
):
    '''
    this function trains the model on the training data, and evaluates the model on the test data.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device info] current using device: {device}")
    if device.type == 'cuda':
        print(f"[GPU name] {torch.cuda.get_device_name(device)}")
    
    #1. model initialization
    model = LNNmodelv2(
        device=device,
        input_size = 1028,
        hidden_size = 128,
        dropout = 0.1
    )
    start_epoch = 0
    best_accuracy = 0.0
    criterion = nn.BCEWithLogitsLoss()                          #binary cross entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  #adam optimizer
    model.to(device)                                            #move model to device
    model.apply(initialize_model)                               #initialize model weight
    
    #2. load validate data once
    print(f"loading validate data")
    val_dataloader = load_data("validate_data", batch_size=batch_size)
    print(f"validate data loaded")
    
    #3. ckpt loading
    if os.path.exists(checkpoint_path):
        train_from_ckpt = True
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f"[checkpoint loaded] loaded checkpoint from epoch {start_epoch}, best accuracy: {best_accuracy}")
    
    #4. torch profiler and training
    print(f"[info]start training")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as profiler:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_loss = 0.0
            train_dataloader = load_data("train_data", batch_size=batch_size)
            with tqdm as pbar:
                pbar.set_description(f"Training Epoch {epoch+1}/{num_epochs}")
                for batch_idx, (X, y) in enumerate(train_dataloader):
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    with torch.autocast(device_type=device.type):
                        outputs = model(X).squeeze()
                        loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * X.size(0)
                    pbar.update(1)
                    pbar.set_postfix({"loss": loss.item()})
            profiler.step()
            
            #5. evaluate model on validate data
            model.eval()
            test_loss = 0.0
            correct = 0
            with torch.no_grad():
                with tqdm as pbar_test:
                    pbar_test.set_description(f"Validating Epoch {epoch+1}/{num_epochs}")
                    for batch_idx, (X, y) in enumerate(val_dataloader):
                        X, y = X.to(device), y.to(device)
                        with torch.autocast(device_type=device.type):
                            outputs = model(X).squeeze()
                            test_loss += criterion(outputs, y).item() * X.size(0)
                            preds = (torch.sigmoid(outputs) > 0.5).float()
                            correct += (preds == y).sum().item()
                        pbar_test.update(1)
            
                        pbar_test.set_postfix({"loss": test_loss.item()})
            train_loss = train_loss / ((batch_idx + 1) * batch_size)
            test_loss = test_loss / ((batch_idx + 1) * batch_size)
            accuracy = correct / ((batch_idx + 1) * batch_size)
            
            print(f"Epoch {epoch+1:03d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Test Loss: {test_loss:.4f} | "
                    f"Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), "LNN_best_model.pth")
                print(f"[saving] Epoch {epoch+1:03d} accuracy to {best_accuracy:.4f}")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_accuracy': best_accuracy
            }, checkpoint_path)
            
    return model


if __name__ == "__main__":
    train_model(
        num_epochs=100,
        batch_size = 256,
        checkpoint_path = "checkpoint.pth"
    )
    
    torch.save(model.state_dict(), "LNN_final_model.pth")
                        
           
    