import toml
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
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

class LNNmodelv3(pl.LightningModule):
    '''
    this class defines the LNN v2 model, a most simple LSTM model with 2 layers and 128 hidden units.
    '''
    def __init__(self, input_size, hidden_size=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.criterion = nn.BCEWithLogitsLoss()
        # set up the accuracy score metrics
        self.acc = torchmetrics.Accuracy(task='binary')

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        loss, pred_y, y = self._common_step(batch, batch_idx)
        pred_y = torch.argmax(pred_y, dim=-1)
        label_acc = self.acc(pred_y, y)
        self.log({'train_loss': loss, 'train_acc': label_acc}, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred_y, y = self._common_step(batch, batch_idx)
        pred_y = torch.argmax(pred_y, dim=-1)
        label_acc = self.acc(pred_y, y)
        self.log({'val_loss': loss, 'val_acc': label_acc}, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def _common_step(self, batch, batch_idx):
        X, y = batch
        y = y.to(torch.int64)
        pred_y = self.model(X)
        pred_y = pred_y.float()
        label_loss = self.criterion(pred_y, y)
        return label_loss, pred_y, y
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading validate data")
    val_dataloader = load_data("validate_data", batch_size=batch_size)

    model = LNNmodelv3(input_size=1028, hidden_size=128, dropout=0.1)

    ckpt_callbacks = ModelCheckpoint(
        dirpath='./log',
        filename='{epoch}-' + '{val_acc:.4f}-{val_loss:.4f}',
        every_n_epochs=1,
        save_top_k= -1,
    )
    
    trainer = Trainer(
        logger = pl.loggers.TensorBoardLogger(save_dir='./log'),
        max_epochs=100,
        strategy=DeepSpeedStrategy(),
        accelerator = 'gpu',
        default_root_dir='./log',
        profiler='simple'
    )

    print(f"loading train data")
    train_dataloader = load_data("train_data", batch_size=batch_size)
    trainer.fit(model, train_dataloader, val_dataloader)

    torch.save(model.state_dict(), "LNN_final_model.pth")
                        
           
    