import torch
import torchvision
import torchvision.transforms as transfroms

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy

class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step (self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x) 
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        return loss, acc
        

#data
transform = transfroms.Compose(
    [transfroms.ToTensor(),
    transfroms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

#학습 데이터 불러오기
trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform=transform)

#학습용 셋은 섞어서 뽑기
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 2)
val_loader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = False, num_workers = 2)
#데이터 불러오기
testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False, num_workers = 2)

#model
model = LitAutoEncoder()

#training
wandb_logger = WandbLogger(project="2018125048_이윤노_pytorch lightning Cifar10")
trainer = pl.Trainer(gpus = 1, precision = 16, limit_train_batches = 0.1, logger=wandb_logger)
trainer.fit(model, trainloader, val_loader)
trainer.test(model, testloader)
