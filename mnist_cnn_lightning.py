import os
import time
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,TQDMProgressBar
from tensorboard import program
from typing import Any, Callable, Dict, List, Mapping, Optional, overload, Sequence, Tuple, Union


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1*1*28*28
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(10, 12, kernel_size=(3,3))
        self.fc1 = nn.Linear(12 * 5 * 5, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)  # 1* 10 * 24 *24
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2,2), stride=(2,2))  # 1* 10 * 12 * 12
        out = self.conv2(out)  # 1* 12 * 10 * 10
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2,2), stride=(2,2))  # 1* 12 * 5 * 5
        out = out.view(in_size, -1)  # 1 * 300
        out = self.fc1(out)  # 1 * 20
        out = F.relu(out)
        out = self.fc2(out)  # 1 * 1
        out = F.log_softmax(out, dim=1)
        return out


class LITCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ConvNet()

        self.metric_mean_loss_train = torchmetrics.MeanMetric()
        self.metric_mean_loss_val = torchmetrics.MeanMetric()
        self.metric_accuracy_train = torchmetrics.Accuracy(task='multiclass', num_classes=10, average='micro')  # average  'macro'->mAcc 'micro'->OA
        self.metric_accuracy_val = torchmetrics.Accuracy(task='multiclass', num_classes=10, average='micro')

        self.example_input_array = torch.Tensor(32, 1, 28, 28)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.model(x, **kwargs)
        return out

    def training_step(self, batch, batch_idx=0):
        data, target = batch
        preds = self.forward(data)
        loss = F.nll_loss(preds, target)

        # l2_regularization_fc2 = torch.norm(self.model.fc2.weight, 2)
        # loss += 1e-3 * l2_regularization_fc2
        inner_products = torch.mm(self.model.fc1.weight,self.model.fc1.weight.t())
        upper_triangular_values = inner_products.triu().flatten()
        loss += 30 * upper_triangular_values.abs().mean()
        loss += 1 * upper_triangular_values.abs().max()

        preds = torch.exp(preds)
        self.metric_mean_loss_train(loss, len(target))
        self.metric_accuracy_train(preds, target)
        self.log('train_loss_step', loss, on_step=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.logger.experiment.add_scalars('_/loss', {'train': self.metric_mean_loss_train.compute().item()}, global_step=self.global_step)
        self.metric_mean_loss_train.reset()
        self.logger.experiment.add_scalars('_/Accuracy', {'train': self.metric_accuracy_train.compute().item()}, global_step=self.global_step)
        self.metric_accuracy_train.reset()

    def validation_step(self, batch, batch_idx=0):
        data, target = batch
        preds = self.forward(data)
        loss = F.nll_loss(preds, target)

        preds = torch.exp(preds)
        self.metric_mean_loss_val(loss, len(target))
        self.metric_accuracy_val(preds, target)
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)  # training_step和validation_step是不一致的，self.global_step默认为training_step，所以on_step在validation中默认为False
        self.log('val_accuracy', self.metric_accuracy_val, on_epoch=True, on_step=False, prog_bar=True)  # reduce_fx=torch.mean 会自动适应变化的batch_size

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        self.logger.experiment.add_scalars('_/loss', {'val': self.metric_mean_loss_val.compute().item()}, global_step=self.global_step)
        self.metric_mean_loss_val.reset()
        self.logger.experiment.add_scalars('_/Accuracy', {'val': self.metric_accuracy_val.compute().item()}, global_step=self.global_step)
        self.metric_accuracy_val.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def on_train_start(self) -> None:
        self.logger.experiment.add_graph(self,self.example_input_array.to(self.device))


class TQDMProgressBarWithoutVal(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.close()
        # bar = tqdm(disable=True)
        return bar


if __name__ == "__main__":
    pl.seed_everything(1234)
    BATCH_SIZE = 512

    mnist_train = MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    mnist_val = MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    # transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1037,), (0.3081,))])
    train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # 大概需要2G的显存
    val_loader = DataLoader(mnist_val, batch_size=BATCH_SIZE, num_workers=0)

    model_lit = LITCNN()
    # tb_logger = pl.loggers.TensorBoardLogger(save_dir="logs/") # logger=tb_logger
    trainer = pl.Trainer(limit_train_batches=1.0,max_epochs=50,accelerator='auto',precision='16-mixed',
                         callbacks=[
                             TQDMProgressBarWithoutVal(),  # TQDMProgressBar(),
                             ModelCheckpoint(save_top_k=1,
                                             dirpath="./checkpoints",
                                             monitor="val_loss",
                                             save_last=True)
                         ],
                         # profiler="simple" # Find training loop bottlenecks
                         )
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', trainer.log_dir, '--port', '8008'])  # tensorboard --logdir ./
    url = tb.launch()  # 'http://localhost:6006/'
    os.system('explorer ' + url)

    trainer.fit(model_lit,train_loader,val_loader) # ckpt_path=""

    # # trainer = pl.Trainer(auto_scale_batch_size=True, ) # the self.batch_size of train_loader should be scaleable
    # trainer = pl.Trainer(auto_lr_find=True)
    # trainer.tune(model_lit,train_loader,val_loader)

    while True:
        time.sleep(2)



