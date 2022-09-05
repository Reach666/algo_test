import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
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

        self.example_input_array = torch.Tensor(32, 1, 28, 28)
        # self.save_hyperparameters()
        # self.learning_rate = 0
        # self.batch_size = 1

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.model(x, **kwargs)
        return out

    def training_step(self, batch, batch_idx=0):
        data, target = batch
        output = self.forward(data)
        loss = F.nll_loss(output, target)
        self.log('train_loss',loss)
        return loss

    def validation_step(self, batch, batch_idx=0):
        data, target = batch
        output = self.forward(data)
        loss = F.nll_loss(output, target, reduction='mean')
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        correct_num = pred.eq(target.view_as(pred)).sum()
        batch_num = target.shape[0]
        self.log_dict({'val_step_loss': loss, 'val_step_correct':correct_num/batch_num})
        return loss*batch_num,correct_num,batch_num # pred

    def validation_epoch_end(self, validation_step_outputs):
        losses = [item[0] for item in validation_step_outputs]
        correct_nums = [item[1] for item in validation_step_outputs]
        batch_nums = [item[2] for item in validation_step_outputs]
        all_correct_num = sum(correct_nums)
        all_batch_num = sum(batch_nums)
        correct_ratio = all_correct_num/all_batch_num
        epoch_loss = sum(losses)/all_batch_num
        print("\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(epoch_loss,
                                                all_correct_num, all_batch_num, 100. * correct_ratio))
        self.log_dict({'val_epoch_loss':epoch_loss, 'val_epoch_correct':correct_ratio})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def on_train_start(self) -> None:
        self.logger.experiment.add_graph(self,self.example_input_array.to(self.device))




if __name__ == "__main__":
    pl.seed_everything(1234)

    mnist_train = MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    mnist_val = MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    # transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1037,), (0.3081,))])
    train_loader = DataLoader(mnist_train, batch_size=512, shuffle=True, num_workers=0) # 大概需要2G的显存
    val_loader = DataLoader(mnist_val, batch_size=512, num_workers=0)

    model_lit = LITCNN() # LITCNN.load_from_checkpoint(PATH)
    # tb_logger = pl.loggers.TensorBoardLogger(save_dir="logs/") # logger=tb_logger
    trainer = pl.Trainer(limit_train_batches=1.0,max_epochs=20,accelerator='auto',precision=16,
                         callbacks=[
                             ModelCheckpoint(save_top_k=2,
                                             dirpath="./checkpoints",
                                             monitor="val_epoch_loss",
                                             save_last=True)
                         ],
                         # profiler="simple" # Find training loop bottlenecks
                         )
    trainer.fit(model_lit,train_loader,val_loader) # ckpt_path=""

    # # trainer = pl.Trainer(auto_scale_batch_size=True, ) # the self.batch_size of train_loader should be scaleable
    # trainer = pl.Trainer(auto_lr_find=True)
    # trainer.tune(model_lit,train_loader,val_loader)



