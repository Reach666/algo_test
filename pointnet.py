import os
import time
import numpy as np
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader,random_split
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,TQDMProgressBar,EarlyStopping
from tensorboard import program
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm


def square_distance(src, dst) -> torch.Tensor:
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def load_dataset():
    B = 4096
    # # N_density和density_ratio都设为4的情况下，难度适中   两个任务叠加  # Acc   FC: 79.6%   PointNet(max): 79.6%   PointNet(avg): 93.0%   Transformer: 94.6%(会过拟合)  PointAttentionNet: 95.3%   PointWiseNet: 98.6%
    # N = 50
    # data = torch.rand(B, N, 3)
    # N_density = N // 4  # the bigger N_density, the harder
    # density_ratio = 4  # the bigger density_ratio, the easier
    # data[:, :N_density, :] = data[:, :N_density, :] / density_ratio + torch.rand(B, 1, 3) * (1 - 1 / density_ratio)
    # # sqrdists = square_distance(data, data)
    # # target = (sqrdists < 1 / density_ratio).sum(dim=-1) > N_density
    # target = torch.zeros(data.shape[:-1])
    # target[:, :N_density] = 1  # 这个单任务 # FC: 76.0%   PointNet(max): 76.0%   PointNet(avg): 92.4%   Transformer: 95.6%  PointAttentionNet: 95.0%   PointWiseNet: 96.1%
    # # data[:, -N_density:, :] = data[:, -N_density:, :]/2
    # # target[:, -N_density:] = 1

    # # 随机数据生成指定密度规则label，如果不知道规则的话分界线非常模糊，很难train  # Acc   FC: 76.7%   PointNet(max): 78.4%   PointNet(avg): 85%   Transformer: 90%  PointWiseNet: 90%
    # # data = torch.rand(B,100,3)
    # # sqrdists = square_distance(data, data) # (1000,100,100)
    # # target = (sqrdists < 0.1).sum(dim=-1).float() > 10
    # # 二维平面数据便于观察   # Acc   FC: 61.0%   PointNet(max): 61.0%   PointNet(avg): 67.4%   Transformer: 85.8%  PointWiseNet: 99.0%
    # data = torch.cat((torch.rand(B, 100, 2), torch.zeros((B, 100, 1))), dim=2)
    # sqrdists = square_distance(data, data)  # (1000,100,100)
    # target = (sqrdists < 0.01).sum(dim=-1).float() > 3

    # # 随机数据生成指定正弦空间距离排序规则label  # Acc   FC: 95.9%   PointNet(max): 96.4%   PointNet(avg): 98.7%   Transformer: %  PointWiseNet: %
    # data = torch.rand(B,100,3)
    # dist = ((data[..., 1] - (torch.sin(2 * torch.pi * data[..., 0])+1)/2) ** 2 + (data[..., 2] - (torch.cos(2 * torch.pi * data[..., 0])+1)/2) ** 2)
    # # target = dist < 0.32
    # target = torch.zeros(data.shape[:-1])
    # target[torch.arange(B).view(-1,1), dist.topk(k=100//2,dim=1)[1]] = 1

    # 随机数据生成指定空间点最近规则label   Acc   FC: 76.9%   PointNet(max): 92.4%   PointNet(avg): 90.6%   Transformer: 93.8%  PointWiseNet: 93.2%  @N=10
    # Acc   FC: 80.6%   PointNet(max): 82.6%   PointNet(avg): 82.5%   Transformer: 87.8%    PointWiseNet: 84.2%   @N=100    虽然PointNet(max)擅长找最值，但未必特别擅长找argmax
    data = torch.rand(B, 10, 3)
    rand_kernel = torch.rand(1,10 // 2,3) # 不能太多了，不然需要较大的网络规模
    dist = square_distance(data,rand_kernel.repeat(B,1,1))
    target = torch.zeros(data.shape[:-1])
    target[torch.arange(B).view(-1, 1), dist.argmin(dim=1)] = 1


    # # 生成带关联点对的数据     # Acc   FC: 52.0%   PointNet(max): 52.0%   PointNet(avg): 52.1%   Transformer: 88.7%(很难训练，学习率要设低)  PointWiseNet: 99.3%
    # N = 50
    # data = torch.rand(B, N, 3)
    # N_ = N // 4
    # data[:,:N_,(2,0,1)] = data[:,N_:2*N_,(0,1,2)] * 0.97 + torch.rand(B, N_, 3) * 0.03
    # # data[:,:N_,(0,1,2)] = data[:,N_:2*N_,(0,1,2)] * 0.97 + torch.rand(B, N_, 3) * 0.03
    # target = torch.zeros(data.shape[:-1])
    # target[:,:2*N_] = 1

    # # 生成带关联三点对的数据     # Acc   FC: 58.4%   PointNet(avg): 58.4%   Transformer: 59.3%(会过拟合)  PointWiseNet: 59.6%  PointWiseNet3: 100%(顿悟)
    # N = 18
    # data = torch.rand(B, N, 3)
    # N_ = N // 6
    # data[:,:N_,:] = data[:,N_:2*N_,:] * 0.5 + data[:,2*N_:3*N_,:] * 0.5
    # data[:, :N_, :] = data[:,:N_,:] * 0.97 + torch.rand(B, N_, 3) * 0.03
    # target = torch.zeros(data.shape[:-1])
    # target[:,:3*N_] = 1 # target[:, :N_] = 1

    # b=0;plt.clf();plt.scatter(data[b,:,0],data[b,:,1],c=target[b,:]);plt.pause(2)
    # plt.clf();plt.hist((sqrdists < 1/density_ratio).sum(dim=-1)[:100].flatten());plt.pause(2)

    # from mpl_toolkits.mplot3d import Axes3D
    # b = 0; fig = plt.figure();ax = fig.add_subplot(111, projection='3d')
    # colors = ['red' if label == 1 else 'green' for label in target[b, :]]
    # ax.scatter(data[b, :, 0], data[b, :, 1], data[b, :, 2], c=colors)
    # ax.set_title(f'3D Scatter Plot for Batch {b}');ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z');plt.tight_layout();plt.pause(1)
    data = data.float()  # data = data.double()
    target = target.long()
    dataset = TensorDataset(data, target)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    return train_dataset, val_dataset, test_dataset


class FCNet(nn.Module):  # Acc: 73.6%
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x:torch.Tensor):
        return self.fc(x)


class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gf_max_num = 128 * 1
        self.gf_num = 32 * 1
        self.k = 0  # 0 -> 10 -> 11

        # batch*100*point_feature_num
        self.conv1 = nn.Sequential(
            nn.Conv1d(3+self.k, 16, (1,)),
            # nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, self.gf_max_num, (1,)),
            # nn.BatchNorm1d(self.gf_max_num),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.gf_max_num, self.gf_max_num, (1,)),
            # nn.BatchNorm1d(self.gf_max_num),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.gf_max_num, self.gf_num, (1,)),
            # nn.BatchNorm1d(self.gf_num),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(3 + self.k + self.gf_num, 128, (1,)),
            # nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 16, (1,)),
            # nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Conv1d(16, 2, (1,)),
            nn.LogSoftmax(dim=-2),
        )

    def forward(self, x:torch.Tensor):
        if self.k:
            B, N, C = x.size()
            sqrdists = square_distance(x, x) # (B,N,N)
            topk_sqrdists,_ = torch.topk(sqrdists,k=self.k,largest=False,dim=-1) # (B,N,3)
            x = torch.cat((x, topk_sqrdists),dim=-1)
        B, N, C = x.size()
        x = x.transpose(-1,-2) # (n,point_feature_num,100,)

        x1 = self.conv1(x)
        # out_pool = F.max_pool1d(x1, kernel_size=(N,))
        out_pool = F.avg_pool1d(x1, kernel_size=(N,))
        # out_pool = torch.cat((F.max_pool1d(x1[:,:self.gf_num//2,:], kernel_size=(N,)),F.avg_pool1d(x1[:,self.gf_num//2:,:], kernel_size=(N,))),dim=-2)
        gf = self.conv2(out_pool)
        out = torch.cat((x, gf.repeat(1,1,N)),dim=-2)
        out = self.conv3(out)
        out = out.transpose(-1,-2).contiguous()
        return out


class PointAttentionNet(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dim = 64 # 16
        num_heads = 16
        self.embed_fc = nn.Sequential(
            nn.Linear(3, embed_dim),
            # nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads,dropout=0., batch_first=True)
        self.out_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(16, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        B, N, C = x.size()
        x = self.embed_fc(x)
        attention_output = self.mha(x,x,x,need_weights=False)[0]
        out = self.out_fc(attention_output)
        return out


class PointTransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dim = 8 * 4
        self.nhead = 2 * 4

        self.in_fc = nn.Sequential(
            nn.Linear(3, embed_dim),
            # nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2), # 生成embedding用dropout效果好像很差
            # nn.Linear(embed_dim, embed_dim),
            # nn.BatchNorm1d(embed_dim),
            # nn.ReLU(inplace=True),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=self.nhead, dropout=0.0, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1 * 1)

        self.out_fc = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(16, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        B, N, C = x.size()
        x = self.in_fc(x.view(-1,C)).view(B, N, -1)
        attention_output = self.transformer_encoder(x)
        out = self.out_fc(attention_output)
        return out


class PointPairwiseRelation(nn.Module):
    """
        Input:
            xq: [B, N, C]
            xk: [B, Nk, C]
        Output:
            [B, N, Co]
    """
    def __init__(self,C=16,Co=16,rela=True):
        super().__init__()
        self.rela = rela
        self.Co = Co
        self.fc = nn.Sequential(
            nn.Linear(C * 2, Co),
            nn.ReLU(inplace=True),
            nn.Linear(Co, Co),
            nn.ReLU(inplace=True),
        )

    def forward(self, xq, xk):
        Co = self.Co
        B, N, C = xq.size()
        _, Nk, _ = xk.size()
        xq = xq.view(B, N, 1, C)
        xk = xk.view(B, 1, Nk, C)
        xq = F.pad(xq, (0, C), mode='constant', value=0)
        xk = F.pad(xk, (C, 0), mode='constant', value=0)
        x_pw = xq + xk  # (B,N,Nk,C*2)
        if self.rela:
            x_pw[..., C:] = x_pw[..., C:] - x_pw[..., :C] # xk相对xq的特征
            # x_pw[..., 0] = torch.norm(x_pw[..., C:],dim=-1) ** 2
        x_pw = self.fc(x_pw)  # (B,N,Nk,Co)
        pw_max = torch.max(x_pw[...,:Co//2], dim=2).values # (B,N,Co//2)
        pw_avg = torch.mean(x_pw[...,Co//2:], dim=2) # (B,N,Co//2)
        out = torch.cat((pw_max, pw_avg), dim=-1)  # (B,N,Co)
        return out


class PointPairwiseAttention(nn.Module):
    """
        Input:
            xq: [B, N, C]
            xk: [B, Nk, C]
        Output:
            [B, N, Co]
    """
    def __init__(self,C=16,Co=16,nhead=1):
        super().__init__()
        Cpw = C * 2
        assert Co % nhead == 0
        Ch = Co // nhead
        self.h = nhead
        self.Ch = Ch
        self.fc_pw = nn.Sequential(
            nn.Identity(),
            # nn.Linear(Cpw, Cpw),
            # nn.ReLU(inplace=True),
        )  # 提供更多的非线性，非线性也可以添加到下面的v/w中，但加在前面可能性价比高
        self.fc_w = nn.Sequential(
            nn.Linear(Cpw, Cpw),
            nn.ReLU(inplace=True),
            nn.Linear(Cpw, nhead),
            nn.ReLU(inplace=True),
        )
        self.fc_v = nn.Sequential(
            # nn.Linear(Cpw, Cpw),  # 提供更多的非线性,但不如nn.Linear(Co, Co)性能好
            # nn.ReLU(inplace=True),
            nn.Linear(Cpw, Co),
            nn.ReLU(inplace=True),
            nn.Linear(Co, Co),
            nn.ReLU(inplace=True),
        )

    def forward(self, xq, xk):
        h = self.h
        Ch = self.Ch
        B, N, C = xq.size()
        _, Nk, _ = xk.size()
        xq = xq.view(B, N, 1, C)
        xk = xk.view(B, 1, Nk, C)
        xq = F.pad(xq, (0, C), mode='constant', value=0)
        xk = F.pad(xk, (C, 0), mode='constant', value=0)
        x_pw = xq + xk  # (B,N,Nk,C*2)
        x_pw[..., C:] = x_pw[..., C:] - x_pw[..., :C] # xk相对xq的特征
        # x_pw[..., 0] = torch.norm(x_pw[..., C:],dim=-1) ** 2

        x_pw = self.fc_pw(x_pw)

        weight = self.fc_w(x_pw).view(B,N,Nk,h,1)  # (B,N,Nk,h) -> (B,N,Nk,h,1)
        # weight = torch.log10(weight+1)  # need relu and nhead=1
        weight = torch.softmax(weight, dim=-3)
        value = self.fc_v(x_pw).view(B,N,Nk,h,Ch)  # (B,N,Nk,Co) -> (B,N,Nk,h,Ch)
        out = (weight * value).sum(dim=-3)  # (B,N,h,Ch)
        out = out.view(B,N,-1)
        return out


class PointPairwiseRelation3(nn.Module):
    """
        Input:
            x: [B, N, C]
            x1: [B, N1, C]
            x2: [B, N2, C]
        Output:
            [B, N, Co]
    """
    def __init__(self,C=16,Co=16):
        super().__init__()
        self.Co = Co
        self.fc = nn.Sequential(
            nn.Linear(C * 3, Co),
            nn.ReLU(inplace=True),
            nn.Linear(Co, Co),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, x1, x2):
        Co = self.Co
        B, N, C = x.size()
        _, N1, _ = x1.size()
        _, N2, _ = x2.size()
        x = x.view(B, N, 1, 1, C)
        x1 = x1.view(B, 1, N1, 1, C)
        x2 = x2.view(B, 1, 1, N2, C)
        x = F.pad(x, (0, 2*C), mode='constant', value=0)
        x1 = F.pad(x1, (C, C), mode='constant', value=0)
        x2 = F.pad(x2, (2*C, 0), mode='constant', value=0)
        x_pw = x + x1 + x2  # (B,N,N1,N2,C*3)
        x_pw[..., C:2*C] = x_pw[..., C:2*C] - x_pw[..., :C] # x1相对x的特征
        x_pw[..., 2*C:] = x_pw[..., 2*C:] - x_pw[..., :C]  # x2相对x的特征
        # # x_pw[..., :C] = x_pw[..., C:2 * C] - x_pw[..., 2 * C:]  # x2相对x1的特征
        # x_pw[..., :C] = x_pw[..., C:2*C] / (x_pw[..., 2*C:] + 1e-6)  # x2相对x1的特征
        # x_pw[..., 0] = torch.var(x_pw[..., :C], dim=-1).clamp(0,1)
        # x_pw[..., 1:C] = 0

        x_pw = self.fc(x_pw)  # (B,N,N1,N2,Co)
        pw_max = torch.amax(x_pw[..., :Co//2], dim=(2,3))  # (B,N,Co//2)
        pw_avg = torch.mean(x_pw[..., Co//2:], dim=(2,3))  # (B,N,Co//2)
        out = torch.cat((pw_max, pw_avg), dim=-1)  # (B,N,Co)
        return out
    

class PointPairwiseRelationNet(nn.Module):
    def __init__(self):
        super().__init__()
        Cgf = 16*4 # 4 # 64
        self.pw_net = PointPairwiseRelation(C=3, Co=Cgf)
        # self.pw_net = PointPairwiseAttention(C=3, Co=Cgf, nhead=4)
        # self.pw_net = PointPairwiseRelation3(C=3, Co=Cgf)
        self.out_fc = nn.Sequential(
            nn.Linear(Cgf, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(16, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        B, N, C = x.size()
        out = self.pw_net(x, x)
        # out = self.pw_net(x, x, x)
        out = self.out_fc(out)
        return out


class LitNN(pl.LightningModule):
    def __init__(self,example_input_array=torch.Tensor(1, 100, 6)):
        super().__init__()
        # self.model = FCNet()
        self.model = PointNet()
        # self.model = PointAttentionNet()
        # self.model = PointTransformerNet()
        # self.model = PointPairwiseRelationNet()
        # self.model = torch.compile(self.model)

        self.criterion = nn.NLLLoss()
        # self.criterion = FocalLoss(gamma=2)

        self.metric_mean_loss_train = torchmetrics.MeanMetric()
        self.metric_mean_loss_val = torchmetrics.MeanMetric()
        self.metric_accuracy_train = torchmetrics.Accuracy(task='multiclass', num_classes=2, average='micro')  # average  'macro'->mAcc 'micro'->OA
        self.metric_accuracy_val = torchmetrics.Accuracy(task='multiclass', num_classes=2, average='micro')
        self.metric_f1_train = torchmetrics.FBetaScore(task='multiclass', num_classes=2, beta=1.0, average=None)
        self.metric_f1_val = torchmetrics.FBetaScore(task='multiclass', num_classes=2, beta=1.0, average=None)
        self.metric_pr = torchmetrics.PrecisionRecallCurve(task='multiclass', num_classes=2)
        self.metric_roc = torchmetrics.ROC(task='multiclass', num_classes=2)
        self.metric_auroc = torchmetrics.AUROC(task='multiclass', num_classes=2, average=None)
        self.metric_ap = torchmetrics.AveragePrecision(task='multiclass', num_classes=2, average=None) # average  'macro'->mAP None->AP
        self.metric_confusion_matrix = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=2)

        self.example_input_array = example_input_array # (BATCH_SIZE, 33, 2)
        # self.save_hyperparameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.model(x, **kwargs)
        return out

    def training_step(self, batch, batch_idx=0): # 基于batch_idx可实现一个epoch内依次训练网络的不同部分
        data, target = batch
        preds = self.forward(data)
        preds = preds.reshape(-1,preds.shape[-1])
        target = target.reshape(-1)
        loss = self.criterion(preds, target)

        preds = torch.exp(preds)
        self.metric_mean_loss_train(loss, len(target))
        self.metric_accuracy_train(preds, target)
        self.metric_f1_train(preds, target)
        self.log('loss_step', loss,on_step=True) # self.trainer.logged_metrics['loss_step']
        return loss

    def on_train_epoch_end(self) -> None:
        self.logger.experiment.add_scalars('_/loss', {'train': self.metric_mean_loss_train.compute().item()}, global_step=self.global_step)
        self.metric_mean_loss_train.reset()
        self.logger.experiment.add_scalars('_/F1_score', {'train': self.metric_f1_train.compute()[1].item()}, global_step=self.global_step)
        self.metric_f1_train.reset()
        self.logger.experiment.add_scalars('_/Accuracy', {'train': self.metric_accuracy_train.compute().item()}, global_step=self.global_step)
        self.metric_accuracy_train.reset()

    def validation_step(self, batch, batch_idx=0):
        data, target = batch
        preds = self.forward(data)
        preds = preds.reshape(-1,preds.shape[-1])
        target = target.reshape(-1)
        loss = self.criterion(preds, target)

        preds = torch.exp(preds)
        self.metric_mean_loss_val(loss,len(target))
        self.metric_accuracy_val(preds,target)
        self.metric_f1_val(preds, target)
        self.metric_auroc(preds, target)
        self.metric_ap(preds, target)
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True) # training_step和validation_step是不一致的，self.global_step默认为training_step，所以on_step在validation中默认为False
        self.log('val_accuracy', self.metric_accuracy_val, on_epoch=True, on_step=False, prog_bar=True) # reduce_fx=torch.mean 会自动适应变化的batch_size
        self.log('_/_AUROC', self.metric_auroc.compute()[1].item(), on_epoch=True, on_step=False, prog_bar=False)
        self.metric_auroc.reset()
        self.log('_/_mAP', self.metric_ap.compute()[1].item(), on_epoch=True, on_step=False, prog_bar=True)
        self.metric_ap.reset()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        self.logger.experiment.add_scalars('_/loss', {'val': self.metric_mean_loss_val.compute().item()}, global_step=self.global_step)
        self.metric_mean_loss_val.reset()
        self.logger.experiment.add_scalars('_/F1_score', {'val': self.metric_f1_val.compute()[1].item()}, global_step=self.global_step)
        self.metric_f1_val.reset()
        self.logger.experiment.add_scalars('_/Accuracy', {'val': self.metric_accuracy_val.compute().item()}, global_step=self.global_step)
        self.metric_accuracy_val.reset()

    def configure_optimizers(self):
        learning_rate = 1e-2
        # optimizer = torch.optim.Adam(self.parameters(),lr=learning_rate)
        # return optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        warmup_epochs = 50
        def lambda_lr(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return 1.0
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'epoch', 'frequency': 1}}

    # def on_train_start(self) -> None:
    #     self.logger.experiment.add_graph(self,self.example_input_array.to(self.device))

    def test_step(self,batch,batch_idx=0):
        data, target = batch
        preds = self.forward(data)
        preds = preds.reshape(-1,preds.shape[-1])
        target = target.reshape(-1)

        preds = torch.exp(preds)
        self.metric_confusion_matrix(preds,target)
        self.metric_pr(preds, target)
        self.metric_roc(preds, target)
        self.metric_auroc(preds, target)
        self.metric_ap(preds, target)

    def on_test_epoch_end(self):
        cfmat = self.metric_confusion_matrix.compute().cpu()
        self.metric_confusion_matrix.reset()
        cfmat_err = cfmat - cfmat.diag().diagflat()
        mat_target = torch.sum(cfmat,dim=-1).reshape(-1,1)
        plt.figure(figsize=(1, 2))
        fig_ = sns.heatmap(mat_target, annot=True, fmt="d", cmap=plt.cm.Blues).get_figure()  # cmap='Spectral'
        plt.close(fig_)
        self.logger.experiment.add_figure("confusion_target_number", fig_)
        plt.figure(figsize=(7, 5))
        fig_ = sns.heatmap(cfmat, annot=True, fmt="d", cmap=plt.cm.Blues).get_figure() # cmap='Spectral' # 纵轴为target，横轴为preds
        plt.close(fig_)
        self.logger.experiment.add_figure("confusion_matrix", fig_)
        plt.figure(figsize=(7, 5))
        fig_ = sns.heatmap(cfmat_err, annot=True, fmt="d", cmap=plt.cm.Reds).get_figure()  # cmap='Spectral'
        plt.close(fig_)
        self.logger.experiment.add_figure("confusion_matrix_error", fig_)

        precision, recall, thresholds = self.metric_pr.compute()
        self.metric_pr.reset()
        fig_ = plt.figure(figsize=(7, 5))
        plt.plot(recall[1].cpu(), precision[1].cpu(), label='Precision-Recall') # 第1个类别
        plt.plot(recall[1][:-1].cpu(), thresholds[1].cpu(), linestyle='dashed', label='Thresholds')  # thresholds的长度比recall少1
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0]); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.grid(True); plt.legend()
        self.logger.experiment.add_figure('_precision_recall_curve', fig_, global_step=self.global_step)
        plt.close(fig_)

        fpr, tpr, thresholds = self.metric_roc.compute()
        self.metric_roc.reset()
        fig_ = plt.figure(figsize=(7, 5))
        plt.plot(fpr[1].cpu(), tpr[1].cpu(), label='ROC')  # 第1个类别
        plt.plot(fpr[1].cpu(), thresholds[1].cpu(), linestyle='dashed', label='Thresholds')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0]); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.grid(True); plt.legend()
        self.logger.experiment.add_figure('_ROC_curve', fig_, global_step=self.global_step)
        plt.close(fig_)

        metric_auroc_compute = self.metric_auroc.compute()
        self.metric_auroc.reset()
        self.logger.experiment.add_scalars('test_AUROC', {str(i):scalar.item() for i,scalar in enumerate(metric_auroc_compute)})
        metric_ap_compute = self.metric_ap.compute()
        self.metric_ap.reset()
        self.logger.experiment.add_scalars('test_mAP', {str(i):scalar.item() for i,scalar in enumerate(metric_ap_compute)})

    def inference(self,data):
        data = data[np.newaxis,...]
        data = torch.from_numpy(data).float()
        data = data.to(self.device)
        with torch.no_grad():
            log_softmax = self.model(data)
        softmax = torch.exp(log_softmax)
        softmax = softmax.squeeze()
        preds = softmax.max(-1, keepdim=True)[1].squeeze()
        return softmax.cpu(),preds.cpu()


class TQDMProgressBarWithoutVal(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.close()
        # bar = tqdm(disable=True)
        return bar


if __name__ == "__main__":
    pl.seed_everything(42) # 42
    # torch.backends.cudnn.deterministic = True # 禁用随机性优化，使得卷积等算子在每次运行时都保持确定性
    # torch.backends.cudnn.benchmark = False # 禁用自动寻找最适合当前硬件的卷积算法，保证结果的一致性
    # # torch.set_num_threads(1) # 线程数设置为1，以确保不会引入多线程随机性
    # # torch.use_deterministic_algorithms(True) # 训练时自动判断操作是否有确定性
    BATCH_SIZE = 512//8 # 1024

    dataset_train, dataset_val, dataset_test = load_dataset()
    train_loader = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=0) # drop_last=False
    val_loader = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=0) # shuffle=False
    test_loader = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=0)

    # model_lit = LitNN(example_input_array=dataset_train[0][0].unsqueeze(0))
    # # ckpt_path = "./checkpoints/last.ckpt"
    # # model_lit = LitNN.load_from_checkpoint(ckpt_path)
    # for batch in train_loader:
    #     inputs, labels = batch
    #     break
    # inputs_ = [input_[:1] for input_ in inputs] if isinstance(inputs, list) else inputs[:1]
    # from torchsummary import summary
    # summary(model_lit.model.cpu(), inputs=(inputs_,))
    # # 计算flops,但会往模型state_dict里添加total_params、total_ops等，导致load_from_checkpoint报错，因此建议放在训练保存后面
    # from thop import profile, clever_format
    # from pprint import pprint
    # macs, params, *ret_dict = profile(model_lit.model.cpu(), inputs=(inputs_,), custom_ops=None, ret_layer_info=True,
    #                                   report_missing=True)
    # macs_cf, params_cf = clever_format([macs, params], "%.3f")
    # pprint(ret_dict, indent=1, depth=None, sort_dicts=False)  # depth递归深度
    # print('MACs:', int(macs), '=', macs_cf, '     Params:', int(params), '=', params_cf)


    hist_dir = './lightning_history/' + time.strftime('%Y_%m_%d_%H_%M_%S')
    log_dir = hist_dir + "/lightning_logs"
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir, '--port', '6006'])  # tensorboard --logdir ./
    url = tb.launch()  # 'http://localhost:6006/'
    os.system('explorer ' + url)
    experiments = 1
    for v in range(experiments):
        model_lit = LitNN(example_input_array=dataset_train[0][0].unsqueeze(0))
        model_checkpoint = ModelCheckpoint(monitor="_/_mAP",  # val_loss
                                           mode='max',  # 'min'
                                           save_last=True,
                                           save_top_k=1,
                                           dirpath=hist_dir+"/ckpt", # +'/'+str(v)
                                           # filename='best-{epoch:02d}-{val_loss:.2f}', # default: epoch=0-step=0.ckpt
                                           )
        trainer = pl.Trainer(limit_train_batches=1.0, max_epochs=300//experiments,
                             accelerator='auto', precision='32-true', # 16-mixed
                             callbacks=[
                                 TQDMProgressBarWithoutVal(), # TQDMProgressBar(),
                                 # model_checkpoint,
                             ],
                             logger=pl.loggers.TensorBoardLogger(save_dir=hist_dir),  # trainer.log_dir = hist_dir + "/lightning_logs"
                             log_every_n_steps=1, # (len(train_loader.dataset)//train_loader.batch_size)
                             # profiler='pytorch', # 'simple' 'advanced' 'pytorch' 'xla'   Find training loop bottlenecks
                             )
        # trainer.fit(model_lit, train_loader, val_loader, ckpt_path="./checkpoints/last.ckpt")
        trainer.fit(model_lit, train_loader, val_loader)

        # best_ckpt_path = model_checkpoint.best_model_path
        # if not best_ckpt_path == '':
        #     print('Best ckpt_path:',best_ckpt_path)
        #     shutil.copyfile(best_ckpt_path,os.path.dirname(best_ckpt_path)+'/best.ckpt')
        # else:
        #     best_ckpt_path = './checkpoints/best.ckpt'
        # trainer.test(model_lit,dataloaders=test_loader,ckpt_path=best_ckpt_path)

    # # trainer = pl.Trainer(auto_scale_batch_size=True, ) # the self.batch_size of train_loader should be scaleable
    # trainer = pl.Trainer(auto_lr_find=True)
    # trainer.tune(model_lit,train_loader,val_loader)

    print('Finished.')
    while True:
        time.sleep(2)




