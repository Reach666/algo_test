import os
import time
import numpy as np
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader,random_split
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
    data = torch.rand(4096,100,3)
    sqrdists = square_distance(data, data) # (1000,100,100)
    target = (sqrdists < 0.1).sum(dim=-1).float() > 10
    data = data.float()
    # data = data.double()
    target = target.long()
    dataset = TensorDataset(data, target)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    return train_dataset, val_dataset, test_dataset


class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gf_num = 25 * 4
        self.k = 10

        # batch*100*point_feature_num
        self.conv1 = nn.Sequential(
            nn.Conv1d(3+self.k, 20, (1,)),
            # nn.BatchNorm1d(20),
            nn.ReLU(inplace=True),
            nn.Conv1d(20, self.gf_num, (1,)),
            # nn.BatchNorm1d(self.gf_num),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.gf_num, self.gf_num, (1,)),
            # nn.BatchNorm1d(self.gf_num),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(self.gf_num + self.gf_num * 1, 20, (1,)),
            # nn.BatchNorm1d(20),
            nn.ReLU(inplace=True),
            nn.Conv1d(20, 10, (1,)),
            # nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Conv1d(10, 2, (1,)),
            nn.LogSoftmax(dim=-2),
        )

    def forward(self, x:torch.Tensor):
        B, N, C = x.size()
        sqrdists = square_distance(x, x) # (B,N,N)
        topk_sqrdists,_ = torch.topk(sqrdists,k=self.k,largest=False,dim=-1) # (B,N,3)
        x = torch.cat((x, topk_sqrdists),dim=-1)
        B, N, C = x.size()
        x = x.transpose(-1,-2) # (n,point_feature_num,100,)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        out_pool = F.avg_pool1d(x2, kernel_size=(N,))
        # out_pool = F.max_pool1d(x2[:,:self.gf_num//2,:], kernel_size=(N,))
        # out_pool = torch.cat((out_pool,F.avg_pool1d(x2[:,self.gf_num//2:,:], kernel_size=(N,))),dim=-2)
        out = torch.cat((x1, out_pool.repeat(1,1,N)),dim=-2)
        out = self.conv3(out)
        out = out.transpose(-1,-2).contiguous()
        return out


class PointTransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 8
        self.nhead = 4

        self.in_fc = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2), # 生成embedding用dropout效果好像很差
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            # nn.ReLU(inplace=True),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.nhead, dropout=0.0, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.out_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 10),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(10, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        B, N, C = x.size()
        x = self.in_fc(x.view(-1,C)).view(B, N, -1)
        attention_output = self.transformer_encoder(x)
        out = self.out_fc(attention_output)
        return out


class LitNN(pl.LightningModule):
    def __init__(self,example_input_array=torch.Tensor(1, 100, 6)):
        super().__init__()
        self.model = PointNet()
        # self.model = PointTransformerNet()
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
        optimizer = torch.optim.Adam(self.parameters(),lr=learning_rate)
        return optimizer

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
    BATCH_SIZE = 512//2 # 1024
    learning_rate = 3e-3

    dataset_train, dataset_val, dataset_test = load_dataset()
    train_loader = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=0) # drop_last=False
    val_loader = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=0) # shuffle=False
    test_loader = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=0)

    model_lit = LitNN(example_input_array=dataset_train[0][0].unsqueeze(0))
    # ckpt_path = "./checkpoints/last.ckpt"
    # model_lit = LitNN.load_from_checkpoint(ckpt_path)

    model_checkpoint = ModelCheckpoint(monitor="_/_mAP", # val_loss
                                       mode='max', # 'min'
                                       save_last=True,
                                       save_top_k=1,
                                       dirpath="./checkpoints",
                                       # filename='best-{epoch:02d}-{val_loss:.2f}', # default: epoch=0-step=0.ckpt
                                       )
    trainer = pl.Trainer(limit_train_batches=1.0, max_epochs=400,
                         accelerator='auto', precision='32-true', # 16-mixed
                         callbacks=[
                             TQDMProgressBarWithoutVal(), # TQDMProgressBar(),
                             model_checkpoint,
                         ],
                         # logger=pl.loggers.TensorBoardLogger(save_dir="logs/"),
                         log_every_n_steps=1, # (len(train_loader.dataset)//train_loader.batch_size)
                         # enable_progress_bar=False,
                         # profiler='pytorch', # 'simple' 'advanced' 'pytorch' 'xla'   Find training loop bottlenecks
                         )
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', trainer.log_dir, '--port', '6006'])  # tensorboard --logdir ./
    url = tb.launch()  # 'http://localhost:6006/'
    os.system('explorer ' + url)

    # trainer.fit(model_lit, train_loader, val_loader, ckpt_path="./checkpoints/last.ckpt")
    trainer.fit(model_lit, train_loader, val_loader)

    best_ckpt_path = model_checkpoint.best_model_path
    if not best_ckpt_path == '':
        print('Best ckpt_path:',best_ckpt_path)
        shutil.copyfile(best_ckpt_path,os.path.dirname(best_ckpt_path)+'/best.ckpt')
    else:
        best_ckpt_path = './checkpoints/best.ckpt'
    trainer.test(model_lit,dataloaders=test_loader,ckpt_path=best_ckpt_path)
    print('Finished.')


    # # trainer = pl.Trainer(auto_scale_batch_size=True, ) # the self.batch_size of train_loader should be scaleable
    # trainer = pl.Trainer(auto_lr_find=True)
    # trainer.tune(model_lit,train_loader,val_loader)

    while True:
        time.sleep(2)




