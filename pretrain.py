import torch
from torch import optim
import torch.nn.functional as F
from torch.backends import cudnn
import numpy as np
import os
import time
import argparse
import random
import torch.nn as nn
from loader.Dataloader_train import get_loaders
from utils.util import check_dirs,add_weight_decay,ModelEma
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import pathlib
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from network.model_train import SSL_MISS_Net
from torch.utils import data
torch.autograd.set_detect_anomaly(True)
from sklearn.model_selection import StratifiedKFold
noise = torch.normal(0, 1, size=(3, 16, 16))

from torchvision.models import resnet34,ResNet34_Weights
from torchmetrics import Accuracy,AUROC,Specificity,Recall

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class ML_loss(nn.Module):

    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0, reduction: str = 'sum') -> None:
        super(ML_loss, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):  # logits:(384,80), targets:(384,80)
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`

        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = logits - self.margin  # (384,80)  margin :1
        pred_pos = torch.sigmoid(logits_margin)
        pred_neg = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        # Hill loss calculation
        los_pos = targets * torch.log(pred_pos)
        los_neg = (1 - targets) * -(self.lamb - pred_neg) * pred_neg ** 2

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def save_checkpoint(state: dict, save_folder: pathlib.Path):
    """Save Training state."""
    save_path = str(save_folder)+'/'
    # save_path.mkdir(parents=True, exist_ok=True)
    best_filename = save_path+ 'model_best'+ '.pth.tar'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, best_filename)
class Solver:
    def __init__(self, data_files, opt):
        self.opt = opt
        self.train_sheet_name = self.opt.train_data_name
        self.phase = self.opt.phase
        self.batch_size = self.opt.batch_size
        self.num_workers = self.opt.num_workers
        self.word_2_vec_path = self.opt.word_2_vec_path
        self.adj_matrix_path = self.opt.adj_matrix_path
        loaders = get_loaders(self,data_files, w2v_path=self.word_2_vec_path)
        self.loaders = loaders
        self.max_epoch = self.opt.max_epoch
        self.lr = self.opt.lr
        self.min_lr = self.opt.min_lr
        self.beta1 = self.opt.beta1
        self.beta2 = self.opt.beta2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = self.opt.checkpoint_dir

    def train(self):
        check_dirs([os.path.join(self.checkpoint_dir)])
        file = open(os.path.join(self.checkpoint_dir, 'final_result.txt'), 'w')
        train_loader = data.DataLoader(dataset=self.loaders, batch_size=self.batch_size,shuffle=True,num_workers= self.num_workers)
        self.G_pretrain = SSL_MISS_Net(num_classes=7, adj_path=self.adj_matrix_path, pretrain=True)
        self.G_pretrain.to(self.device)
        start_epoch = 0

        ema = ModelEma(self.G_pretrain, 0.9997)
        lr = self.lr
        loss_1 = nn.MSELoss()

        parameters = add_weight_decay(self.G_pretrain, lr)
        optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
        steps_per_epoch = len(train_loader)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=self.max_epoch,
                                            pct_start=0.2)
        scaler = GradScaler()

        early_stopping = EarlyStopping(50, verbose=True)
        print('\nStart training...')
        best_SSIM = 10000
        for epoch in range(start_epoch, self.max_epoch):
            self.G_pretrain.train()
            train_loss = 0
            loop = tqdm(enumerate(train_loader), total =len(train_loader))
            for i, batch_data in loop:

                t1c_input = batch_data[0].to(self.device)
                flair_input = batch_data[1].to(self.device)
                inp = batch_data[3].to(self.device)
                m_d = random.choice([0,1])

                re_images  = self.G_pretrain(t1c_input, flair_input,m_d, inp)

                images_label = torch.cat([flair_input, t1c_input], dim=1)

                Loss = loss_1(re_images, images_label)

                train_loss += Loss.item()
                self.G_pretrain.zero_grad()
                scaler.scale(Loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                ema.update(self.G_pretrain)
                loop.set_description(f'Epoch [{epoch}/{self.max_epoch}]')
                loop.set_postfix(loss=Loss.item())
            print("Epoch [{}], Loss [{}]".format(epoch,train_loss/steps_per_epoch))
            file.write(str("Epoch [{}], Loss [{}]".format(epoch,train_loss/steps_per_epoch) + '\n'))

            epoch_loss = (train_loss / steps_per_epoch)
            if epoch_loss < best_SSIM:
                best_SSIM = epoch_loss
                print(f"Saving the best model with Loss {best_SSIM}")
                model_dict = self.G_pretrain.state_dict()
                save_checkpoint(
                    dict(
                        state_dict=model_dict,
                    ),
                    save_folder=args.checkpoint_dir)

            early_metric = best_SSIM

            early_stopping(early_metric)
            # 若满足 early stopping 要求
            if early_stopping.early_stop:
                print("Early stopping")
                file.write(str("Early stopping\n"))
                # 结束模型训练
                break
        file.close()


if __name__ == '__main__':
    cudnn.benchmark = True

    args = argparse.ArgumentParser()
    args.add_argument('--train_list', type=str,
                      default='./dataset/pre_data.xlsx')
    args.add_argument('--train_data_name', type=str,
                      default='Sheet1')
    args.add_argument('--phase', type=str, default='train')
    args.add_argument('--batch_size', type=int, default=2)
    args.add_argument('--num_workers', type=int, default=4)

    args.add_argument('--lr', type=float, default=1e-5)
    args.add_argument('--min_lr', type=float, default=1e-9)
    args.add_argument('--beta1', type=float, default=0.9)
    args.add_argument('--beta2', type=float, default=0.999)
    args.add_argument('--seed', type=int, default=1234)

    args.add_argument('--device', type=bool, default=True)
    args.add_argument('--gpu_id', type=str, default='1')

    args.add_argument('--max_epoch', type=int, default=150)

    args.add_argument('--TF_methods', type=str, default='TF')
    args.add_argument('--checkpoint_dir', type=str, default='./checkpoint/')
    args.add_argument('--save_name', type=str, default='Pretrain')
    args.add_argument('--lr_update_epoch', type=int, default=1)
    args.add_argument('--word_2_vec_path', type=str, default='idh_pq_OAGBM.pkl')
    args.add_argument('--adj_matrix_path', type=str, default='adj_idh_pq_OAGBM.pkl')

    args = args.parse_args()

    print('-----Config-----')
    for k, v in sorted(vars(args).items()):
        print('%s:\t%s' % (str(k), str(v)))
    print('-------End------\n')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    data_files = dict(train=args.train_list)
    args.checkpoint_dir = args.checkpoint_dir + args.save_name
    solver = Solver(data_files, args)
    if args.phase == 'train':
        solver.train()
    print('Done!')
