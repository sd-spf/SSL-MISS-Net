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

def save_checkpoint(state: dict, save_folder: pathlib.Path, index,trials):
    """Save Training state."""
    save_path = str(save_folder)+'/'+str(trials)+'/'+str(index)+'/'
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
        self.pretrain_model_path = self.opt.pretrain_model_path

    def train(self):
        import copy
        check_dirs([os.path.join(self.checkpoint_dir)])
        file_1 = open(os.path.join(self.checkpoint_dir, 'final_result.txt'), 'w')
        skf_k = 0
        data_loader = self.loaders.data
        yy = [i[6] for i in data_loader]
        skf = StratifiedKFold(n_splits=5,shuffle=True)
        for train_idx, valid_idx in skf.split(data_loader, yy):
            skf_k += 1
            print('-----------------第{}折-------------------'.format(skf_k))
            check_dirs([os.path.join(self.checkpoint_dir, str(skf_k))])
            file = open(os.path.join(self.checkpoint_dir, str(skf_k), 'result.txt'), 'w')

            train_loader = copy.deepcopy(self.loaders)
            val_loader = copy.deepcopy(self.loaders)
            train_loader.data = [train_loader.data[ti] for ti in train_idx]
            val_loader.data = [val_loader.data[vi] for vi in valid_idx]

            train_loader = data.DataLoader(dataset=train_loader, batch_size=self.batch_size,shuffle=True,num_workers= self.num_workers)
            val_loader = data.DataLoader(dataset=val_loader, batch_size=1, shuffle=True,
                                           num_workers=self.num_workers)

            self.G_train = SSL_MISS_Net(num_classes=7, adj_path=self.adj_matrix_path, pretrain=False)
            self.G_train.load_state_dict(torch.load(self.pretrain_model_path)['state_dict'], strict=False)
            self.G_train.resnet_t1c = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.G_train.resnet_flair = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.G_train.to(self.device)
            for param in self.G_train.encoder.parameters():
                param.requires_grad = False

            start_epoch = 0
            ema = ModelEma(self.G_train, 0.9997)
            lr = self.lr
            loss_1 = ML_loss()

            parameters = add_weight_decay(self.G_train, lr)
            optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
            steps_per_epoch = len(train_loader)
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=self.max_epoch,
                                                pct_start=0.2)
            scaler = GradScaler()
            best_acc_val, best_AUC_val, best_acc_test, best_AUC_test = 0, 0, 0, 0
            best_spe_val, best_sen_val,best_index_val = 0,0,0
            best_spe_test, best_sen_test, best_index_test = 0, 0, 0
            early_stopping = EarlyStopping(50, verbose=True)
            print('\nStart training...')
            train_loss = 0
            for epoch in range(start_epoch, self.max_epoch):
                self.G_train.train()

                loop = tqdm(enumerate(train_loader), total =len(train_loader))
                for i, batch_data in loop:

                    t1c_input = batch_data[0].to(self.device)
                    flair_input = batch_data[1].to(self.device)
                    label = batch_data[2].to(self.device)
                    inp = batch_data[3].to(self.device)
                    m_d = batch_data[4].tolist()

                    fc_student = self.G_train(t1c_input, flair_input,m_d, inp)
                    Loss = loss_1(fc_student, label)
                    train_loss += Loss.item()
                    self.G_train.zero_grad()
                    scaler.scale(Loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    ema.update(self.G_train)
                    loop.set_description(f'Epoch [{epoch}/{self.max_epoch}]')
                    loop.set_postfix(loss=Loss.item())
                print("Epoch [{}], Loss [{}]".format(epoch,train_loss/steps_per_epoch))
                file.write(str("Epoch [{}], Loss [{}]".format(epoch,train_loss/steps_per_epoch) + '\n'))
                train_loss = 0
                if (epoch + 1) % self.val_epoch == 0:
                    print('\n')
                    idh_acc_val, tert_acc_val, pq_acc_val, idh_AUC_val, tert_AUC_val, PQ_AUC_val,idh_spe_final, pq_spe_final, tert_spe_final,idh_sen_final, pq_sen_final, tert_sen_final = self.val(epoch,file,val_loader)
                    avg_acc_val = (idh_acc_val + tert_acc_val + pq_acc_val) / 3
                    avg_AUC_val = (idh_AUC_val + tert_AUC_val + PQ_AUC_val) / 3
                    avg_SPE_val = (idh_spe_final+pq_spe_final+tert_spe_final) /3
                    avg_SEN_val = (idh_sen_final+pq_sen_final+tert_sen_final) / 3
                    avg_index_val = (avg_acc_val+avg_AUC_val)/2

                    if best_index_val < avg_index_val:
                        best_index_val = avg_index_val
                        best_spe_val = avg_SPE_val
                        best_sen_val = avg_SEN_val
                        print(f"Saving the best model {avg_index_val} in val")
                        model_dict = self.G_train.state_dict()
                        save_checkpoint(
                            dict(
                                state_dict=model_dict,
                            ),
                            save_folder=args.checkpoint_dir, index='Best', trials=skf_k)

                    if best_acc_val < avg_acc_val:
                        best_acc_val = avg_acc_val

                        print(f"Saving the best model with ACC {avg_acc_val} in val")
                        model_dict = self.G_train.state_dict()
                        save_checkpoint(
                            dict(
                                state_dict=model_dict,
                            ),
                            save_folder=args.checkpoint_dir, index='ACC', trials=skf_k)

                    if best_AUC_val < avg_AUC_val:
                        best_AUC_val = avg_AUC_val
                        print(f"Saving the best model with AUC {avg_AUC_val} in val")
                        model_dict = self.G_train.state_dict()
                        save_checkpoint(
                            dict(
                                state_dict=model_dict,
                            ),
                            save_folder=args.checkpoint_dir, index="AUC", trials=skf_k)
                    print(
                        "[###验证->val_epoch{} ]### avg_acc {}, avg_auc {} ,avg_spe {}, avg_sen {} ### best_acc {}  best_auc {} best_spe {}  best_sen {}".format(
                            epoch, avg_acc_val, avg_AUC_val, avg_SPE_val, avg_SEN_val, best_acc_val, best_AUC_val,best_spe_val,best_sen_val))
                    file.write(
                        str("avg_acc {}, avg_auc {}, avg_spe {}, avg_sen {}, best_acc {}, best_auc {},best_spe {}  best_sen {}".format(avg_acc_val,
                                                                                                              avg_AUC_val,
                                                                                                              avg_SPE_val,
                                                                                                              avg_SEN_val,
                                                                                                              best_acc_val,
                                                                                                              best_AUC_val,best_spe_val,best_sen_val) + '\n'))

                early_metric = (avg_acc_val+avg_AUC_val)/2

                early_stopping(early_metric)
                # 若满足 early stopping 要求
                if early_stopping.early_stop:
                    print("Early stopping")
                    file.write(str("Early stopping\n"))
                    # 结束模型训练
                    break
            file.close()
            file_1.write(str("Fold:{},best_acc: {}, best_auc: {},best_spe: {}, best_sen: {} ".format(skf_k,best_acc_val,best_AUC_val,best_spe_val,best_sen_val) + '\n'))
            file_1.write(
                str("Fold:{},best_acc: {}, best_auc: {},best_spe: {}, best_sen: {} ".format(skf_k, best_acc_test,
                                                                                            best_AUC_test, best_spe_test,
                                                                                            best_sen_test) + '\n'))
        file_1.close()

    def val(self, epoch, file,val_loader):
        # save_dir = os.path.join(self.result_dir, str(epoch))
        print('Start validation at iter {}...'.format(epoch))
        softmax = nn.Softmax(dim=1)

        #metirc
        idh_acc = Accuracy(task="binary")
        idh_auc = AUROC(task="binary")
        idh_spe = Specificity(task="binary")
        idh_sen = Recall(task="binary")

        pq_acc = Accuracy(task="binary")
        pq_auc = AUROC(task="binary")
        pq_spe = Specificity(task="binary")
        pq_sen = Recall(task="binary")

        tert_acc = Accuracy(task="multiclass", num_classes=3)
        tert_auc = AUROC(task="multiclass", num_classes=3, average='weighted')
        tert_spe = Specificity(task="multiclass", num_classes=3, average='weighted')
        tert_sen = Recall(task="multiclass", num_classes=3, average='weighted')

        self.G_train.eval()
        with torch.no_grad():

            for i, batch_data in tqdm(enumerate(val_loader), total=len(val_loader)):
                t1c_input = batch_data[0].to(self.device)
                flair_input = batch_data[1].to(self.device)
                label = batch_data[2]
                inp = batch_data[3].to(self.device)
                m_d = batch_data[4].tolist()
                class_prob = self.G_train(t1c_input,flair_input,m_d,inp)
                class_prob = class_prob.cpu()

                real_idh = label[:, :2]
                real_pq = label[:, 2:4]
                real_tert = label[:, 4:7]

                if real_idh.sum() != 0:
                    real_idh = torch.max(real_idh, dim=1)[1]
                    # 真实的label
                    pre_idh = softmax(class_prob[:, :2])  # 预测值

                    pre_idh = torch.mean(pre_idh, dim=0, keepdim=True)
                    fusion_idh = torch.max(pre_idh, dim=1)[1]  # 预测的label

                    idh_acc.update(fusion_idh, real_idh)
                    idh_auc.update(pre_idh, real_idh)
                    idh_spe.update(pre_idh, real_idh)
                    idh_sen.update(pre_idh, real_idh)

                if real_pq.sum() != 0:
                    real_pq = torch.max(real_pq, dim=1)[1]

                    pre_pq = softmax(class_prob[:, 2:4])  # 预测值
                    pre_pq = torch.mean(pre_pq, dim=0, keepdim=True)
                    fusion_pq = torch.max(pre_pq, dim=1)[1]  # 预测的label

                    pq_acc.update(fusion_pq, real_pq)
                    pq_auc.update(pre_pq, real_pq)
                    pq_spe.update(pre_pq, real_pq)
                    pq_sen.update(pre_pq, real_pq)

                if real_tert.sum() != 0:
                    real_tert = torch.max(real_tert, dim=1)[1]
                    pre_tert = softmax(class_prob[:, 4:7])  # 预测值
                    pre_tert = torch.mean(pre_tert, dim=0, keepdim=True)
                    fusion_tert = torch.max(pre_tert, dim=1)[1]  # 真实的label

                    tert_acc.update(fusion_tert, real_tert)
                    tert_auc.update(pre_tert, real_tert)
                    tert_spe.update(pre_tert, real_tert)
                    tert_sen.update(pre_tert, real_tert)

        idh_acc_final = idh_acc.compute().item()
        idh_auc_final = idh_auc.compute().item()
        idh_spe_final = idh_spe.compute().item()
        idh_sen_final = idh_sen.compute().item()

        pq_acc_final = pq_acc.compute().item()
        pq_auc_final = pq_auc.compute().item()
        pq_spe_final = pq_spe.compute().item()
        pq_sen_final = pq_sen.compute().item()

        tert_acc_final = tert_acc.compute().item()
        tert_auc_final = tert_auc.compute().item()
        tert_spe_final = tert_spe.compute().item()
        tert_sen_final = tert_sen.compute().item()


        print("[###验证->val_epoch{} ]###ACC: idh:{}, pq:{}, tert:{}###,".format(
            epoch, idh_acc_final, pq_acc_final, tert_acc_final))
        file.write(
            str("[###验证->val_epoch{} ]###ACC: idh:{}, pq:{}, tert:{}###".format(epoch,idh_acc_final, pq_acc_final, tert_acc_final) + '\n'))

        print("###AUC: idh:{}, pq:{}, tert:{}###,".format(idh_auc_final, pq_auc_final, tert_auc_final))
        file.write(str("###AUC: idh:{}, pq:{}, tert:{}###".format(idh_auc_final, pq_auc_final,tert_auc_final) + '\n'))

        print("###SPE: idh:{}, pq:{}, tert:{}###,".format(idh_spe_final, pq_spe_final, tert_spe_final))
        file.write(str("###SPE: idh:{}, pq:{}, tert:{}###".format(idh_spe_final, pq_spe_final, tert_spe_final) + '\n'))

        print("###SEN: idh:{}, pq:{}, tert:{}###,".format(idh_sen_final, pq_sen_final, tert_sen_final))
        file.write(str("###SEN: idh:{}, pq:{}, tert:{}###".format(idh_sen_final, pq_sen_final, tert_sen_final) + '\n'))


        return idh_acc_final, tert_acc_final, pq_acc_final, idh_auc_final,pq_auc_final,tert_auc_final,idh_spe_final, pq_spe_final, tert_spe_final,idh_sen_final, pq_sen_final, tert_sen_final

if __name__ == '__main__':
    cudnn.benchmark = True

    args = argparse.ArgumentParser()
    args.add_argument('--train_list', type=str,
                      default='./dataset/train.xlsx')
    args.add_argument('--train_data_name', type=str,
                      default='Sheet1')
    args.add_argument('--phase', type=str, default='train')
    args.add_argument('--batch_size', type=int, default=32)
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
    args.add_argument('--save_name', type=str, default='Ours')
    args.add_argument('--lr_update_epoch', type=int, default=1)
    args.add_argument('--word_2_vec_path', type=str, default='idh_pq_OAGBM.pkl')
    args.add_argument('--adj_matrix_path', type=str, default='adj_idh_pq_OAGBM.pkl')
    args.add_argument('--pretrain_model_path', type=str,
                      default='./checkpoint/Pretrained/model_best.pth.tar')
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
