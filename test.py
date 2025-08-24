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
from loader.Dataloader_test import get_loaders
from utils.util import check_dirs,add_weight_decay,ModelEma
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import pathlib
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler
from network.model_train import SSL_MISS_Net
from torch.utils import data
torch.autograd.set_detect_anomaly(True)
from sklearn.model_selection import StratifiedKFold
noise = torch.normal(0, 1, size=(3, 16, 16))

from torchvision.models import resnet34,ResNet34_Weights
from torchmetrics import Accuracy,AUROC,Specificity,Recall

class Solver:
    def __init__(self, data_files, opt):
        self.opt = opt
        self.test_data_name = self.opt.test_data_name
        self.phase = self.opt.phase
        self.word_2_vec_path = self.opt.word_2_vec_path
        self.adj_matrix_path = self.opt.adj_matrix_path
        loaders = get_loaders(data_files, w2v_path=self.word_2_vec_path,test_dataset_name=self.test_data_name)
        self.loaders = loaders
        self.max_epoch = self.opt.max_epoch
        self.lr = self.opt.lr
        self.min_lr = self.opt.min_lr
        self.beta1 = self.opt.beta1
        self.beta2 = self.opt.beta2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = self.opt.checkpoint_dir
        self.save_name = self.opt.save_name

    def test(self):
        val_loader = self.loaders
        self.test = SSL_MISS_Net(num_classes=7, adj_path=self.adj_matrix_path, pretrain=False)

        self.test.load_state_dict(torch.load(self.checkpoint_dir)['state_dict'])
        self.test.to(self.device)
        self.test.eval()
        val_loader = data.DataLoader(dataset=val_loader, batch_size=1, shuffle=False,
                                     num_workers=0)
        softmax = nn.Softmax(dim=1)

        #metirc
        idh_acc = Accuracy(task="binary")
        idh_auc = AUROC(task="multiclass", num_classes=2, average='weighted')
        idh_spe = Specificity(task="multiclass", num_classes=2, average='weighted')
        idh_sen = Recall(task="multiclass", num_classes=2, average='weighted')

        pq_acc = Accuracy(task="binary")
        pq_auc = AUROC(task="multiclass", num_classes=2, average='weighted')
        pq_spe = Specificity(task="multiclass", num_classes=2, average='weighted')
        pq_sen = Recall(task="multiclass", num_classes=2, average='weighted')

        tert_acc = Accuracy(task="multiclass", num_classes=3)
        tert_auc = AUROC(task="multiclass", num_classes=3, average='weighted')
        tert_spe = Specificity(task="multiclass", num_classes=3, average='weighted')
        tert_sen = Recall(task="multiclass", num_classes=3, average='weighted')

        with torch.no_grad():

            for i, batch_data in tqdm(enumerate(val_loader), total=len(val_loader)):
                t1c_input = batch_data[0].squeeze(0).to(self.device)
                flair_input = batch_data[1].squeeze(0).to(self.device)
                label = batch_data[2]
                inp = batch_data[3].to(self.device)
                m_d = batch_data[4].tolist()

                class_prob = self.test(t1c_input,flair_input,m_d,inp)

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


        print("[###Test###]###ACC: idh:{}, pq:{}, tert:{}###,".format(
            idh_acc_final, pq_acc_final, tert_acc_final))


        print("###AUC: idh:{}, pq:{}, tert:{}###,".format(idh_auc_final, pq_auc_final, tert_auc_final))

        print("###SPE: idh:{}, pq:{}, tert:{}###,".format(idh_spe_final, pq_spe_final, tert_spe_final))


        print("###SEN: idh:{}, pq:{}, tert:{}###,".format(idh_sen_final, pq_sen_final, tert_sen_final))


        return idh_acc_final, tert_acc_final, pq_acc_final, idh_auc_final,pq_auc_final,tert_auc_final,idh_spe_final, pq_spe_final, tert_spe_final,idh_sen_final, pq_sen_final, tert_sen_final

if __name__ == '__main__':
    cudnn.benchmark = True

    args = argparse.ArgumentParser()
    args.add_argument('--test_list', type=str,
                      default='./dataset/test.xlsx')
    args.add_argument('--test_data_name', type=str,
                      default='yiyuan')
    args.add_argument('--phase', type=str, default='test')
    args.add_argument('--lr', type=float, default=1e-5)
    args.add_argument('--min_lr', type=float, default=1e-9)
    args.add_argument('--beta1', type=float, default=0.9)
    args.add_argument('--beta2', type=float, default=0.999)
    args.add_argument('--seed', type=int, default=1234)

    args.add_argument('--device', type=bool, default=True)
    args.add_argument('--gpu_id', type=str, default='1')

    args.add_argument('--max_epoch', type=int, default=150)

    args.add_argument('--TF_methods', type=str, default='TF')
    args.add_argument('--checkpoint_dir', type=str, default='./checkpoint/Ours/model_best.pth.tar')
    args.add_argument('--save_name', type=str, default='Ours')
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
    data_files = dict(test=args.test_list)

    solver = Solver(data_files, args)
    if args.phase == 'test':
        solver.test()
    print('Done!')
