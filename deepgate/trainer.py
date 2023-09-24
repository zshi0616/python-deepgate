from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from torch import nn
import time
from progress.bar import Bar

from .arch.mlp import MLP
from .utils.utils import zero_normalization, AverageMeter

class Trainer():
    def __init__(self,
                 model, 
                 training_id = 'default',
                 save_dir = './exp', 
                 lr = 1e-4,
                 prob_rc_func_weight = [3.0, 1.0, 2.0],
                 emb_dim = 128, 
                 device = 'cpu'
                 ):
        super(Trainer, self).__init__()
        # Config
        self.emb_dim = emb_dim
        self.device = device
        self.lr = lr
        self.lr_step = -1
        self.prob_rc_func_weight = prob_rc_func_weight
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.log_dir = os.path.join(save_dir, training_id)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_path = os.path.join(self.log_dir, 'log.txt')
        
        # Loss and Optimizer
        self.reg_loss = nn.L1Loss().to(self.device)
        self.clf_loss = nn.BCELoss().to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Model
        self.model = model.to(self.device)
        self.readout_rc = MLP(emb_dim * 2, 32, num_layer=3, p_drop=0.2, norm_layer='batchnorm', sigmoid=True).to(self.device)
        self.model_epoch = 0
        
        # Print
        print('[INFO] Device: {}'.format(self.device))
        
    def set_training_args(self, prob_rc_func_weight=[], lr=-1, lr_step=-1, device='null'):
        if len(prob_rc_func_weight) == 3 and prob_rc_func_weight != self.prob_rc_func_weight:
            print('[INFO] Update prob_rc_func_weight from {} to {}'.format(self.prob_rc_func_weight, prob_rc_func_weight))
            self.prob_rc_func_weight = prob_rc_func_weight
        if lr > 0 and lr != self.lr:
            print('[INFO] Update learning rate from {} to {}'.format(self.lr, lr))
            self.lr = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        if lr_step > 0 and lr_step != self.lr_step:
            print('[INFO] Update learning rate step from {} to {}'.format(self.lr_step, lr_step))
            self.lr_step = lr_step
        if device != 'null' and device != self.device:
            print('[INFO] Update device from {} to {}'.format(self.device, device))
            self.device = device
            self.model = self.model.to(self.device)
            self.reg_loss = self.reg_loss.to(self.device)
            self.clf_loss = self.clf_loss.to(self.device)
            self.optimizer = self.optimizer
            self.readout_rc = self.readout_rc.to(self.device)
        
    def save(self, path):
        data = {
            'epoch': self.model_epoch, 
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(data, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model_epoch = checkpoint['epoch']
        self.model.load(path)
        return path
        
    def run_batch(self, batch):
        hs, hf = self.model(batch)
        prob = self.model.pred_prob(hf)
        # Task 1: Probability Prediction 
        prob_loss = self.reg_loss(prob, batch['prob'])
        # Task 2: Structural Prediction
        rc_emb = torch.cat([hs[batch['rc_pair_index'][0]], hs[batch['rc_pair_index'][1]]], dim=1)
        is_rc = self.readout_rc(rc_emb)
        rc_loss = self.clf_loss(is_rc, batch['is_rc'])
        # Task 3: Functional Similarity 
        node_a = hf[batch['tt_pair_index'][0]]
        node_b = hf[batch['tt_pair_index'][1]]
        emb_dis = 1 - torch.cosine_similarity(node_a, node_b, eps=1e-8)
        emb_dis_z = zero_normalization(emb_dis)
        tt_dis_z = zero_normalization(batch['tt_dis'])
        func_loss = self.reg_loss(emb_dis_z, tt_dis_z)
        
        return prob_loss, rc_loss, func_loss
    
    def train(self, num_epoch, train_dataset, val_dataset):
        batch_time = AverageMeter()
        prob_loss_stats, rc_loss_stats, func_loss_stats = AverageMeter(), AverageMeter(), AverageMeter()
        self.log_file = open(self.log_path, 'w')
        for epoch in range(num_epoch): 
            for phase in ['train', 'val']:
                if phase == 'train':
                    dataset = train_dataset
                    self.model.train()
                    self.model.to(self.device)
                else:
                    dataset = val_dataset
                    self.model.eval()
                    self.model.to(self.device)
                    torch.cuda.empty_cache()
                bar = Bar('{} {:}/{:}'.format(phase, epoch, num_epoch), max=len(dataset))
                for iter_id, batch in enumerate(dataset):
                    batch = batch.to(self.device)
                    time_stamp = time.time()
                    # Get loss
                    prob_loss, rc_loss, func_loss = self.run_batch(batch)
                    loss = prob_loss * self.prob_rc_func_weight[0] + rc_loss * self.prob_rc_func_weight[1] + \
                            func_loss * self.prob_rc_func_weight[2]
                    loss /= sum(self.prob_rc_func_weight)
                    loss = loss.mean()
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    # Print and save log
                    batch_time.update(time.time() - time_stamp)
                    prob_loss_stats.update(prob_loss.item())
                    rc_loss_stats.update(rc_loss.item())
                    func_loss_stats.update(func_loss.item())
                    Bar.suffix = '[{:}/{:}]|Tot: {total:} |ETA: {eta:} '.format(iter_id, len(dataset), total=bar.elapsed_td, eta=bar.eta_td)
                    Bar.suffix += '|Prob: {:.4f} |RC: {:.4f} |Func: {:.4f} '.format(prob_loss_stats.avg, rc_loss_stats.avg, func_loss_stats.avg)
                    Bar.suffix += '|Net: {:.2f}s '.format(batch_time.avg)
                    bar.next()
                if phase == 'train':
                    self.save(os.path.join(self.log_dir, 'model_last.pth'))
                self.log_file.write('{}| Epoch: {:}/{:} |Prob: {:.4f} |RC: {:.4f} |Func: {:.4f} |Net: {:.2f}s\n'.format(
                    phase, epoch, num_epoch, prob_loss_stats.avg, rc_loss_stats.avg, func_loss_stats.avg, batch_time.avg))
                bar.finish()
            
            # Learning rate decay
            self.model_epoch += 1
            if self.lr_step > 0 and self.model_epoch % self.lr_step == 0:
                self.lr *= 0.1
                print('[INFO] Learning rate decay to {:.4f}'.format(self.lr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
            
        self.log_file.close()
                