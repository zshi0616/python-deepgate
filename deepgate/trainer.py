from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from torch import nn
import time
from progress.bar import Bar
from torch_geometric.loader import DataLoader

from .arch.mlp import MLP
from .utils.utils import zero_normalization, AverageMeter, get_function_acc
from .utils.logger import Logger

class Trainer():
    def __init__(self,
                 model, 
                 training_id = 'default',
                 save_dir = './exp', 
                 lr = 1e-4,
                 prob_rc_func_weight = [3.0, 1.0, 2.0],
                 emb_dim = 128, 
                 device = 'cpu', 
                 batch_size=16, num_workers=4, 
                 distributed = False
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
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        
        # Distributed Training 
        self.local_rank = 0
        if self.distributed:
            if 'LOCAL_RANK' in os.environ:
                self.local_rank = int(os.environ['LOCAL_RANK'])
            self.device = 'cuda:%d' % self.local_rank
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            print('Training in distributed mode. Device {}, Process {:}, total {:}.'.format(
                self.device, self.rank, self.world_size
            ))
        else:
            print('Training in single device: ', self.device)
        
        # Loss and Optimizer
        self.reg_loss = nn.L1Loss().to(self.device)
        self.clf_loss = nn.BCELoss().to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Model
        self.model = model.to(self.device)
        self.readout_rc = MLP(emb_dim * 2, 32, num_layer=3, p_drop=0.1, norm_layer='batchnorm', tanh=True).to(self.device)
        self.model_epoch = 0
        
        # Logger
        if self.local_rank == 0:
            self.logger = Logger(self.log_path)
        
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
        print('[INFO] Continue training from epoch {:}'.format(self.model_epoch))
        return path
        
    def run_batch(self, batch):
        hs, hf = self.model(batch)
        prob = self.model.pred_prob(hf)
        # Task 1: Probability Prediction 
        prob_loss = self.reg_loss(prob, batch['prob'])
        # Task 2: Structural Prediction
        rc_emb = torch.cat([hs[batch['rc_pair_index'][0]], hs[batch['rc_pair_index'][1]]], dim=1)
        is_rc = self.readout_rc(rc_emb)
        is_rc = torch.clamp(is_rc, min=1e-6, max=1-1e-6)
        rc_loss = self.clf_loss(is_rc, batch['is_rc'])
        # Task 3: Functional Similarity 
        node_a = hf[batch['tt_pair_index'][0]]
        node_b = hf[batch['tt_pair_index'][1]]
        emb_dis = 1 - torch.cosine_similarity(node_a, node_b, eps=1e-8)
        emb_dis_z = zero_normalization(emb_dis)
        tt_dis_z = zero_normalization(batch['tt_dis'])
        func_loss = self.reg_loss(emb_dis_z, tt_dis_z)
        loss_status = {
            'prob_loss': prob_loss, 
            'rc_loss': rc_loss,
            'func_loss': func_loss
        }
        
        return hs, hf, loss_status
    
    def train(self, num_epoch, train_dataset, val_dataset):
        # Distribute Dataset
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,
                                    num_workers=self.num_workers, sampler=train_sampler)
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,
                                     num_workers=self.num_workers, sampler=val_sampler)
        else:
            train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
        
        # AverageMeter
        batch_time = AverageMeter()
        prob_loss_stats, rc_loss_stats, func_loss_stats = AverageMeter(), AverageMeter(), AverageMeter()
        acc_stats = AverageMeter()
        
        # Train
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
                if self.local_rank == 0:
                    bar = Bar('{} {:}/{:}'.format(phase, epoch, num_epoch), max=len(dataset))
                for iter_id, batch in enumerate(dataset):
                    batch = batch.to(self.device)
                    time_stamp = time.time()
                    # Get loss
                    hs, hf, loss_status = self.run_batch(batch)
                    loss = loss_status['prob_loss'] * self.prob_rc_func_weight[0] + \
                        loss_status['rc_loss'] * self.prob_rc_func_weight[1] + \
                        loss_status['func_loss'] * self.prob_rc_func_weight[2]
                    loss /= sum(self.prob_rc_func_weight)
                    loss = loss.mean()
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    # Print and save log
                    batch_time.update(time.time() - time_stamp)
                    prob_loss_stats.update(loss_status['prob_loss'].item())
                    rc_loss_stats.update(loss_status['rc_loss'].item())
                    func_loss_stats.update(loss_status['func_loss'].item())
                    acc = get_function_acc(batch, hf)
                    acc_stats.update(acc)
                    if self.local_rank == 0:
                        Bar.suffix = '[{:}/{:}]|Tot: {total:} |ETA: {eta:} '.format(iter_id, len(dataset), total=bar.elapsed_td, eta=bar.eta_td)
                        Bar.suffix += '|Prob: {:.4f} |RC: {:.4f} |Func: {:.4f} '.format(prob_loss_stats.avg, rc_loss_stats.avg, func_loss_stats.avg)
                        Bar.suffix += '|Acc: {:.2f}%% '.format(acc*100)
                        Bar.suffix += '|Net: {:.2f}s '.format(batch_time.avg)
                        bar.next()
                if phase == 'train':
                    self.save(os.path.join(self.log_dir, 'model_last.pth'))
                if self.local_rank == 0:
                    self.logger.write('{}| Epoch: {:}/{:} |Prob: {:.4f} |RC: {:.4f} |Func: {:.4f} |ACC: {:.4f} |Net: {:.2f}s\n'.format(
                        phase, epoch, num_epoch, prob_loss_stats.avg, rc_loss_stats.avg, func_loss_stats.avg, acc_stats.avg, batch_time.avg))
                    bar.finish()
            
            # Learning rate decay
            self.model_epoch += 1
            if self.lr_step > 0 and self.model_epoch % self.lr_step == 0:
                self.lr *= 0.1
                if self.local_rank == 0:
                    print('[INFO] Learning rate decay to {:.4f}'.format(self.lr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
            