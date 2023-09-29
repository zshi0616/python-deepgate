from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepgate
import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data/train/'

if __name__ == '__main__':
    circuit_path = os.path.join(DATA_DIR, 'graphs.npz')
    label_path = os.path.join(DATA_DIR, 'labels.npz')
    num_epochs = 60
    # model_path = './exp/default/model_last.pth'
    
    print('[INFO] Parse Dataset')
    dataset = deepgate.NpzParser(DATA_DIR, circuit_path, label_path)
    train_dataset, val_dataset = dataset.get_dataset()
    print('[INFO] Create Model and Trainer')
    model = deepgate.Model()
    
    trainer = deepgate.Trainer(model, distributed=True)
    # trainer.load(model_path)
    trainer.set_training_args(prob_rc_func_weight=[2.0, 1.0, 0.0], lr=1e-4, lr_step=-1)
    print('[INFO] Stage 1 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)
    
    print('[INFO] Stage 2 Training ...')
    trainer.set_training_args(prob_rc_func_weight=[3.0, 1.0, 2.0], lr=1e-4, lr_step=-1)
    trainer.train(num_epochs, train_dataset, val_dataset)
    
