from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepgate

if __name__ == '__main__':
    data_dir = './data/train'
    circuit_path = './data/train/graphs.npz'
    label_path = './data/train/labels.npz'
    num_epochs = 100
    
    print('[INFO] Parse Dataset')
    dataset = deepgate.NpzParser(data_dir, circuit_path, label_path)
    train_dataset, val_dataset = dataset.get_dataset()
    print('[INFO] Create Model and Trainer')
    model = deepgate.Model()
    trainer = deepgate.Trainer(model)
    print('[INFO] Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)
    