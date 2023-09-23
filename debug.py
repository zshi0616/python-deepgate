import os 
import torch
from torch_geometric.loader import DataLoader, DataListLoader

import deepgate
from config import get_parse_args

if __name__ == '__main__':
    args = get_parse_args()
    dataset = deepgate.Dataset(root=args.data_dir, args=args)
    perm = torch.randperm(len(dataset))
    dataset = dataset[perm]
    data_len = len(dataset)
    if args.local_rank == 0:
        print("Size: ", len(dataset))
        print('Splitting the dataset into training and validation sets..')
    training_cutoff = int(data_len * args.trainval_split)
    if args.local_rank == 0:
        print('# training circuits: ', training_cutoff)
        print('# validation circuits: ', data_len - training_cutoff)
    train_dataset = []
    val_dataset = []
    train_dataset = dataset[:training_cutoff]
    val_dataset = dataset[training_cutoff:]
    train_dataset = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_dataset = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = deepgate.Model()
    trainer = deepgate.Trainer(model)
    trainer.train(args.num_epochs, train_dataset, val_dataset)
    print(model)
    
    