import os 
import torch
from torch_geometric.loader import DataLoader, DataListLoader

import deepgate
from config import get_parse_args

if __name__ == '__main__':
    # # Train 
    # data_dir = './data/train'
    # circuit_path = './data/train/graphs.npz'
    # label_path = './data/train/labels.npz'
    # num_epochs = 60
    
    # dataset = deepgate.NpzParser(data_dir, circuit_path, label_path)
    # train_dataset, val_dataset = dataset.get_dataset()
    # model = deepgate.Model()
    # trainer = deepgate.Trainer(model)
    # trainer.train(num_epochs, train_dataset, val_dataset)
    # print(model)
    
    
    # Test one aig
    aig_filepath = './examples/test.aiger'
    model = deepgate.Model()
    parser = deepgate.AigParser()
    graph = parser.read_aiger(aig_filepath)
    hs, hf = model(graph)
    print()
    
    
    
    