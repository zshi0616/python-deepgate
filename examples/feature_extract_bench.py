from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepgate
import torch
import time 

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[INFO] Create and load pretrained DeepGate')
    print('[INFO] Device: ', device)
    model = deepgate.Model()    # Create DeepGate
    model.load_pretrained()      # Load pretrained model
    model = model.to(device)
    
    bench_path = './tmp/test.bench'
    print('[INFO] Parse Bench: ', bench_path)
    parser = deepgate.BenchParser()   # Create BenchParser

    graph = parser.read_bench(bench_path) # Parse Bench into Graph
    graph = graph.to(device)
    print('[INFO] Get embeddings ...')
    start_time = time.time()
    hs, hf = model(graph)       # Model inference 
    end_time = time.time()
    
    # hs: structural embeddings, hf: functional embeddings
    # hs/hf: [N, D]. N: number of gates, D: embedding dimension (default: 128)
    print(hs.shape, hf.shape)   
    print('Time: ', end_time - start_time)