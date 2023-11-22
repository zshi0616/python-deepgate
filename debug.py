from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepgate
import torch

if __name__ == '__main__':
    print('[INFO] Create and load pretrained DeepGate')
    model = deepgate.Model()    # Create DeepGate
    model.load_pretrained()      # Load pretrained model
    
    aig_path = './examples/b05_comb.aig'
    print('[INFO] Parse AIG: ', aig_path)
    parser = deepgate.AigParser()   # Create AigParser
    graph = parser.read_aiger(aig_path) # Parse AIG into Graph
    print('[INFO] Get embeddings ...')
    hs, hf = model(graph)       # Model inference 
    
    # hs: structural embeddings, hf: functional embeddings
    # hs/hf: [N, D]. N: number of gates, D: embedding dimension (default: 128)
    print(hs.shape, hf.shape)   
    