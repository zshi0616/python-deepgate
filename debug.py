import deepgate 
import torch 
import torch.nn.functional as F

import sys
sys.setrecursionlimit(1000000)

if __name__ == '__main__': 
    print('[INFO] Create and load pretrained DeepGate') 
    model = deepgate.Model()    # Create DeepGate 
    model.load_pretrained()      # Load pretrained model 
    parser = deepgate.AigParser()   # Create AigParser 
    
    tmp_path = './tmp/miter.aig'
    g = parser.read_aiger(tmp_path)
    hs, hf = model(g)
    
    aig_1_path = './tmp/h29.aiger'
    g1 = parser.read_aiger(aig_1_path)
    hs1, hf1 = model(g1)
    po1 = hf1[g1.POs]
    
    aig_2_path = './tmp/h29_02.aiger'
    g2 = parser.read_aiger(aig_2_path)
    hs2, hf2 = model(g2)
    po2 = hf2[g2.POs]
    
    sim = F.cosine_similarity(po1, po2, dim=1)
    
    print(sim)