from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepgate
import torch
import numpy as np

from deepgate.utils.aiger_utils import xdata_to_cnf
from deepgate.utils.circuit_utils import get_fanin_fanout

if __name__ == '__main__':
    
    aig_path = './examples/mult_op_DEMO1_6_6_TOP3.blif.aiger'
    print('[INFO] Parse AIG: ', aig_path)
    parser = deepgate.AigParser()   # Create AigParser
    graph = parser.read_aiger(aig_path) # Parse AIG into Graph
    
    edge_index = np.transpose(graph.edge_index.numpy())
    x_data = []
    for idx in range(len(graph.x)):
        if graph.x[idx][0] == 1:
            gate_type = 0
        elif graph.x[idx][1] == 1:
            gate_type = 1
        else:
            gate_type = 2
        x_data.append([idx, gate_type])
    fanin_list, fanout_list = get_fanin_fanout(x_data, edge_index)
    cnf = xdata_to_cnf(x_data, fanin_list, const_1=[int(graph.POs[0])])
    
    print(len(cnf))
    
    
            