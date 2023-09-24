from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Callable, List
import os.path as osp

import torch
import shutil
import os
import copy
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from .utils.data_utils import construct_node_feature, add_skip_connection, add_edge_attr, one_hot
from .utils.data_utils import read_npz_file
from .utils.dag_utils import return_order_info

class OrderedData(Data):
    def __init__(self, edge_index=None, x=None, y=None, \
                 tt_pair_index=None, tt_dis=None, min_tt_dis=None, \
                 rc_pair_index=None, is_rc=None, \
                 forward_level=None, forward_index=None, backward_level=None, backward_index=None):
        super().__init__()
        self.edge_index = edge_index
        self.tt_pair_index = tt_pair_index
        self.x = x
        self.y = y
        self.tt_dis = tt_dis
        self.min_tt_dis = min_tt_dis
        self.forward_level = forward_level
        self.forward_index = forward_index
        self.backward_level = backward_level
        self.backward_index = backward_index
        self.rc_pair_index = rc_pair_index
        self.is_rc = is_rc
    
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'forward_index' or key == 'backward_index':
            return 0
        elif key == "edge_index" or key == 'tt_pair_index' or key == 'rc_pair_index':
            return 1
        else:
            return 0

def parse_pyg_mlpgate(x, edge_index, tt_dis, min_tt_dis, tt_pair_index, y, rc_pair_index, is_rc, \
    use_edge_attr=False, reconv_skip_connection=False, no_node_cop=False, node_reconv=False, un_directed=False, num_gate_types=9, dim_edge_feature=32, logic_implication=False, mask=False):
    x_torch = construct_node_feature(x, no_node_cop, node_reconv, num_gate_types)

    tt_pair_index = torch.tensor(tt_pair_index, dtype=torch.long)
    tt_pair_index = tt_pair_index.t().contiguous()
    rc_pair_index = torch.tensor(rc_pair_index, dtype=torch.long)
    rc_pair_index = rc_pair_index.t().contiguous()
    tt_dis = torch.tensor(tt_dis)
    is_rc = torch.tensor(is_rc, dtype=torch.float32).unsqueeze(1)
    min_tt_dis = torch.tensor(min_tt_dis)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = add_edge_attr(len(edge_index), dim_edge_feature, 1)
    if reconv_skip_connection:
        edge_index, edge_attr = add_skip_connection(x, edge_index, edge_attr, dim_edge_feature)
    
    edge_index = edge_index.t().contiguous()
    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, x_torch.size(0))

    if use_edge_attr:
        raise 'Unsupport edge attr'
    else:
        graph = OrderedData(x=x_torch, edge_index=edge_index, 
                            rc_pair_index=rc_pair_index, is_rc=is_rc,
                            tt_pair_index=tt_pair_index, tt_dis=tt_dis, min_tt_dis=min_tt_dis, 
                            forward_level=forward_level, forward_index=forward_index, 
                            backward_level=backward_level, backward_index=backward_index)
        graph.use_edge_attr = False

    # add reconvegence info
    # graph.rec = torch.tensor(x[:, 3:4], dtype=torch.float)
    # graph.rec_src = torch.tensor(x[:, 4:5], dtype=torch.float)
    # add gt info
    # add indices for gate types
    graph.gate = torch.tensor(x[:, 1:2], dtype=torch.float)
    graph.prob = torch.tensor(y).reshape((len(x), 1))

    return graph


class Dataset(InMemoryDataset):
    def __init__(self, root, args, transform=None, pre_transform=None, pre_filter=None):
        self.name = 'MIG'
        self.args = args

        assert (transform == None) and (pre_transform == None) and (pre_filter == None), "Cannot accept the transform, pre_transfrom and pre_filter args now."

        # Reload
        inmemory_dir = os.path.join(args.data_dir, 'inmemory')
        if args.reload_dataset and os.path.exists(inmemory_dir):
            shutil.rmtree(inmemory_dir)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        if self.args.small_train:
            name = 'inmemory_small'
        else:
            name = 'inmemory'
        if self.args.no_rc:
            name += '_norc'
        return osp.join(self.root, name)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.args.circuit_file, self.args.label_file]

    @property
    def processed_file_names(self) -> str:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        tot_pairs = 0
        circuits = read_npz_file(self.args.circuit_file, self.args.data_dir)['circuits'].item()
        labels = read_npz_file(self.args.label_file, self.args.data_dir)['labels'].item()
        
        if self.args.small_train:
            subset = 100

        for cir_idx, cir_name in enumerate(circuits):
            print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100))
            x = circuits[cir_name]["x"]
            edge_index = circuits[cir_name]["edge_index"]

            tt_dis = labels[cir_name]['tt_dis']
            min_tt_dis = labels[cir_name]['min_tt_dis']
            tt_pair_index = labels[cir_name]['tt_pair_index']
            prob = labels[cir_name]['prob']

            if self.args.no_rc:
                rc_pair_index = [[0, 1]]
                is_rc = [0]
            else:
                rc_pair_index = labels[cir_name]['rc_pair_index']
                is_rc = labels[cir_name]['is_rc']

            if len(tt_pair_index) == 0 or len(rc_pair_index) == 0:
                print('No tt or rc pairs: ', cir_name)
                continue

            tot_pairs += len(tt_dis)

            # check the gate types
            # assert (x[:, 1].max() == (len(self.args.gate_to_index)) - 1), 'The gate types are not consistent.'
            graph = parse_pyg_mlpgate(
                x, edge_index, tt_dis, min_tt_dis, tt_pair_index, prob, rc_pair_index, is_rc, 
                self.args.use_edge_attr, self.args.reconv_skip_connection, self.args.no_node_cop,
                self.args.node_reconv, self.args.un_directed, self.args.num_gate_types,
                self.args.dim_edge_feature, self.args.logic_implication, self.args.mask
            )
            graph.name = cir_name
            data_list.append(graph)
            if self.args.small_train and cir_idx > subset:
                break

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
        print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'