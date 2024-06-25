from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
from torch import nn
from torch.nn import LSTM, GRU
from .utils.dag_utils import subgraph, custom_backward_subgraph
from .utils.utils import generate_hs_init

from .arch.mlp import MLP
from .arch.mlp_aggr import MlpAggr
from .arch.tfmlp import TFMlpAggr
from .arch.gcn_conv import AggConv

class Model(nn.Module):
    '''
    Recurrent Graph Neural Networks for Circuits.
    '''
    def __init__(self, 
                 num_rounds = 1, 
                 dim_hidden = 128, 
                 enable_encode = True,
                 enable_reverse = False
                ):
        super(Model, self).__init__()
        
        # configuration
        self.num_rounds = num_rounds
        self.enable_encode = enable_encode
        self.enable_reverse = enable_reverse        # TODO: enable reverse

        # dimensions
        self.dim_hidden = dim_hidden
        self.dim_mlp = 32

        # Network 
        self.aggr_and_strc = TFMlpAggr(self.dim_hidden*1, self.dim_hidden)
        self.aggr_not_strc = TFMlpAggr(self.dim_hidden*1, self.dim_hidden)
        self.aggr_and_func = TFMlpAggr(self.dim_hidden*2, self.dim_hidden)
        self.aggr_not_func = TFMlpAggr(self.dim_hidden*1, self.dim_hidden)
            
        self.update_and_strc = GRU(self.dim_hidden, self.dim_hidden)
        self.update_and_func = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_strc = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_func = GRU(self.dim_hidden, self.dim_hidden)

        # Readout 
        self.readout_prob = MLP(self.dim_hidden, self.dim_mlp, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')

        # # consider the embedding for the LSTM/GRU model initialized by non-zeros
        # self.one = torch.ones(1)
        # # self.hs_emd_int = nn.Linear(1, self.dim_hidden)
        # self.hf_emd_int = nn.Linear(1, self.dim_hidden)
        # self.one.requires_grad = False

    def forward(self, G):
        device = next(self.parameters()).device
        num_nodes = len(G.gate)
        num_layers_f = max(G.forward_level).item() + 1
        num_layers_b = max(G.backward_level).item() + 1
        
        # initialize the structure hidden state
        if self.enable_encode:
            hs = torch.zeros(num_nodes, self.dim_hidden)
            hs = generate_hs_init(G, hs, self.dim_hidden)
        else:
            hs = torch.zeros(num_nodes, self.dim_hidden)
        
        # initialize the function hidden state
        # hf = self.hf_emd_int(self.one).view(1, -1) # (1 x 1 x dim_hidden)
        # hf = hf.repeat(num_nodes, 1) # (1 x num_nodes x dim_hidden)
        hf = torch.zeros(num_nodes, self.dim_hidden)
        hs = hs.to(device)
        hf = hf.to(device)
        
        edge_index = G.edge_index

        node_state = torch.cat([hs, hf], dim=-1)
        and_mask = G.gate.squeeze(1) == 1
        not_mask = G.gate.squeeze(1) == 2

        for _ in range(self.num_rounds):
            for level in range(1, num_layers_f):
                # forward layer
                layer_mask = G.forward_level == level

                # AND Gate
                l_and_node = G.forward_index[layer_mask & and_mask]
                if l_and_node.size(0) > 0:
                    and_edge_index, and_edge_attr = subgraph(l_and_node, edge_index, dim=1)
                    
                    
                    # Update structure hidden state
                    msg = self.aggr_and_strc(hs, and_edge_index, and_edge_attr)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hs_and = torch.index_select(hs, dim=0, index=l_and_node)
                    _, hs_and = self.update_and_strc(and_msg.unsqueeze(0), hs_and.unsqueeze(0))
                    hs[l_and_node, :] = hs_and.squeeze(0)
                    # Update function hidden state
                    msg = self.aggr_and_func(node_state, and_edge_index, and_edge_attr)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hf_and = torch.index_select(hf, dim=0, index=l_and_node)
                    _, hf_and = self.update_and_func(and_msg.unsqueeze(0), hf_and.unsqueeze(0))
                    hf[l_and_node, :] = hf_and.squeeze(0)

                # NOT Gate
                l_not_node = G.forward_index[layer_mask & not_mask]
                if l_not_node.size(0) > 0:
                    not_edge_index, not_edge_attr = subgraph(l_not_node, edge_index, dim=1)
                    # Update structure hidden state
                    msg = self.aggr_not_strc(hs, not_edge_index, not_edge_attr)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hs_not = torch.index_select(hs, dim=0, index=l_not_node)
                    _, hs_not = self.update_not_strc(not_msg.unsqueeze(0), hs_not.unsqueeze(0))
                    hs[l_not_node, :] = hs_not.squeeze(0)
                    # Update function hidden state
                    msg = self.aggr_not_func(hf, not_edge_index, not_edge_attr)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hf_not = torch.index_select(hf, dim=0, index=l_not_node)
                    _, hf_not = self.update_not_func(not_msg.unsqueeze(0), hf_not.unsqueeze(0))
                    hf[l_not_node, :] = hf_not.squeeze(0)
                
                # Update node state
                node_state = torch.cat([hs, hf], dim=-1)

        node_embedding = node_state.squeeze(0)
        hs = node_embedding[:, :self.dim_hidden]
        hf = node_embedding[:, self.dim_hidden:]

        return hs, hf
    
    def pred_prob(self, hf):
        prob = self.readout_prob(hf)
        prob = torch.clamp(prob, min=0.0, max=1.0)
        return prob
    
    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict_ = checkpoint['state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = self.state_dict()
        
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)
        
    def load_pretrained(self, pretrained_model_path = ''):
        if pretrained_model_path == '':
            pretrained_model_path = os.path.join(os.path.dirname(__file__), 'pretrained', 'model.pth')
        self.load(pretrained_model_path)
