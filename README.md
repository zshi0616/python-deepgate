# DeepGate2: Functionality-Aware Circuit Representation Learning

Official code repository for the paper:  
[**DeepGate2: Functionality-Aware Circuit Representation Learning**](https://ieeexplore.ieee.org/document/10323798)

Authors: Zhengyuan Shi, Hongyang Pan, Sadaf Khan, Min Li, Yi Liu, Junhua Huang, Hui-Ling Zhen, Mingxuan Yuan, Zhufei Chu and Qiang Xu

DeepGate2 can serve as a circuit encoder to embed logic circuit into gate-level embedding vectors. Such general embeddings with rich structural and functional information can be applied to various downstream tasks, including testability analysis, SAT problem solving and logic synthesis. If you plan to explore more potential downstream tasks, please feel free to discuss with us (Email: zyshi21@cse.cuhk.edu.hk). We are looking forward to collaborate with you! 

DeepGate2 可以用作逻辑电路的Encoder，将逻辑电路的每个节点表示为一个embedding vector。这种富含结构和功能的向量能够被用于下游任务，包括但不限于：可测试性分析、SAT问题求解和逻辑综合。如果您希望探索更多可能，欢迎与我们讨论(邮箱：zyshi21@cse.cuhk.edu.hk)，期待与您合作！

## Abstract 
Circuit representation learning aims to obtain neural representations of circuit elements and has emerged as a promising research direction that can be applied to various EDA and logic reasoning tasks. Existing solutions, such as DeepGate, have the potential to embed both circuit structural information and functional behavior. However, their capabilities are limited due to weak supervision or flawed model design, resulting in unsatisfactory performance in downstream tasks. In this paper, we introduce **DeepGate2**, a novel functionality-aware learning framework that significantly improves upon the original DeepGate solution in terms of both learning effectiveness and efficiency. Our approach involves using pairwise truth table differences between sampled logic gates as training supervision, along with a well-designed and scalable loss function that explicitly considers circuit functionality. Additionally, we consider inherent circuit characteristics and design an efficient one-round graph neural network (GNN), resulting in an order of magnitude faster learning speed than the original DeepGate solution. Experimental results demonstrate significant improvements in two practical downstream tasks: logic synthesis and Boolean satisfiability solving.

## Installation
```sh
bash install.sh
```

## Prepare Training Dataset
```sh
mkdir data; cd data
wget https://github.com/Ironprop-Stone/python-deepgate/releases/download/dataset/train.zip
unzip train.zip 
```

## Pretrain DeepGate2
```sh
NUM_PROC=4
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC examples/train.py
```

## Generate Embedding Vectors 
This repo supports AIG and Bench format. 
For AIG format, see `examples/feature_extract.py`
```python
import deepgate
model = deepgate.Model()    # Create DeepGate
model.load_pretrained()      # Load pretrained model
parser = deepgate.AigParser()   # Create AigParser
graph = parser.read_aiger('./examples/test.aiger') # Parse AIG into Graph
hs, hf = model(graph)       # Model inference 
# hs: structural embeddings, hf: functional embeddings
# hs/hf: [N, D]. N: number of gates, D: embedding dimension (default: 128)
print(hs.shape, hf.shape)   
```

For Bench format, see `examples/feature_extract_bench.py`

## Cite DeepGate2
If DeepGate Family could help your project, please cite our work: 
```sh
@INPROCEEDINGS{10323798,
  author={Shi, Zhengyuan and Pan, Hongyang and Khan, Sadaf and Li, Min and Liu, Yi and Huang, Junhua and Zhen, Hui-Ling and Yuan, Mingxuan and Chu, Zhufei and Xu, Qiang},
  booktitle={2023 IEEE/ACM International Conference on Computer Aided Design (ICCAD)}, 
  title={DeepGate2: Functionality-Aware Circuit Representation Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-9},
  doi={10.1109/ICCAD57390.2023.10323798}}
```
