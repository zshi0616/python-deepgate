# DeepGate2: Functionality-Aware Circuit Representation Learning

Code repository for the paper:  
**DeepGate2: Functionality-Aware Circuit Representation Learning**, accepted by ICCAD23

Zhengyuan Shi, Hongyang Pan, Sadaf Khan, Min Li, Yi Liu, Junhua Huang, Hui-Ling Zhen, Mingxuan Yuan, Zhufei Chu and Qiang Xu

## Abstract 
Circuit representation learning aims to obtain neural representations of circuit elements and has emerged as a promising research direction that can be applied to various EDA and logic reasoning tasks. Existing solutions, such as DeepGate, have the potential to embed both circuit structural information and functional behavior. However, their capabilities are limited due to weak supervision or flawed model design, resulting in unsatisfactory performance in downstream tasks. In this paper, we introduce **DeepGate2**, a novel functionality-aware learning framework that significantly improves upon the original DeepGate solution in terms of both learning effectiveness and efficiency. Our approach involves using pairwise truth table differences between sampled logic gates as training supervision, along with a well-designed and scalable loss function that explicitly considers circuit functionality. Additionally, we consider inherent circuit characteristics and design an efficient one-round graph neural network (GNN), resulting in an order of magnitude faster learning speed than the original DeepGate solution. Experimental results demonstrate significant improvements in two practical downstream tasks: logic synthesis and Boolean satisfiability solving.

## Installation
```sh
python setup.py install
```



