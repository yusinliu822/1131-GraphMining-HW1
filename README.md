# Graph Mining and Optimization HW

## You can create the environment with anaconda

```bash
$ virtualenv venv
$ source venv/bin/activate
(venv)$ pip install -r ./requirements.txt
```

## Install packages

* scipy, networkx

```
pip install scipy networkx
```

* dgl, pytorch

* Install dgl
  * https://www.dgl.ai/pages/start.html
* Install pytorch
  * https://pytorch.org/get-started/locally/
  * recommend 2.2.0 version

## Run sample code

```python
python train.py --es_iters 30 --epochs 300 --use_gpu
```

## Dataset

* Unknown graph data
  * Label Split:
    * Train: 60, Valid: 30, Test: 1000
* File name description

```
  dataset
  │   ├── private_features.pkl # node feature
  │   ├── private_graph.pkl # graph edges
  │   ├── private_num_classes.pkl # number of classes
  │   ├── private_test_labels.pkl # X
  │   ├── private_test_mask.pkl # nodes indices of testing set
  │   ├── private_train_labels.pkl # nodes labels of training set
  │   ├── private_train_mask.pkl # nodes indices of training set
  │   ├── private_val_labels.pkl # nodes labels of validation set
  │   └── private_val_mask.pkl # nodes indices of validation set
```

## Model

I've experimented the following models with DGL:

- GCN 
    - Sample Code
        - dropout except first layer
    - 2-Layer
        - modified from sample code, but dropout except last layer
- GAT
    - I've tried GATConv and GATv2Conv
    - The best result of GATs is
        - `GATv2, hidden_size=16, num_heads=3, dropout=0.2, output_layer_heads=1`

- GraphSAGE
    - 2-layer (My Best Public Score)
        - `SAGE, hidden_dim=3, aggregator_type=gcn, epoch=300, droupout=0.5, optimizer=adam, lr=1e-2, weight_decay=5e-4`
        - I've tried all aggregator type that DGL supports(gcn, lstm, mean, pool). In this case, the performance rank is gcn > mean > pool >> lstm.

I also tried to apply contrastive learning for loss computing but failed with loss.backward() implementation.


## Kaggle Competition
Wang , Ching-Kai. 2024 NCU Graph Mining and Optimization HW. https://kaggle.com/competitions/2024-ncu-graph-mining-and-optimization-hw, 2024. Kaggle.