<h1>MVCL</h1>

Multi-view Graph Contrastive Learning for Cancer Driver Gene Identification

## Introduction

MNGCL is a multi-network contrastive learning framework for cancer driver gene identification. First, MNGCL constructs multiple gene-gene relationship networks, and these different networks describe the relationships among genes from different views. Then, it performs contrastive learning on different relationship networks to learn consistent gene feature representation in different networks from a holistic perspective. Thirdly considering that genes play different roles in different networks, we input the gene features learned in the previous step into each network-specific Chebyshev graph convolutional encoder to learn the unique feature representations of genes in the respective networks. Finally, we pass the unique features of genes learned from the three networks through a logistic regression classifier for the downstream cancer-driven gene identification task.

<h2>Requirements</h2>

- Python 3.8.18
- torch 2.1.2+cu121
- torch Geometric 2.1.0
- torch scatter 2.1.2
- torch sparse 0.6.18
- torch cluster 1.6.3
- torch spline conv 1.2.2
- pyYaml 6.0.1
- numpy 1.24.4

## Data

| File Name          | Format           | Size        | Description                                                                                                    |
| -------------------|:----------------:|:-----------:| -------------------------------------------------------------------------------------------------------------- |
| `CPDB_data.pkl`    | --               | --          | This file contains the PPI network, gene features, gene names, and gene label information of the CPDB dataset. |
| `ppi.pkl`          | torch.sparse_coo | 13627,13627 | Adjacency matrix (sparse matrix) of PPI network.                                                               |
| `ppi_selfloop.pkl` | torch.sparse_coo | 13627,13627 | Adjacency matrix (sparse matrix) of PPI network with self connection.                                          |
| `GO.pkl`           | torch.sparse_coo | 13627,13627 | Adjacency matrix (sparse matrix) of Gene functional similarity network.                                        |
| `pathway.pkl`      | torch.sparse_coo | 13627,13627 | Adjacency matrix (sparse matrix) of Pathway co-occurrence network.                                             |
| `Seq_Sim.pkl`      | torch.sparse_coo | 13627,13627 | Adjacency matrix (sparse matrix) of Protein sequence similarity network.                                       |
| `k_sets.pkl`       | dict             | --          | It preserves the data partitioning of the model during ten-fold cross-validation tests.                        |
| `Str_feature.pkl`  | tensor           | 13627,16    | This is a structural feature obtained through the Node2VEC algorithm on PPI network.                           |

## Running MVCL

Firstly,you should set the hyperparameter of the model through the configuration file config.yaml.

- `drop_edge_rate_1`:edge abandonment probability of Protein-protein network.

- `drop_edge_rate_2`:edge abandonment probability of Gene functional similarity network.

- `drop_edge_rate_3`:edge abandonment probability of Pathway co-occurrence network.

- `drop_edge_rate_4`:edge abandonment probability of Protein sequence similarity network.

- `drop_feature_rate_1`:feature masking probability of Protein-protein network.

- `drop_feature_rate_2`:feature masking probability of Gene functional similarity network.

- `drop_feature_rate_3`:feature masking probability of Pathway co-occurrence network.

- `drop_feature_rate_4`:feature masking probability of Protein sequence similarity network.

- `tau`:the contrastive learning loss temperature hyperparameter, `[0,1]`

- `num_clusters`:number of clusters, an integer > 0.

- `threshold`:confidence threshold for selecting high-confidence samples, `[0,1]`

Then,you can run `python train.py --dataset=CPDB --cancer_type=pan-cancer`

`--dataset`default is `CPDB` dataset,`--cancer_type`default is `pan-cancer`.

If you want to train a single cancer model, you can change the `cancer_type` for training, such as `python train.py --cancer_type=brca`
