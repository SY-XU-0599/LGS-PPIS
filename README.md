# LGS-PPIS

Code for the official implementation of paper "LGS-PPIS: A Local-Global Structural Information Aggregation Framework for Predicting Protein-Protein Interaction Sites"
---

Dependencies
---

python == 3.7.16

pytorch == 1.13.1

PyG (torch-geometric) == 2.3.1

torch-cluster == 2.1.1

torch-scatter == 2.0.5

torch-sparse == 0.6.17

torch-spline-conv == 1.2.2

scikit-learn == 1.0.2

scipy == 1.7.3

numpy == 1.21.5

Data preparation
---
The relevant data and trained model can be available at the [Link](https://pan.baidu.com/s/1Z1HBAHZnmzXCFjHVvbO7Ag?pwd=1234). If the above link fails, you can choose an [Alternate Link](https://zenodo.org/records/13952369/files/Data.rar?download=1).

Unzip the above file to the corresponding directory according to the code run prompt.

Test
---
`python test.py` used to reproduct the performence recorded in the paper.

Train
---
`python main.py`
