# PGCN: Disease gene prioritization by disease and gene embedding through GCN
Disease gene prioritization is a fundamental step towards molecular diagnosis and treatment of diseases. This problem is highly challenging due to the very limited yet noisy knowledge of genes, diseases and, even more, on their associations. Despite the development of computational methods for disease gene prioritization, the performance of the existing methods is limited by manually-crafted features, network topology, or pre-defined rules of data fusion. Here we propose a novel graph convolutional network-based disease gene prioritization method, PGCN, through the systematic embedding of the heterogeneous network made by genes and diseases, as well as their individual features. The embedding learning model and the association prediction model are trained together in an end-to-end manner. We compared PGCN with five state-of-the-art methods on the Online Mendelian Inheritance in Man (OMIM) dataset, by challenging them on recovering missing associations, and on discovering associations for novel genes and/or diseases that are not seen in the training. Results show the significant improvements of PGCN over the existing methods. We further demonstrate that our embedding has biological meaning and can capture functional groups of genes.

More details can be referred to the [paper](https://www.biorxiv.org/content/10.1101/532226v1).

```
@article{li2019pgcn,
  title={PGCN: Disease gene prioritization by disease and gene embedding through graph convolutional neural networks},
  author={Li, Yu and Kuwahara, Hiroyuki and Yang, Peng and Song, Le and Gao, Xin},
  journal={bioRxiv},
  pages={532226},
  year={2019},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Overview
<p align="center">
<img src="https://github.com/lykaust15/Disease_gene_prioritization_GCN/blob/master/figures/link_prediction.png" width="400"/>
</p>
Disease gene prioritization as a link prediction problem. The heterogeneous network contains three components, the gene-gene interaction network, the disease similarity network, and the disease-gene association network. The potential disease gene associations can be considered as missing links in the disease-gene association network. Our goal is to predict those links given the heterogeneous network and additional raw representations of the nodes (diseases and genes).

<p align="center">
<img src="https://github.com/lykaust15/Disease_gene_prioritization_GCN/blob/master/figures/gcn.png" width="800"/>
</p>
Overview of the proposed method. (A) The input of our model contains two components, the heterogeneous network and the additional information for the nodes. As for the heterogeneous network, we used HumanNet as the gene network, disease similarity network as the disease network, and the associations from OMIM as the disease-gene network. For the additional information of diseases, we used Disease Ontology similarity and the TF-IDF calculated from OMIM. For the additional information of genes, we used association matrices from other species and the gene expression microarray data. (B) Examples of one layer of the graph convolutional neural network update for learning node embeddings. For each node, the model aggregates information from its neighbor nodes' previous layer embeddings and then apply activation to obtain the current layer embedding of that node. Note that for different nodes, the computational graphs can be different but the parameters are shared for the same operation in different computational graphs. (C) The link prediction model. We model the edge prediction from the learned node embeddings with bilinear edge decoder. (D) The cross-entropy loss calculated from the ground truth and the output of the link prediction model for certain edges (or non-edges) is used as the loss function to train both the node embedding model and the edge decoding model jointly in an end-to-end fashion.

## Tested environment
* Centos 7
* Python 3.6.7

## Install requirements
All the related packages have been summarized in *requirements.txt*. One can install all the packages with following command.
```
pip install -r requirements.txt
```

(better to construct a virtual environment using conda and install the package inside the environment)

## Download the data
Due to the limit of the file size on Github, we store the data on Google Drive. Please download the data here: 
[data](https://drive.google.com/open?id=18yPVBjAvjtqLolno2RTAYt0Y_P-Hbdq7).

## Run the code
One can run the code using the following command after configuring the environment and downloading the data.
```
python main_prioritization.py
```

## Result
The prediction matrix file can be downloaded here: [result](https://drive.google.com/open?id=1CDCrL9qmlirJUktnUULprUbDj9oUY0-W).

Here is the embedding clustering result. For more explanation, please refer to the manuscript.
<p align="center">
<img src="https://github.com/lykaust15/Disease_gene_prioritization_GCN/blob/master/figures/embedding.png" width="400"/>
</p>

## More explanation
For calculating BEDROC, here we provide the function from the skchem package for the reference. For more accurate calculation, one can output the prediction and use R packages to do the calculation.


## Credits
We would like to thank for the SNAP group for open-sourcing the decagon code: [decagon](https://github.com/marinkaz/decagon).

*This tool is for academic purposes and research use only. Any commercial use is subject for authorization from King Abdullah University of Science and technology “KAUST”. Please contact us at ip@kaust.edu.sa.*