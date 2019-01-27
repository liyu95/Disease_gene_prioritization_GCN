Due to the limit of the file size on Github, we store the data on Google Drive. Please download the data here: 
[data](https://drive.google.com/open?id=18yPVBjAvjtqLolno2RTAYt0Y_P-Hbdq7).

## genes_phenes.mat

1. GeneGene_Hs: The HumanNet gene interaction network of size 12331 x 12331. 
2. GenePhene: a cell array containing Gene-Phenotype networks of 9 species. 
3. GP_SPECIES: The names of the species corresponding to the networks in 'GenePhene' variable.
4. geneIds: The entrezdb ids of genes, corresponding to the rows of the matrix 'GeneGene_Hs' (or 'GenePhene' matrices).
5. pheneIds: a cell array containing OMIM ids for phenotypes of 9 species.
6. PhenotypeSimilaritiesLog: Similarity network between OMIM diseases.

## GeneFeatures.mat
Microarray expression data - vector of real-valued features for a gene per row (Refer paper for details).

## clinicalfeatures_tfidf.mat 
OMIM word-count data - term-document matrix for OMIM diseases (Refer paper for details).

