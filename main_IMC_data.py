from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
import time
import os

import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from skchem.metrics import bedroc_score
import pickle

from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Train on GPU
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

np.random.seed(0)

###########################################################
#
# Functions
#
###########################################################
def tsne_visualization(matrix):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, random_state=0,
            n_iter=1000)
    tsne_results = tsne.fit_transform(matrix)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def draw_graph(adj_matrix):
    G = nx.from_scipy_sparse_matrix(adj_matrix)
    pos = nx.spring_layout(G, iterations=100)
    d = dict(nx.degree(G))
    nx.draw(G, pos, node_color=range(3215), nodelist=d.keys(), 
        node_size=[v*20+20 for v in d.values()], cmap=plt.cm.Dark2)
    plt.show()

def get_accuracy_scores(edges_pos, edges_neg, edge_type, name=None):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)

        # need to deal with the ground truth which is not 1
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=200)
    bedroc_sc = bedroc_score(labels_all, preds_all)
    if name!=None:
        with open(name, 'wb') as f:
            pickle.dump([labels_all, preds_all], f)
    # calculate the apk for each disease
    if edge_type==(0,1,0):
        predicted_matrix = rec
        true_matrix = adj_mats_orig[edge_type[:2]][edge_type[2]].toarray()
        apk_list= list()
        for i in list(set(list(edges_pos[edge_type[:2]][edge_type[2]][:,1]))):
            actual_pos = np.where(true_matrix[:,i])[0]
            if len(actual_pos)==0:
                continue
            predicted_pos = np.argsort(predicted_matrix[:,i])[::-1]
            apk_tmp = rank_metrics.apk(list(actual_pos), list(predicted_pos), k=1)
            apk_list.append(apk_tmp)
        print(np.mean(apk_list))

        # calculate average recall at k for each disease
        for k in range(1, 20):
            ark_list= list()
            for i in list(set(list(edges_pos[edge_type[:2]][edge_type[2]][:,1]))):
                actual_pos = np.where(true_matrix[:,i])[0]
                if len(actual_pos)==0:
                    continue
                predicted_pos = np.argsort(predicted_matrix[:,i])[::-1]
                ark_tmp = rank_metrics.ark(list(actual_pos), list(predicted_pos), k=k)
                ark_list.append(ark_tmp)
            print(np.mean(ark_list))

        # for new disease, no link in training data, while there are link in the test data
        training_disease = list(set(list(minibatch.train_edges[edge_type[:2]][edge_type[2]][:,1])))
        for k in range(1, 20):
            ark_list= list()
            for i in list(set(list(edges_pos[edge_type[:2]][edge_type[2]][:,1]))):
                actual_pos = np.where(true_matrix[:,i])[0]
                if len(actual_pos)==0 or i in training_disease:
                    continue
                predicted_pos = np.argsort(predicted_matrix[:,i])[::-1]
                ark_tmp = rank_metrics.ark(list(actual_pos), list(predicted_pos), k=k)
                ark_list.append(ark_tmp)
            print(np.mean(ark_list))


        # for new genes, no link in training data, while there are link in the test data
        training_genes = set(list(minibatch.train_edges[edge_type[:2]][edge_type[2]][:,0]))
        for k in range(1, 20):
            ark_list= list()
            for i in list(set(list(edges_pos[edge_type[:2]][edge_type[2]][:,1]))):
                actual_pos = np.where(true_matrix[:,i])[0]
                actual_pos = set(list(actual_pos)) - training_genes
                if len(actual_pos)==0:
                    continue
                predicted_pos = np.argsort(predicted_matrix[:,i])[::-1]
                ark_tmp = rank_metrics.ark(list(actual_pos), list(predicted_pos), k=k)
                ark_list.append(ark_tmp)
            print(np.mean(ark_list))

        # for novel associations
        for k in range(1, 20):
            ark_list= list()
            for i in list(set(list(edges_pos[edge_type[:2]][edge_type[2]][:,1]))):
                actual_pos = np.where(true_matrix[:,i])[0]
                actual_pos = set(list(actual_pos)) - training_genes
                if len(actual_pos)==0 or i in training_disease:
                    continue
                predicted_pos = np.argsort(predicted_matrix[:,i])[::-1]
                ark_tmp = rank_metrics.ark(list(actual_pos), list(predicted_pos), k=k)
                ark_list.append(ark_tmp)
            print(np.mean(ark_list))        

    return roc_sc, aupr_sc, apk_sc, bedroc_sc


def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
        'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i,j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders

def network_edge_threshold(network_adj, threshold):
    edge_tmp, edge_value, shape_tmp = preprocessing.sparse_to_tuple(network_adj)
    preserved_edge_index = np.where(edge_value>threshold)[0]
    preserved_network = sp.csr_matrix(
        (edge_value[preserved_edge_index], 
        (edge_tmp[preserved_edge_index,0], edge_tmp[preserved_edge_index, 1])),
        shape=shape_tmp)
    return preserved_network


# Construct the three networks
gene_phenes_path = '../IMC/genes_phenes.mat'
f = h5py.File(gene_phenes_path, 'r')
gene_network_adj = sp.csc_matrix((np.array(f['GeneGene_Hs']['data']),
    np.array(f['GeneGene_Hs']['ir']), np.array(f['GeneGene_Hs']['jc'])),
    shape=(12331,12331))
gene_network_adj = gene_network_adj.tocsr()
disease_network_adj = sp.csc_matrix((np.array(f['PhenotypeSimilarities']['data']),
    np.array(f['PhenotypeSimilarities']['ir']), np.array(f['PhenotypeSimilarities']['jc'])),
    shape=(3215, 3215))
disease_network_adj = disease_network_adj.tocsr()

# try to cut a threshold of the disease network to reduce the number of edges
disease_network_adj = network_edge_threshold(disease_network_adj, 0.2)


dg_ref = f['GenePhene'][0][0]
gene_disease_adj = sp.csc_matrix((np.array(f[dg_ref]['data']),
    np.array(f[dg_ref]['ir']), np.array(f[dg_ref]['jc'])),
    shape=(12331, 3215))
gene_disease_adj = gene_disease_adj.tocsr()


# load novel associations
novel_associations_adj = sp.csc_matrix((np.array(f['NovelAssociations']['data']),
    np.array(f['NovelAssociations']['ir']), np.array(f['NovelAssociations']['jc'])),
    shape=(12331,3215))

# Build the gene feature
gene_feature_path = '../IMC/GeneFeatures.mat'
f_gene_feature = h5py.File(gene_feature_path,'r')
gene_feature_exp = np.array(f_gene_feature['GeneFeatures'])
gene_feature_exp = np.transpose(gene_feature_exp)
gene_network_exp = sp.csc_matrix(gene_feature_exp)

row_list = [3215, 1137, 744, 2503, 1143, 324, 1188, 4662, 1243]
gene_feature_list_other_spe = list()
for i in range(1,9):
    dg_ref = f['GenePhene'][i][0]
    disease_gene_adj_tmp = sp.csc_matrix((np.array(f[dg_ref]['data']),
        np.array(f[dg_ref]['ir']), np.array(f[dg_ref]['jc'])),
        shape=(12331, row_list[i]))
    gene_feature_list_other_spe.append(disease_gene_adj_tmp)

# Build the disease feature
disease_tfidf_path = '../IMC/clinicalfeatures_tfidf.mat'
f_disease_tfidf = h5py.File(disease_tfidf_path)
disease_tfidf = np.array(f_disease_tfidf['F'])
disease_tfidf = np.transpose(disease_tfidf)
disease_tfidf = sp.csc_matrix(disease_tfidf)

# finish the drug drug network
drug_drug_adj_list= list()
drug_drug_adj_list.append(disease_network_adj)

val_test_size = 0.1
n_genes = 12331
n_drugs = 3215
n_drugdrug_rel_types = len(drug_drug_adj_list)
gene_adj = gene_network_adj
gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()

gene_drug_adj = gene_disease_adj
drug_gene_adj = gene_drug_adj.transpose(copy=True)

drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]


# data representation
adj_mats_orig = {
    (0, 0): [gene_adj, gene_adj.transpose(copy=True)],
    (0, 1): [gene_drug_adj],
    (1, 0): [drug_gene_adj],
    (1, 1): drug_drug_adj_list + [x.transpose(copy=True) for x in drug_drug_adj_list],
}
degrees = {
    0: [gene_degrees, gene_degrees],
    1: drug_degrees_list + drug_degrees_list,
}

##################need to consider non-sparse feature##################
# featureless (genes)
gene_feat = sp.hstack(gene_feature_list_other_spe+[gene_feature_exp])
gene_nonzero_feat, gene_num_feat = gene_feat.shape
gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

# gene_feat = sp.identity(n_genes)
# gene_nonzero_feat, gene_num_feat = gene_feat.shape
# gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

# features (drugs)
drug_feat = disease_tfidf
drug_nonzero_feat, drug_num_feat = drug_feat.shape
drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())


# drug_feat = sp.identity(n_drugs)
# drug_nonzero_feat, drug_num_feat = drug_feat.shape
# drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

##################need to consider non-sparse feature##################


# data representation
num_feat = {
    0: gene_num_feat,
    1: drug_num_feat,
}
nonzero_feat = {
    0: gene_nonzero_feat,
    1: drug_nonzero_feat,
}
feat = {
    0: gene_feat,
    1: drug_feat,
}

edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
# edge_type2decoder = {
#     (0, 0): 'bilinear',
#     (0, 1): 'bilinear',
#     (1, 0): 'bilinear',
#     (1, 1): 'bilinear',
# }

edge_type2decoder = {
    (0, 0): 'innerproduct',
    (0, 1): 'innerproduct',
    (1, 0): 'innerproduct',
    (1, 1): 'innerproduct',
}

edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
num_edge_types = sum(edge_types.values())
print("Edge types:", "%d" % num_edge_types)

if __name__ == '__main__':

    ###########################################################
    #
    # Settings and placeholders
    #
    ###########################################################

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 1, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
    flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
    flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
    flags.DEFINE_boolean('bias', True, 'Bias term.')
    # Important -- Do not evaluate/print validation performance every iteration as it can take
    # substantial amount of time
    PRINT_PROGRESS_EVERY = 150

    print("Defining placeholders")
    placeholders = construct_placeholders(edge_types)

    ###########################################################
    #
    # Create minibatch iterator, model and optimizer
    #
    ###########################################################

    print("Create minibatch iterator")
    minibatch = EdgeMinibatchIterator(
        adj_mats=adj_mats_orig,
        feat=feat,
        edge_types=edge_types,
        batch_size=FLAGS.batch_size,
        val_test_size=val_test_size
    )

    print("Create model")
    model = DecagonModel(
        placeholders=placeholders,
        num_feat=num_feat,
        nonzero_feat=nonzero_feat,
        edge_types=edge_types,
        decoders=edge_type2decoder,
    )

    print("Create optimizer")
    with tf.name_scope('optimizer'):
        opt = DecagonOptimizer(
            embeddings=model.embeddings,
            latent_inters=model.latent_inters,
            latent_varies=model.latent_varies,
            degrees=degrees,
            edge_types=edge_types,
            edge_type2dim=edge_type2dim,
            placeholders=placeholders,
            batch_size=FLAGS.batch_size,
            margin=FLAGS.max_margin
        )

    print("Initialize session")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feed_dict = {}
    saver = tf.train.Saver()
    saver.restore(sess,'./model.ckpt')

    ###########################################################
    #
    # Train model
    #
    ###########################################################

    print("Train model")
    for epoch in range(FLAGS.epochs):

        minibatch.shuffle()
        itr = 0
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
            feed_dict = minibatch.update_feed_dict(
                feed_dict=feed_dict,
                dropout=FLAGS.dropout,
                placeholders=placeholders)

            t = time.time()

            # Training step: run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict)
            train_cost = outs[1]
            batch_edge_type = outs[2]

            if itr % PRINT_PROGRESS_EVERY == 0:
                val_auc, val_auprc, val_apk, val_bedroc = get_accuracy_scores(
                    minibatch.val_edges, minibatch.val_edges_false,
                    minibatch.idx2edge_type[minibatch.current_edge_type_idx])

                print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:", "%04d" % batch_edge_type,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                      "val_apk=", "{:.5f}".format(val_apk), "val_bedroc=", "{:.5f}".format(val_bedroc),
                      "time=", "{:.5f}".format(time.time() - t))

            itr += 1

    print("Optimization finished!")

    for et in [2,3]:
        roc_score, auprc_score, apk_score, bedroc = get_accuracy_scores(
            minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
        print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
        print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(roc_score))
        print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(auprc_score))
        print("Edge type:", "%04d" % et, "Test AP@k score", "{:.5f}".format(apk_score))
        print("Edge type:", "%04d" % et, "Test BEDROC score", "{:.5f}".format(bedroc))
        print()
    # saver.save(sess, './model_no_disease_feature.ckpt')

# import pandas as pd
# df = pd.DataFrame()
# df['label'] =labels_all
# df['predict'] = preds_all
# df.to_csv('result_sample.csv',sep=',')

# Get AUC, PRC

# edges_pos, edges_neg, edge_type = minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[2]

# check the novel associations predictions
# first remove all the existing edges from the prediction
# rec_flatten = rec.flatten()
# previous_network_flatten = gene_disease_adj.toarray().flatten()
# previous_edge_index = np.where(previous_network_flatten)[0]
# new_edge_index = np.where(novel_associations_adj.toarray().flatten())[0]
# sort_index = np.argsort(-rec_flatten)
# rank_metrics.apk(list(new_edge_index), list(sort_index), k=10000)