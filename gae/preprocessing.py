import numpy as np
import scipy.sparse as sp
import networkx as nx


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj, adj_label, adj_norm, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['adj_label']: adj_label})
    feed_dict.update({placeholders['adj_norm']: adj_norm})
    return feed_dict


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


## New functions start here
def dummy_features(num_nodes):
    return sp.identity(num_nodes)


def load_graph(graph_path):
    graph = nx.from_dict_of_lists(nx.read_gpickle(graph_path))
    return nx.adjacency_matrix(graph), np.array(graph.degree)[:, 1]


def load_features(feature_path):
    features = nx.read_gpickle(feature_path)
    features = sparse_to_tuple(features.tocoo())
    return features


## This function is modified from mask_test_edges
def split_graph(adj, val_ratio=0.2, test_ratio=0.1):
    # Set negative example. Use != to speed up
    adj_false = sp.coo_matrix(np.ones(adj.shape)) - (adj != 0).astype(int)
    # Remove diagonal elements
    adj_false -= sp.dia_matrix((adj_false.diagonal()[np.newaxis, :], [0]), shape=adj_false.shape)

    adj.eliminate_zeros()
    adj_false.eliminate_zeros()

    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0
    assert np.diag(adj_false.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_triu_false = sp.triu(adj_false)
    edges = sparse_to_tuple(adj_triu)[0]
    edges_false = sparse_to_tuple(adj_triu_false)[0]
    edges_all = sparse_to_tuple(adj)[0]

    # Shuffle for positive examples
    np.random.shuffle(edges)
    num_test = int(np.floor(edges.shape[0] * test_ratio))
    num_val = int(np.floor(edges.shape[0] * val_ratio))
    val_edges = edges[:num_val]
    test_edges = edges[num_val:(num_val + num_test)]

    # Rest of the edges. Negative examples are not needed
    # Pick up first because num_val and num_test might change later
    train_edges = edges[(num_val + num_test):]

    # Shuffle for negative examples with same size or less
    np.random.shuffle(edges_false)
    if edges_false.shape[0] < num_val + num_test:
        num_test = int(np.floor(edges_false.shape[0] * test_ratio))
        num_val = int(np.floor(edges_false.shape[0] * val_ratio))
    val_edges_false = edges_false[:num_val]
    test_edges_false = edges_false[num_val:(num_val + num_test)]

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, val_edges, val_edges_false, test_edges, test_edges_false
