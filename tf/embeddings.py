"""
This module generates binary embeddings for graphs.
"""
import tensorflow as tf
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def hamming_distance(matrix: tf.Tensor) -> tf.Tensor:
    """
    Calculates the hamming distance between every pair of entries in the matrix.
    Elements are rows, features are columns

    Args:
        matrix: An N x x matrix with N number of objects/samples and d number
            of features.

    Returns:
        distances: Distance matrix of size N, N.
    """
    return tf.matmul(matrix, tf.transpose(tf.subtract(1., matrix))) + \
           tf.matmul(tf.subtract(1., matrix), tf.transpose(matrix))


def _denominator(dist: tf.Tensor) -> tf.Tensor:
    """
    Calculate the denominator of the softmax:

    ```
    d_{ij} = d_H(p_i, p_j)
    \head{d} = \max_{l\in\nodes}d_H(p_i, p_l)
    \head{d} + log\left(\sum_{k\in\nodes}\exp(-(d_{ik} - \head{d}))\right)
    ```

    Args:
        dist: Distance matrix, output of hamming_distance.

    Returns:
        (1, N) vector.
    """
    d_head = tf.stop_gradient(tf.reduce_max(dist))
    log_sum = tf.math.log(tf.reduce_sum(tf.exp(dist - d_head), axis=1))
    log_sum = d_head + log_sum
    return tf.expand_dims(log_sum, axis=0)


def _log_prob(log_p_matrix: tf.Variable, adj: tf.Tensor) -> float:
    matrix = tf.nn.sigmoid(log_p_matrix)
    dist = tf.multiply(-1, hamming_distance(matrix))
    denom = _denominator(dist)

    log_edge_probs = tf.subtract(dist,  denom)
    log_prob = tf.reduce_sum(tf.multiply(tf.expand_dims(adj, 0), log_edge_probs))
    return -1 * log_prob


adj = nx.adj_matrix(nx.karate_club_graph()).todense().astype(np.float32)
r = np.random.randn(adj.shape[0], 8).astype(np.float32)
# r = r - r.max()
log_ps = tf.Variable(r)

opt = tf.keras.optimizers.Adam()
losses = []
for i in range(10000):
    with tf.GradientTape() as tape:
        loss = -1 * _log_prob(log_ps, adj)
        losses.append(loss)
    grads = tape.gradient(loss, log_ps)
    opt.apply_gradients(zip([grads], [log_ps]))
    if len(losses) % 100 == 0:
        ax = plt.subplot()
        ax.plot(losses)
        ax.set_xlabel("Steps")
        ax.set_ylabel("LogProb")
        plt.savefig('loss.pdf', format='pdf')
        plt.close('all')

ps = tf.nn.sigmoid(log_ps)
tmp = hamming_distance(tf.cast(ps > 0.5, tf.float32)).numpy()
for i in range(tmp.shape[0]):
    for j in range(tmp.shape[1]):
        print("{:.4f}".format(tmp[i, j]), end=' ')
    print()
tmp = tf.cast(ps > 0.5, tf.float32).numpy()
for i in range(tmp.shape[0]):
    for j in range(tmp.shape[1]):
        print("{:.4f}".format(tmp[i, j]), end=' ')
    print()
plt.imshow(hamming_distance(tf.cast(ps > 0.5, tf.float32)), cmap='Greens')
plt.savefig("distmat.pdf", format="pdf")

a = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)
a = np.array([
    [0.1, 0.1],
    [0.2, 0.8],
    [0.99, 0.1],
    [0.7, 0.8]
], dtype=np.float32)
print(hamming_distance(a).numpy())

g = nx.from_numpy_array(nx.adj_matrix(nx.karate_club_graph()).todense().astype(np.float32))

