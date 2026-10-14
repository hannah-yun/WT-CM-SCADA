import tensorflow as _tf

tf = _tf.compat.v1
tf.disable_v2_behavior()
import random

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds, labels, adj):
        preds_sub = preds
        labels_sub = labels

        self.num_nodes = tf.cast(tf.shape(adj)[0], tf.float32)
        self.total = tf.sparse_reduce_sum(adj)
        self.pos_weight = (self.num_nodes * self.num_nodes - self.total) / self.total
        self.norm = self.num_nodes * self.num_nodes / ((self.num_nodes * self.num_nodes - self.total) * 2)

        self.cost = self.norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=self.pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, adj):
        preds_sub = preds
        labels_sub = labels

        self.num_nodes = tf.cast(tf.shape(adj, tf.float32)[0])
        self.total = tf.sparse_reduce_sum(adj)
        self.pos_weight = (self.num_nodes * self.num_nodes - self.total) / self.total
        self.norm = self.num_nodes * self.num_nodes / ((self.num_nodes * self.num_nodes - self.total) * 2)

        self.cost = self.norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=self.pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / self.num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
