"""
Initializers used by the (legacy) Graph Autoencoder code.

Minimal updates for Python 3.11 / Apple Silicon environments where TensorFlow is
installed as TF2.x.

We keep the original TF1-style API surface (tf.random_uniform, tf.Variable, etc.)
by routing through `tf.compat.v1`.
"""

from __future__ import annotations

import numpy as np
import tensorflow as _tf

# Keep TF1 graph-style ops available under TF2 installations.
tf = _tf.compat.v1
tf.disable_v2_behavior()


def weight_variable_glorot(input_dim: int, output_dim: int, name: str = ""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010) init.

    Uses numpy for sampling so that np.random.seed() controls initialization.
    TF1 compat graph-level seeds are unreliable in TF2 compat mode.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = np.random.uniform(-init_range, init_range, (input_dim, output_dim)).astype(np.float32)
    return tf.Variable(initial, name=name)
