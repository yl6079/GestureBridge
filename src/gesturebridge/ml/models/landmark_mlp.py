"""Tiny landmark-only MLP for ASL29.

Input: 63-d vector (21 hand landmarks × xyz, wrist-centered + scale-normalized).
Output: 29 class probabilities.

This is the architecture pattern from the pre-yizheng `shufeng` baseline,
modernized for tf.keras. ~30K params; trains in minutes on CPU.
"""
from __future__ import annotations

import tensorflow as tf


def build_landmark_mlp(num_classes: int = 29, dropout: float = 0.3) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(63,), name="landmarks")
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="asl29_landmark_mlp")
