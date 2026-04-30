from __future__ import annotations

import tensorflow as tf


def build_mobilenetv3_small_classifier(
    image_size: int,
    num_classes: int,
    dropout: float,
    train_backbone: bool = False,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    backbone = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
        pooling=None,
    )
    backbone._name = "MobileNetV3SmallBackbone"
    backbone.trainable = train_backbone
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="asl29_mobilenetv3_small")
    return model


def set_backbone_trainable_layers(model: tf.keras.Model, unfreeze_layers: int) -> None:
    backbone: tf.keras.Model | None = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "MobileNetV3Small" in layer.name:
            backbone = layer
            break

    if backbone is None:
        raise TypeError("Unable to locate MobileNetV3Small backbone in model.")

    backbone.trainable = True

    if unfreeze_layers <= 0:
        for layer in backbone.layers:
            layer.trainable = False
        return

    freeze_until = max(len(backbone.layers) - unfreeze_layers, 0)
    for idx, layer in enumerate(backbone.layers):
        layer.trainable = idx >= freeze_until

