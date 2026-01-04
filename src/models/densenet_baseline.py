import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


def build_densenet_baseline(input_shape=(224, 224, 3), trainable_base=False):
    """
    DenseNet121 baseline model (no attention)
    """

    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    base_model.trainable = trainable_base

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    return model
