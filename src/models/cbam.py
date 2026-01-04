import tensorflow as tf
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Dense,
    Reshape,
    Multiply,
    Concatenate,
    Conv2D
)

def channel_attention(input_feature, ratio=8):
    """
    Channel Attention Module
    """
    channel = input_feature.shape[-1]

    shared_dense_one = Dense(channel // ratio,
                              activation='relu',
                              kernel_initializer='he_normal',
                              use_bias=True)

    shared_dense_two = Dense(channel,
                              kernel_initializer='he_normal',
                              use_bias=True)

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    cbam_feature = tf.keras.layers.Add()([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)

    return Multiply()([input_feature, cbam_feature])


def spatial_attention(input_feature):
    """
    Spatial Attention Module
    """
    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)

    concat = Concatenate(axis=-1)([avg_pool, max_pool])

    cbam_feature = Conv2D(filters=1,
                          kernel_size=7,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    return Multiply()([input_feature, cbam_feature])


def cbam_block(input_feature, ratio=8):
    """
    Convolutional Block Attention Module (CBAM)
    """
    x = channel_attention(input_feature, ratio)
    x = spatial_attention(x)
    return x
