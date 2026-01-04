import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from src.models.densenet_baseline import build_densenet_baseline
from src.models.densenet_cbam import build_densenet_cbam


def get_data_generators(data_dir, img_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.densenet.preprocess_input,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.densenet.preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary"
    )

    val_gen = val_datagen.flow_from_directory(
        f"{data_dir}/val",
        target_size=img_size,
        batch_size=batch_size,
