import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataset(filename):
    data = pd.read_csv(filename)
    pixels = data['pixels'].apply(lambda x: np.fromstring(x, sep=' '))
    images = np.vstack(pixels.values).reshape(-1, 48, 48, 1).astype('float32')
    images /= 255.0
    emotions = to_categorical(data['emotion'])
    return train_test_split(images, emotions, test_size=0.2, random_state=42)


def get_data_generators(X_train, X_test, y_train, y_test):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(X_train, y_train, batch_size=64)
    test_generator = test_datagen.flow(X_test, y_test, batch_size=64, shuffle=False)

    return train_generator, test_generator, X_test, y_test
