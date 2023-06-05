import os
from .train import create_model

from PIL import Image
from numpy import asarray
import tensorflow as tf
import numpy as np
from io import BytesIO

from .constants import IMAGE_HEIGHT, IMAGE_WIDTH

def predict(image_in):
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_in = os.path.join(current_path, '../sample_data', 'model', 'gesturelearner_weights.h5')

    model = create_model()
    model.load_weights(model_in)

    image = Image.open(BytesIO(image_in))
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = image.convert('L')
    image = asarray(image)
    image = tf.cast(image, tf.float32)
    image = image * (1. / 255) - 0.5
    image = tf.reshape(image, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    image = np.array(image)
    image[image != -0.5] = 0.5
    images = np.array(image)
    results = model.predict(images)
    
    return np.argmax(results)