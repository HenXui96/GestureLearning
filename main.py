import os
from gesturelearner.train import create_model

from PIL import Image
from numpy import asarray
import tensorflow as tf
import numpy as np

from gesturelearner.constants import IMAGE_HEIGHT, IMAGE_WIDTH

def main(model_in, image_in):
    model = create_model()
    model.load_weights(model_in)

    image = Image.open(image_in)
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = image.convert('L')
    # print(image.format)
    # print(image.size)
    # print(image.mode)
    # image.show()
    # convert image to numpy array
    image = asarray(image)
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    np.set_printoptions(suppress=True)
    image = tf.cast(image, tf.float32)
    image = image * (1. / 255) - 0.5
    image = tf.reshape(image, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    # image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image = np.array(image)
    image[image != -0.5] = 0.5
    # image[image != -0.5] = 1.0
    # image[image == -0.5] = 0.0
    # print(image)

    images = np.array(image)
    results = model.predict(images)
    print(results)
    print(np.argmax(results))

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_in = os.path.join(current_path, 'sample_data', 'model', 'gesturelearner_weights.h5')
    image_in = os.path.join(current_path, 'sample_data', 'image', 'test.png')
    main(model_in, image_in)