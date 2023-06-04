import os

import numpy as np
import tensorflow as tf

from keras import Sequential, layers, Model

from gesturelearner.constants import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_LABEL_INDEXES

def read_train_file(file_name):
    # Create a TFRecordDataset
    dataset = tf.data.TFRecordDataset([file_name])

    # Define the feature description
    feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string)
        # Add more features as needed
    }

    # Parse the record function
    def parse_record(record):
        example = tf.io.parse_single_example(record, feature_description)
        # Process the features as needed
        height = example['height']
        width = example['width']
        label = example['label']
        image = example['image']
        image = tf.io.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = image * (1. / 255) - 0.5
        image = tf.reshape(image, [height, width, 1])
        resized_image = tf.image.resize_with_crop_or_pad(
            image=image,
            target_height=IMAGE_HEIGHT,
            target_width=IMAGE_WIDTH
        )
        label_index = tf.reshape(label, [1, 1])
        sparse_tensor = tf.SparseTensor(label_index, [1.0], [NUM_LABEL_INDEXES])
        label = tf.sparse.to_dense(sparse_tensor)
        return height, width, label, resized_image

    # Apply the parse_record function to the dataset
    parsed_dataset = dataset.map(parse_record)
    # parsed_dataset = parsed_dataset.shuffle(buffer_size=1000)
    # parsed_dataset = parsed_dataset.batch(50)

    images = []
    labels = []
    for batch in parsed_dataset:
        image = np.array(batch[3])
        image[image != -0.5] = 0.5
        label = np.array(batch[2])
        images.append(image)
        labels.append(label)

    # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    # for batch in parsed_dataset:
    #     data = tf.reshape(batch[3], [IMAGE_HEIGHT, IMAGE_WIDTH])
    #     data = np.array(data)
    #     data[data != -0.5] = 1.0
    #     data[data == -0.5] = 0.0
    #     print(data)
    return np.array(images), np.array(labels)


def create_model():
    # Define the CNN architecture
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer=initializer),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=NUM_LABEL_INDEXES, activation='softmax')
    ])
    # print model
    model.summary()
    return model


def main(training_file, test_file, model_in=None, model_out=None):
    train_images, train_labels = read_train_file(training_file)
    test_images, test_labels = read_train_file(test_file)

    model = create_model()
    if model_in is not None:
        model.load_weights(model_in)
        # print(model.get_weights())
        # return
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(train_images, train_labels, epochs=10, batch_size=1105, shuffle=True)

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(test_loss)
    print(test_accuracy)

    if model_out is not None:
        model.save_weights(model_out)

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    training_file = os.path.join(current_path, 'sample_data', 'data_filtered.tfrecords')
    test_file = os.path.join(current_path, 'sample_data', 'data_filtered_test.tfrecords')
    model_in = os.path.join(current_path, 'sample_data', 'model', 'gesturelearner_weights.h5')
    model_out = os.path.join(current_path, 'sample_data', 'model', 'gesturelearner_weights.h5')
    main(training_file, test_file, model_in, model_out)