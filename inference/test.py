"""
TODO: add file and function descriptions
"""

import os.path
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

base_path = "../../../Data/"

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='../models/model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def load_image_into_numpy_array(path):
    """

    :param path:
    :return:
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    input_shape = input_details[0]['shape']
    image = tf.image.resize(image, (input_shape[1], input_shape[2]))
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
    return tf.expand_dims(image, 0).numpy()     # Add batch dimension


def run_inference(image_np):
    """

    :param image_np:
    :return:
    """
    interpreter.set_tensor(input_details[0]['index'], image_np)
    interpreter.invoke()

    # Get the results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes_scores = interpreter.get_tensor(output_details[1]['index'])

    return boxes, classes_scores


def visualize_detections(image_path, boxes, classes_scores, threshold=0.5):
    """

    :param image_path:
    :param boxes:
    :param classes_scores:
    :param threshold:
    :return:
    """
    image = plt.imread(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    height, width, _ = image.shape
    for box, class_score in zip(boxes[0], classes_scores[0]):
        score = np.max(class_score)
        if score > threshold:
            ymin, xmin, ymax, xmax = box
            rect = patches.Rectangle((xmin*width, ymin*height), (xmax-xmin)*width, (ymax-ymin)*height,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            class_id = np.argmax(class_score)
            plt.text(xmin*width, ymin*height, f'{class_id}: {score:.2f}', color='red')

    plt.show()


def main():
    """

    :return:
    """

    test_path = '../data/rparis6k/sets/test.txt'

    if not os.path.exists(test_path):
        print(f"Error: Test file not found: {test_path}")
        return

    with open(test_path, 'r') as f:
        test_images = [line.strip() for line in f]

    for image_name in test_images[:5]:
        image_path = os.path.join(base_path, 'datasets', 'rparis6k', 'images', image_name)
        image_np = load_image_into_numpy_array(image_path)
        boxes, classes_scores = run_inference(image_np)
        visualize_detections(image_path, boxes, classes_scores)


if __name__ == '__main__':
    main()
