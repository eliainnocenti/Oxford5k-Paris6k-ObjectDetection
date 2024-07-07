"""
TODO: add description
"""

import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import xml.etree.ElementTree as ET

base_path = "../../../Data/"

def plot_bounding_box_annotation(image_path, annotation_xml):
    """

    :param image_path:
    :param annotation_xml:
    :return:
    """
    # Load the image
    image = Image.open(image_path)

    # Load the XML annotation
    tree = ET.parse(annotation_xml)
    root = tree.getroot()

    # Plot the image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Colors based on difficulty
    colors = {0: 'red', 1: 'orange', 2: 'yellow'}

    # Iterate over each object in the XML
    for obj in root.findall('object'):
        # Extract bounding box coordinates
        # TODO: check if there's another way to handle no integer coordinates
        xmin = int(round(float(obj.find('bndbox/xmin').text)))
        ymin = int(round(float(obj.find('bndbox/ymin').text)))
        xmax = int(round(float(obj.find('bndbox/xmax').text)))
        ymax = int(round(float(obj.find('bndbox/ymax').text)))

        # Extract difficulty and assign color
        difficulty = int(obj.find('difficult').text)
        color = colors.get(difficulty, 'red')  # Default to red if difficulty is not found

        # Extract object name for label
        name = obj.find('name').text

        # Create and add the bounding box rectangle with label
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none',
                                 label=name)
        ax.add_patch(rect)
        plt.text(xmin, ymin, name, color=color, weight='bold')

    ax.axis('off')
    plt.show()

def plot_bounding_box(image_path, xmin, ymin, xmax, ymax):
    """

    :param image_path:
    :param xmin:
    :param ymin:
    :param xmax:
    :param ymax:
    :return:
    """
    # Load the image
    image = Image.open(image_path)

    # Plot the image and the bounding box
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Create and add the bounding box rectangle
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    ax.axis('off')

    plt.show()

def test_query(query=0):
    """

    :param query:
    :return:
    """
    if query == 0:
        image_path = os.path.join(base_path, 'datasets', 'rparis6k', 'images', 'paris_defense_000605.jpg')
        annotation_xml = '../data/rparis6k/annotations/xml/paris_defense_000605.xml'
        if os.path.exists(image_path):
            if os.path.exists(annotation_xml):
                plot_bounding_box_annotation(image_path, annotation_xml)
            else:
                print(f"Annotation file not found: {annotation_xml}")
        else:
            print(f"Image file not found: {image_path}")

def test_image(dataset_name, image_name):
    """

    :param dataset_name:
    :param image_name:
    :return:
    """
    image_path = os.path.join(base_path, 'datasets', dataset_name, 'images', image_name + '.jpg')
    annotation_xml = os.path.join('../data/', dataset_name, 'annotations/xml', image_name + '.xml')
    if os.path.exists(image_path):
        if os.path.exists(annotation_xml):
            plot_bounding_box_annotation(image_path, annotation_xml)
        else:
            print(f"Annotation file not found: {annotation_xml}")
    else:
        print(f"Image file not found: {image_path}")

def test_monument(dataset_name, monument_name, size_test):
    """

    :param dataset_name:
    :param monument_name:
    :param size_test:
    :return:
    """
    images = []
    xmls = []
    j = 0
    for i in range(size_test):
        _dataset_name = dataset_name[1:-2]
        image_path, annotation_xml = '', ''
        while not os.path.exists(image_path) or not os.path.exists(annotation_xml):
            image_path = os.path.join(base_path, 'datasets', dataset_name, 'images', f'{_dataset_name}_{monument_name}_{j:06d}.jpg')
            annotation_xml = os.path.join('../data/', dataset_name, 'annotations/xml', f'{_dataset_name}_{monument_name}_{j:06d}.xml')
            j += 1
        images.append(image_path)
        xmls.append(annotation_xml)

    print(f"Number of images: {len(images)}")
    print(f"Number of xmls: {len(xmls)}")

    for i in range(size_test):
        plot_bounding_box_annotation(images[i], xmls[i])

def main():
    """

    :return:
    """
    #test_query(0)
    image_name = "paris_defense_000060" # easy example
    image_name = "paris_general_002529" # junk example
    image_name = "paris_defense_000101" # hard example
    #test_image('rparis6k', image_name)

    test_monument('rparis6k', 'defense', 10)

if __name__ == "__main__":
    main()
