"""
TODO: add file and function descriptions
This script provides functions to visualize bounding box annotations on images for the Revisited Paris (rparis6k) dataset.

License:
---------
# TODO: check the license
This script uses image and annotation data from the Revisited Paris (rparis6k) dataset, which is provided under its respective license.
For more information on the dataset's license, please refer to the official dataset documentation.

Functions:
----------
1. plot_bounding_box_annotation(image_path, annotation_xml):
    Plots an image with its bounding box annotations overlaid.

2. plot_bounding_box(image_path, xmin, ymin, xmax, ymax):
    Plots an image with a single bounding box specified by the coordinates.

3. test_query(query=0):
    Tests the plotting of a bounding box annotation for a specific query image.

4. test_image(dataset_name, image_name):
    Tests the plotting of a bounding box annotation for a specified image in a specified dataset.

5. test_monument(dataset_name, monument_name, size_test):
    Tests the plotting of bounding box annotations for multiple images of a specified monument in a specified dataset.

Directory Structure:
---------------------
<root_data_directory>/
    datasets/
        rparis6k/
            images/
        annotations/
            xml/

Dependencies: # TODO: update
-------------
- os
- matplotlib
- PIL (Python Imaging Library)
- xml.etree.ElementTree

Usage:
------
To run this script, ensure that the required libraries are installed and the data directory is correctly set.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import xml.etree.ElementTree as ET

from matplotlib import text, patheffects
from collections import defaultdict
import math
import json

base_path = "../../../Data/"


def plot_bounding_box_annotation(image_path, annotation_xml):
    """
    Plots an image with bounding box annotations overlaid.

    :param image_path: Path to the image file.
    :param annotation_xml: Path to the XML annotation file.
    :return: None
    """
    # FIXME: handle the case where the annotation file is json
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


def plot_bounding_box(image_path, xmin, ymin, xmax, ymax, color='red'):
    """
    Plots an image with a single bounding box.

    :param image_path: Path to the image file.
    :param xmin: Minimum x-coordinate of the bounding box.
    :param ymin: Minimum y-coordinate of the bounding box.
    :param xmax: Maximum x-coordinate of the bounding box.
    :param ymax: Maximum y-coordinate of the bounding box.
    :param color: Color of the bounding box (default is 'red').
    :return: None
    """
    # Load the image
    image = Image.open(image_path)

    # Plot the image and the bounding box
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    if color == 'red':
        color = 'r'
    elif color == 'yellow':
        color = 'yellow'
    elif color == 'green':
        color = 'green'
    else:
        print("Color not recognized")
        return

    # Create and add the bounding box rectangle
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

    ax.axis('off')

    plt.show()


def test_query(query=0):
    """
    Tests the plotting of a bounding box annotation for a specific query image.

    :param query: Query number to test (default is 0).
    :return: None
    """
    # FIXME: with this implementations xml annotations are needed
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
    Tests the plotting of a bounding box annotation for a specified image in a specified dataset.

    :param dataset_name: Name of the dataset (e.g., 'rparis6k').
    :param image_name: Name of the image file (without extension).
    :return: None
    """
    # FIXME: with this implementations xml annotations are needed
    image_path = os.path.join(base_path, 'datasets', dataset_name, 'images', image_name + '.jpg')

    '''
    annotation_xml = os.path.join('../data/', dataset_name, 'annotations/xml', image_name + '.xml')
    if os.path.exists(image_path):
        if os.path.exists(annotation_xml):
            plot_bounding_box_annotation(image_path, annotation_xml)
        else:
            print(f"Annotation file not found: {annotation_xml}")
    else:
        print(f"Image file not found: {image_path}")
    '''

    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    # load json
    with open(os.path.join('../data', dataset_name, 'annotations', 'json', 'labels.json'), 'r') as file:
        labels_json = json.load(file)
    # search image id in labels_json['images']
    image_id = None
    for image in labels_json['images']:
        if image['file_name'] == image_name + '.jpg':
            image_id = image['id']
            break

    if image_id is None:
        print(f"Image not found in labels.json: {image_name}")
        return

    # get annotations for the image
    annotations = [annotation for annotation in labels_json['annotations'] if annotation['image_id'] == image_id]

    # plot image with annotations
    img = plt.imread(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    draw_bbox(ax, annotations, {item["id"]:item["name"] for item in labels_json["categories"]}, img.shape)
    plt.show()


def test_monument(dataset_name, monument_name, size_test):
    """
    Tests the plotting of bounding box annotations for multiple images of a specified monument in a specified dataset.

    :param dataset_name: Name of the dataset (e.g., 'rparis6k').
    :param monument_name: Name of the monument (e.g., 'defense').
    :param size_test: Number of images to test.
    :return: None
    """
    # FIXME: with this implementations xml annotations are needed
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


def test_image_imlist_85():
    image_path = os.path.join(base_path, 'datasets', 'rparis6k', 'images', 'paris_defense_000060.jpg')
    # 130, 17, 843, 732 red
    # 258, 88, 805, 619 red
    # 213, 67, 765, 609 red
    # 8, 104, 707, 913 red
    # 226, 77, 645, 510 red
    # 0, 0, 1023, 767 yellow
    # 134.7, 37.6, 893.3, 691.2 yellow
    # 0, 0, 767, 1023 yellow
    # 121.2, 48.8, 806.0, 733.0 green
    plot_bounding_box(image_path, 130, 17, 843, 732, 'red')
    plot_bounding_box(image_path, 258, 88, 805, 619, 'red')
    plot_bounding_box(image_path, 213, 67, 765, 609, 'red')
    plot_bounding_box(image_path, 8, 104, 707, 913, 'red')
    plot_bounding_box(image_path, 226, 77, 645, 510, 'red')
    plot_bounding_box(image_path, 0, 0, 1023, 767, 'yellow')
    plot_bounding_box(image_path, 134.7, 37.6, 893.3, 691.2, 'yellow')
    plot_bounding_box(image_path, 0, 0, 767, 1023, 'yellow')
    plot_bounding_box(image_path, 121.2, 48.8, 806.0, 733.0, 'green')

# TODO: it may be interesting if it shows two random images per category

def draw_outline(obj):
    obj.set_path_effects([patheffects.Stroke(linewidth=4,  foreground='black'), patheffects.Normal()])

def draw_box(ax, bb):
    patch = ax.add_patch(patches.Rectangle((bb[0],bb[1]), bb[2], bb[3], fill=False, edgecolor='red', lw=2))
    draw_outline(patch)

def draw_text(ax, bb, txt, disp):
    text = ax.text(bb[0], (bb[1]-disp), txt, verticalalignment='top', color='white', fontsize=10, weight='bold')
    draw_outline(text)

def draw_bbox(ax, annotations_list, id_to_label, image_shape):
    for annotation in annotations_list:
        cat_id = annotation["category_id"]
        bbox = annotation["bbox"]
        draw_box(ax, bbox)
        draw_text(ax, bbox, id_to_label[cat_id], image_shape[0] * 0.05)

def visualize(dataset_folder, max_examples=None):
    with open(os.path.join(dataset_folder, "labels.json"), "r") as f:
        labels_json = json.load(f)

    images = labels_json["images"]
    cat_id_to_label = {item["id"]:item["name"] for item in labels_json["categories"]}
    image_annots = defaultdict(list)

    for annotation_obj in labels_json["annotations"]:
        image_id = annotation_obj["image_id"]
        image_annots[image_id].append(annotation_obj)

    if max_examples is None:
        max_examples = len(image_annots.items())

    n_rows = math.ceil(max_examples / 3)
    fig, axs = plt.subplots(n_rows, 3, figsize=(24, n_rows*8)) # 3 columns(2nd index), 8x8 for each image

    for ind, (image_id, annotations_list) in enumerate(list(image_annots.items())[:max_examples]):
        ax = axs[ind//3, ind%3]
        img = plt.imread(os.path.join(dataset_folder, "images", images[image_id]["file_name"]))
        ax.imshow(img)
        draw_bbox(ax, annotations_list, cat_id_to_label, img.shape)

    plt.show()

#visualize(train_dataset_path, 9)


def main():
    """
    Main function to test the bounding box annotation visualization functions.

    Uncomment the relevant lines to test specific examples.

    :return: None
    """
    #test_query(0)
    image_name = "paris_defense_000060" # easy example
    image_name = "paris_general_002529" # junk example
    image_name = "paris_defense_000101" # hard example
    #test_image('rparis6k', image_name)

    #test_monument('rparis6k', 'defense', 10)

    #test_image_imlist_85()

    test_image('rparis6k', 'paris_defense_000000')


if __name__ == "__main__":
    main()
