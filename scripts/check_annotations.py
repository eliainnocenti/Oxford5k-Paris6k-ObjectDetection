"""
This script checks the consistency of image annotations for datasets such as rparis6k, ensuring that
each image has corresponding objects correctly annotated.

Functions:
----------
1. load_pickle(file_path):
    Loads data from a pickle file.

2. list_objects_in_annotations(annotation_xml):
    Lists all objects in an XML annotation file.

3. check_annotations(data_dir, xml_dir):
    Checks annotations for consistency, identifying images with no objects or objects with different names.

4. find_imlist_idx(data, imname):
    Finds the index of an image in the 'imlist'.

5. find_qimlist_idx(data, qimname):
    Finds the index of a query image in the 'qimlist'.

6. find_presence_in_gnd(data, imname):
    Finds the presence of an image in the ground truth data.

7. main():
    Main function to check annotations and identify inconsistencies.

Dependencies:
-------------
- os
- pickle
- xml.etree.ElementTree

Usage:
------
To run this script, ensure that the required libraries are installed and the data directory is correctly set.
"""

import os
import pickle
import xml.etree.ElementTree as ET

base_path = "../../../Data/"


def load_pickle(file_path):
    """
    Loads data from a pickle file.

    :param file_path: Path to the pickle file.
    :return: Loaded data.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def list_objects_in_annotations(annotation_xml):
    """
    Lists all objects in an XML annotation file.

    :param annotation_xml: Path to the XML annotation file.
    :return: List of object names in the annotation.
    """
    # Load the XML annotation
    tree = ET.parse(annotation_xml)
    root = tree.getroot()

    # List of objects
    objects = []

    # Iterate over each object in the XML
    for obj in root.findall('object'):
        # Extract object name
        name = obj.find('name').text
        objects.append(name)

    return objects


def check_annotations(data_dir, xml_dir):
    """
    Checks annotations for consistency, identifying images with no objects or objects with different names.

    :param data_dir: Directory containing image files.
    :param xml_dir: Directory containing XML annotation files.
    :return: Tuple of lists (images_with_no_objects, images_with_objects_with_different_name).
    """
    # iterate over all images (that have annotations) and check if they have objects (monuments) in the annotation file that have a different name than the image file
    images_with_no_objects = []
    images_with_objects_with_different_name = []
    for root, _, files in os.walk(xml_dir):
        for file in files:
            if file.endswith('.xml'):
                image_file = os.path.splitext(file)[0] + '.jpg'
                image_path = os.path.join(data_dir, image_file)
                annotation_path = os.path.join(root, file)

                if not os.path.exists(image_path):
                    print(f"Error: Image file not found: {image_path}")
                    continue

                if not os.path.exists(annotation_path):
                    print(f"Error: Annotation file not found: {annotation_path}")
                    continue

                image_name = os.path.splitext(image_file)[0]
                annotation_name = os.path.splitext(file)[0]

                if image_name != annotation_name:
                    print(f"Error: Image and annotation file names do not match: {image_name} != {annotation_name}")
                    continue

                image_name = image_name[6:]
                _image_name = image_name.split('_')[0]

                image_objects = list_objects_in_annotations(annotation_path)

                if len(image_objects) == 0:
                    #print(f"Error: No objects found in annotation file: {annotation_path}")
                    images_with_no_objects.append(image_name)
                    continue

                if _image_name not in image_objects:
                    #print(f"Error: Image name not found in objects: {image_name} not in {image_objects}")
                    images_with_objects_with_different_name.append(image_name)
                    continue

    return images_with_no_objects, images_with_objects_with_different_name


def find_imlist_idx(data, imname):
    """
    Finds the index of an image in the 'imlist'.

    :param data: Data loaded from the pickle file.
    :param imname: Image name to find in the 'imlist'.
    :return: Index of the image in the 'imlist', or -1 if not found.
    """
    imlist = data['imlist']
    for i, im in enumerate(imlist):
        if im == imname:
            return i
    return -1


def find_qimlist_idx(data, qimname):
    """
    Finds the index of a query image in the 'qimlist'.

    :param data: Data loaded from the pickle file.
    :param qimname: Query image name to find in the 'qimlist'.
    :return: Index of the query image in the 'qimlist', or -1 if not found.
    """
    qimlist = data['qimlist']
    for i, qim in enumerate(qimlist):
        if qim == qimname:
            return i
    return -1


def find_presence_in_gnd(data, imname):
    """
    Finds the presence of an image in the ground truth data.

    :param data: Data loaded from the pickle file.
    :param imname: Image name to check in the ground truth data.
    :return: Dictionary of presence information.
    """
    imlist_idx = find_imlist_idx(data, imname)
    gnd = data['gnd']
    presence = {}
    for query_idx in range(len(gnd)):
        for level in ['easy', 'hard', 'junk']:
            for im in gnd[query_idx][level]:
                if im == imlist_idx:
                    if query_idx not in presence:
                        presence[query_idx] = {}
                    presence[query_idx][level] = []
                    presence[query_idx][level].append(im)
    return presence


def main():
    """
    Main function to check annotations and identify inconsistencies.

    :return: None
    """
    data_dir = os.path.join(base_path, 'datasets', 'rparis6k', 'images')
    xml_dir = os.path.join('../data', 'rparis6k', 'annotations', 'xml')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return
    if not os.path.exists(xml_dir):
        print(f"Error: XML directory not found: {xml_dir}")
        return

    images_with_no_objects, images_with_objects_with_different_name = check_annotations(data_dir, xml_dir)

    #images_with_no_objects += ['paris_defense_000060'] # paris_defense_000060 has objects

    print(f"Number of images with no objects: {len(images_with_no_objects)}")
    print(f"Number of images with objects with different name: {len(images_with_objects_with_different_name)}")

    #print(f"Images with no objects: {images_with_no_objects}")
    #print(f"Images with objects with different name: {images_with_objects_with_different_name}")

    n_xml_files = len([name for name in os.listdir(xml_dir) if name.endswith('.xml')])
    print(f"\"Valid\" images: {n_xml_files - (len(images_with_no_objects) + len(images_with_objects_with_different_name))}")

    data = load_pickle(os.path.join(base_path, 'datasets', 'rparis6k', 'gnd_rparis6k.pkl'))

    for imname in images_with_no_objects:
        presence = find_presence_in_gnd(data, imname)
        if len(presence) != 0:
            print(f"Error: Image {imname} has presence in the ground truth.")

    '''
    # I'm checking if the image has truly no objects
    data = load_pickle(os.path.join(base_path, 'datasets', 'rparis6k', 'gnd_rparis6k.pkl'))
    imname = 'paris_museedorsay_000402'
    imlist_idx = find_imlist_idx(data, imname)
    qimlist_idx = find_qimlist_idx(data, imname)
    if imlist_idx == -1:
        print(f"Error: Image {imname} not found in the imlist.")
        if qimlist_idx == -1:
            print(f"Error: Image {imname} not found in the qimlist.")
        else:
            print(f"Image {imname} is at index {qimlist_idx} in the qimlist.")
    else:
        print(f"Image {imname} is at index {imlist_idx} in the imlist.")
    #presence = find_presence_in_gnd(data, imname)
    #print(f"Presence: {presence}")
    imname = 'paris_defense_000060'
    imlist_idx = find_imlist_idx(data, imname)
    qimlist_idx = find_qimlist_idx(data, imname)
    if imlist_idx == -1:
        print(f"Error: Image {imname} not found in the imlist.")
        if qimlist_idx == -1:
            print(f"Error: Image {imname} not found in the qimlist.")
        else:
            print(f"Image {imname} is at index {qimlist_idx} in the qimlist.")
    else:
        print(f"Image {imname} is at index {imlist_idx} in the imlist.")
    presence = find_presence_in_gnd(data, imname)
    print(f"Presence: {presence}")
    '''


if __name__ == '__main__':
    main()
