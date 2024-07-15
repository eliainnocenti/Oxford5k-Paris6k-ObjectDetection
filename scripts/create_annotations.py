"""
This script creates annotations for the Oxford 5k and Paris 6k datasets in XML or JSON format.

Classes:
--------
- BoundingBox: Represents a bounding box with coordinates for the top-left (xmin, ymin) and bottom-right (xmax, ymax) corners.
- Object: Represents an object in an image, including its name, pose, truncation, difficulty, and bounding box.

Functions:
----------
1. load_pickle(file_path):
    Loads and returns data from a pickle file.

2. check_pickle_structure(data):
    Checks if the structure of the pickle file data is as expected.

3. map_query_to_monument(query_images, monuments):
    Maps query images to their respective monuments based on the monument names.

4. find_monument_by_query_number(query_number, monuments_dict):
    Finds the monument corresponding to a given query number.

5. get_dataset_size(image_folder):
    Gets the size of the dataset based on the number of images in the folder.

6. check_substring(full_string, prefix, target_substring):
    Checks if a target substring is present in a full string after a specified prefix.

7. create_xml(folder_name, image_name, width, height, objects, output_folder):
    Creates an XML annotation file for a given image.

8. create_json(categories, images, annotations, output_folder, name="labels"):
    Creates a JSON file for the annotations.

9. create_classes_list(data, folder_name):
    Creates a list of classes from the dataset.

10. get_id_by_name(categories, name):
    Gets the ID of a category by its name.

11. process_data(folder_name, data, image_folder, output_folder, monuments_list, type='xml', levels=2):
    Processes the dataset to create annotations in XML or JSON format.

12. main(datasets=None, type='xml', levels=2):
    Main function to create annotations for the specified datasets.

Dependencies:
-------------
- pickle
- xml.etree.ElementTree
- minidom
- os
- json
- PIL (Python Imaging Library)

Usage:
------
To run this script, ensure that the required libraries are installed and the data directory is correctly set.
"""

import pickle
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import json
from PIL import Image

# TODO: extract the same code from the process_data function and make it common for both types of annotations


class BoundingBox:
    """
    Represents a bounding box with coordinates for the top-left (xmin, ymin) and bottom-right (xmax, ymax) corners.
    """
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def __str__(self):
        return f"({self.xmin}, {self.ymin}, {self.xmax}, {self.ymax})"


class Object:
    """
    Represents an object in an image, including its name, pose, truncation, difficulty, and bounding box.
    """
    def __init__(self, name, pose, truncated, difficult, bounding_box=None):
        self.name = name
        self.pose = pose
        self.truncated = truncated
        self.difficult = difficult
        self.bounding_box = bounding_box

    def __str__(self):
        return f"{self.name} {self.pose} {self.truncated} {self.difficult} {self.bounding_box}"


def load_pickle(file_path):
    """
    Loads a pickle file from the given file path.

    :param file_path: Path to the pickle file.
    :return: Data loaded from the pickle file.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def check_pickle_structure(data):
    """
    Checks if the structure of the pickle file data is as expected.

    :param data: Data loaded from the pickle file.
    :return: True if structure is valid, False otherwise.
    """
    # Check if the structure of the pkl file is the expected one
    if 'gnd' in data and 'imlist' in data and 'qimlist' in data:
        return True
    return False


def map_query_to_monument(query_images, monuments):
    """
    Maps query images to their respective monuments based on the monument names.

    :param query_images: Dictionary of query images.
    :param monuments: List of monuments.
    :return: Dictionary mapping each monument to the list of query image indices.
    """
    query_to_monument = {}
    for monument in monuments:
        query_to_monument[monument] = []
        for idx, query_image in query_images.items():    # Iterate over items of the dictionary
            if monument in query_image:                  # Check if monument is a substring of the query image
                query_to_monument[monument].append(idx)  # Append the index of the query image to the list
    return query_to_monument


def find_monument_by_query_number(query_number, monuments_dict):
    """
    Finds the monument corresponding to a given query number.

    :param query_number: Query number.
    :param monuments_dict: Dictionary mapping monuments to query image indices.
    :return: Monument name or "Query number not found".
    """
    for monument, queries in monuments_dict.items():
        if query_number in queries:
            return monument
    return "Query number not found"


def get_dataset_size(image_folder):
    return len(os.listdir(image_folder))


def check_substring(full_string, prefix, target_substring): # TODO: check usage
    """

    :param full_string:
    :param prefix:
    :param target_substring:
    :return:
    """
    if full_string.startswith(prefix):
        substring_after_prefix = full_string[len(prefix):]
        return substring_after_prefix.startswith(target_substring)
    return False


def create_xml(folder_name, image_name, width, height, objects, output_folder):
    """
    Creates an XML annotation file for a given image.

    :param folder_name: Name of the folder containing the images.
    :param image_name: Name of the image file.
    :param width: Width of the image.
    :param height: Height of the image.
    :param objects: List of objects in the image.
    :param output_folder: Folder to save the XML file.
    :return: None
    """
    root = ET.Element("annotation")

    ET.SubElement(root, "folder").text = folder_name
    ET.SubElement(root, "filename").text = image_name

    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    if folder_name == "rparis6k":
        database.text = "Paris 6k"
    elif folder_name == "roxford5k":
        database.text = "Oxford 5k"

    # Owner field omitted

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"  # Assuming all images are RGB

    ET.SubElement(root, "segmented").text = "0"

    for object in objects:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = object.name
        ET.SubElement(obj, "pose").text = object.pose
        ET.SubElement(obj, "truncated").text = object.truncated
        ET.SubElement(obj, "difficult").text = object.difficult
        bbox = object.bounding_box
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(bbox.xmin)
        ET.SubElement(bndbox, "ymin").text = str(bbox.ymin)
        ET.SubElement(bndbox, "xmax").text = str(bbox.xmax)
        ET.SubElement(bndbox, "ymax").text = str(bbox.ymax)

    xml_string = ET.tostring(root)
    xml_pretty = minidom.parseString(xml_string).toprettyxml(indent="  ")

    image_name = image_name.split(".")[0]
    with open(os.path.join(output_folder, f"{image_name}.xml"), "w") as f:
        f.write(xml_pretty)


def create_json(categories, images, annotations, output_folder, name="labels"):
    """
    Creates a JSON file for the annotations.

    :param categories: List of categories for the annotations.
    :param images: List of images with their IDs and filenames.
    :param annotations: List of annotations with their bounding boxes.
    :param output_folder: Folder to save the JSON file.
    :param name: Name of the JSON file (default is "labels").
    :return: None
    """
    data = {
        "categories": categories,
        "images": images,
        "annotations": annotations
    }

    with open(os.path.join(output_folder, f"{name}.json"), "w") as f:
        json.dump(data, f, indent=4) # TODO: check if the format is correct


def create_classes_list(data, folder_name):  # FIXME: update function
    """
    Creates a list of classes from the dataset.

    :param data: Data loaded from the pickle file.
    :param folder_name: Name of the folder containing the dataset.
    :return: None
    """
    # TODO: check if the file already exists
    classes = {}
    for idx, query_image in enumerate(data['qimlist']):
        if idx not in classes:
            classes[idx] = []
            classes[idx].append(query_image)
    classes_file_path = os.path.join('../data/', folder_name, 'sets', 'classes.txt')
    with open(classes_file_path, 'w') as f:
        for key in classes.keys():
            # Assuming each key in classes has a list with at least one element
            f.write(f"{key}: {classes[key][0]}\n")


def get_id_by_name(categories, name):
    """
    Gets the ID of a category by its name.

    :param categories: List of categories.
    :param name: Name of the category.
    :return: ID of the category or None if not found.
    """
    for category in categories:
        if category['name'] == name:
            return category['id']
    return None


def convert_bbox(bbox):
    """
    Converts the bounding box from (xmin, ymin, xmax, ymax) to (xmin, ymin, width, height) format.

    :param bbox: Bounding box in (xmin, ymin, xmax, ymax) format.
    :return: Bounding box in (xmin, ymin, width, height) format.
    """
    xmin, ymin, xmax, ymax = bbox
    width = round(xmax - xmin, 1)
    height = round(ymax - ymin, 1)

    return [xmin, ymin, width, height]


def _process_data_xml(folder_name, data, image_folder, output_folder, monuments_list, levels=2):
    """
    Processes the dataset to create annotations in XML format.

    :param folder_name: Name of the folder containing the dataset.
    :param data: Data loaded from the pickle file.
    :param image_folder: Folder containing the images.
    :param output_folder: Folder to save the annotations.
    :param monuments_list: List of monuments to annotate.
    :param levels: Number of difficulty levels to consider (1, 2, or 3).
    :return: None
    """
    # FIXME: fix function

    gnd = data['gnd']
    imlist = data['imlist']
    qimlist = data['qimlist']

    if levels == 3:
        levels = ["easy", "hard", "junk"]
    elif levels == 2:
        levels = ["easy", "hard"]
    elif levels == 1:
        levels = ["easy"]
    else:
        print("Invalid number of levels")
        return

    # Create a xml file for each image in imlist and qimlist
    # In each xml file, there must be at least one bounding box # TODO: check
    # If the image is a query image, the bounding box must be the related query bounding box
    # Otherwise, there must be one bounding box for each presence of the image in the gnd dictionary

    query_images = {idx: image_name for idx, image_name in enumerate(data['qimlist'])}

    monuments_dict = map_query_to_monument(query_images, monuments_list)

    # query images
    for idx, query_image in enumerate(qimlist):
        image_name = query_image + ".jpg"
        image_path = os.path.join(image_folder, image_name)
        if not os.path.exists(image_path):  # TODO: check
            print(f"Warning: Image {image_path} not found. Skipping.")
            continue
        with Image.open(image_path) as img:
            width, height = img.size
        _bbox = gnd[idx]['bbx']
        bbox = BoundingBox(_bbox[0], _bbox[1], _bbox[2], _bbox[3])
        monument = find_monument_by_query_number(idx, monuments_dict)
        objects = [Object(f"{monument}", "Unspecified", "0", "0", bbox)]  # TODO: check attributes
        create_xml(folder_name, image_name, width, height, objects, output_folder)

    # imlist images
    for idx, image in enumerate(imlist):
        image_name = image + ".jpg"
        image_path = os.path.join(image_folder, image_name)
        if not os.path.exists(image_path):  # TODO: check
            print(f"Warning: Image {image_path} not found. Skipping.")
            continue
        with Image.open(image_path) as img:
            width, height = img.size
        # find all the bboxes related to the image
        presences = []  # TODO: update name
        for query_idx in range(len(qimlist)):
            for level in levels:
                if idx in gnd[query_idx][level]:
                    presences.append((query_idx, level))
        objects = []
        _objects = {}
        for presence in presences:
            query_idx, level = presence
            _bbox = gnd[query_idx]['bbx']
            bbox = BoundingBox(_bbox[0], _bbox[1], _bbox[2], _bbox[3])
            if level == "easy":
                level = 0
            elif level == "hard":
                level = 1
            elif level == "junk":
                level = 2
            else:
                level = 0
            monument = find_monument_by_query_number(query_idx, monuments_dict)
            if monument not in _objects:
                _objects[monument] = []
            _objects[monument].append(Object(f"{monument}", "Unspecified", "0", str(level), bbox))
        # merge bbox for the same monument
        for monument in _objects.keys():
            xmin_avg, ymin_avg, xmax_avg, ymax_avg = 0, 0, 0, 0
            difficulty = 0
            for obj in _objects[monument]:
                if obj.difficult == "1":
                    difficulty = 1
                xmin_avg += obj.bounding_box.xmin
                ymin_avg += obj.bounding_box.ymin
                xmax_avg += obj.bounding_box.xmax
                ymax_avg += obj.bounding_box.ymax
            xmin_avg = round(xmin_avg / len(_objects[monument]), 1)
            ymin_avg = round(ymin_avg / len(_objects[monument]), 1)
            xmax_avg = round(xmax_avg / len(_objects[monument]), 1)
            ymax_avg = round(ymax_avg / len(_objects[monument]), 1)
            bbox = BoundingBox(xmin_avg, ymin_avg, xmax_avg, ymax_avg)
            objects.append(Object(f"{monument}", "Unspecified", "0", str(difficulty), bbox))
        create_xml(folder_name, image_name, width, height, objects, output_folder)


def _process_data_json(data, image_folder, output_folder, monuments_list, levels=2):
    """
    Processes the dataset to create annotations in JSON format.

    :param data: Data loaded from the pickle file.
    :param image_folder: Folder containing the images.
    :param output_folder: Folder to save the annotations.
    :param monuments_list: List of monuments to annotate.
    :param levels: Number of difficulty levels to consider (1, 2, or 3).
    :return: None
    """
    gnd = data['gnd']
    imlist = data['imlist']
    qimlist = data['qimlist']

    if levels == 3:
        levels = ["easy", "hard", "junk"]
    elif levels == 2:
        levels = ["easy", "hard"]
    elif levels == 1:
        levels = ["easy"]
    else:
        print("Invalid number of levels")
        return

    query_images = {idx: image_name for idx, image_name in enumerate(data['qimlist'])}

    monuments_dict = map_query_to_monument(query_images, monuments_list)

    # Creating JSON categories field
    categories = []
    i = 1
    categories.append({"id": 0, "name": "background"})
    for monument in monuments_list:
        categories.append({"id": i, "name": monument})
        i += 1

    # Creating JSON images field
    images = []
    for idx, query_image in enumerate(qimlist):
        image_name = query_image + ".jpg"
        images.append({"id": idx, "file_name": image_name})
    offset = len(images)
    for idx, image in enumerate(imlist):
        image_name = image + ".jpg"
        images.append({"id": idx + offset, "file_name": image_name})

    # Creating JSON annotations field
    annotations = []

    # query images
    query_images_objects = {}
    for idx, query_image in enumerate(qimlist):
        image_name = query_image + ".jpg"
        image_path = os.path.join(image_folder, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found. Skipping.")
            continue
        with Image.open(image_path) as img:
            width, height = img.size
        _bbox = gnd[idx]['bbx']
        _bbox = convert_bbox(_bbox) # TODO: check
        bbox = BoundingBox(_bbox[0], _bbox[1], _bbox[2], _bbox[3])
        monument = find_monument_by_query_number(idx, monuments_dict)
        query_images_objects[idx] = []
        query_images_objects[idx].append(Object(f"{monument}", "Unspecified", "0", "0", bbox))

    i = 0
    for idx in query_images_objects.keys():
        for obj in query_images_objects[idx]:
            category_id = get_id_by_name(categories, obj.name)
            if category_id is None:
                print("Error: Category not found")
                continue
            annotations.append({
                "id": i,
                "image_id": idx,
                "category_id": category_id,
                "bbox": [obj.bounding_box.xmin, obj.bounding_box.ymin, obj.bounding_box.xmax, obj.bounding_box.ymax]
            })
            i += 1

    # imlist images
    other_images_objects = {}
    for idx, image in enumerate(imlist):
        image_name = image + ".jpg"
        image_path = os.path.join(image_folder, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found. Skipping.")
            continue
        with Image.open(image_path) as img:
            width, height = img.size

        # find all the bboxes related to the image
        presences = []  # TODO: update name
        for query_idx in range(len(qimlist)):
            for level in levels:
                if idx in gnd[query_idx][level]:
                    presences.append((query_idx, level))

        objects = []
        _objects = {}
        other_images_objects[idx] = []

        for presence in presences:
            query_idx, level = presence
            _bbox = gnd[query_idx]['bbx']
            bbox = BoundingBox(_bbox[0], _bbox[1], _bbox[2], _bbox[3])
            if level == "easy":
                level = 0
            elif level == "hard":
                level = 1
            elif level == "junk":
                level = 2
            else:
                level = 0
            monument = find_monument_by_query_number(query_idx, monuments_dict)
            if monument not in _objects:
                _objects[monument] = []
            _objects[monument].append(Object(f"{monument}", "Unspecified", "0", str(level), bbox))

        # merge bbox for the same monument # FIXME: choose another way to merge (?)
        for monument in _objects.keys():
            xmin_avg, ymin_avg, xmax_avg, ymax_avg = 0, 0, 0, 0
            difficulty = 0
            for obj in _objects[monument]:
                if obj.difficult == "1":
                    difficulty = 1
                xmin_avg += obj.bounding_box.xmin
                ymin_avg += obj.bounding_box.ymin
                xmax_avg += obj.bounding_box.xmax
                ymax_avg += obj.bounding_box.ymax
            xmin_avg = round(xmin_avg / len(_objects[monument]), 1)
            ymin_avg = round(ymin_avg / len(_objects[monument]), 1)
            xmax_avg = round(xmax_avg / len(_objects[monument]), 1)
            ymax_avg = round(ymax_avg / len(_objects[monument]), 1)
            _bbox = [xmin_avg, ymin_avg, xmax_avg, ymax_avg]
            _bbox = convert_bbox(_bbox)  # TODO: check
            bbox = BoundingBox(_bbox[0], _bbox[1], _bbox[2], _bbox[3])
            objects.append(Object(f"{monument}", "Unspecified", "0", str(difficulty), bbox))
        other_images_objects[idx] = objects

    i = len(annotations) # TODO: check
    offset = len(qimlist)
    for idx in other_images_objects.keys():
        for obj in other_images_objects[idx]:
            category_id = get_id_by_name(categories, obj.name)
            if category_id is None:
                print("Error: Category not found")
                continue
            annotations.append({
                "id": i,
                "image_id": idx + offset,
                "category_id": category_id,
                "bbox": [obj.bounding_box.xmin, obj.bounding_box.ymin, obj.bounding_box.xmax, obj.bounding_box.ymax]
            })
            i += 1

    create_json(categories, images, annotations, output_folder)


def process_data(folder_name, data, image_folder, output_folder, monuments_list, type='json', levels=2):
    """
    Processes the dataset to create annotations in XML or JSON format.
    :param folder_name: Name of the folder containing the dataset.
    :param data: Data loaded from the pickle file.
    :param image_folder: Folder containing the images.
    :param output_folder: Folder to save the annotations.
    :param monuments_list: List of monuments to annotate.
    :param type: Type of annotation file ('xml' or 'json').
    :param levels: Number of difficulty levels to consider (1, 2, or 3).
    :return: None
    """
    if not check_pickle_structure(data):
        print("Invalid pickle file structure")
        return

    if type == "xml":

        output_folder = os.path.join(output_folder, "xml")

        os.makedirs(output_folder, exist_ok=True)

        _process_data_xml(folder_name, data, image_folder, output_folder, monuments_list, levels)

        # create_classes_list(data, folder_name)  # FIXME: do i need the file classes.txt?

        print(f"XML annotations created in: {output_folder}")
        print(f"Dataset size: {get_dataset_size(image_folder)}")  # number of images in the dataset
        print(f"Annotations created: {len(os.listdir(output_folder))}")  # number of annotations created


    elif type == "json":

        output_folder = os.path.join(output_folder, "json")

        os.makedirs(output_folder, exist_ok=True)

        _process_data_json(data, image_folder, output_folder, monuments_list, levels)

        print(f"JSON annotations created in: {output_folder}")

    print("Annotations created successfully")


def main(datasets=None, type='json', levels=1):
    """
    Main function to create annotations for the specified datasets.

    :param datasets: List of datasets to process (default is None).
    :param type: Type of annotation file ('xml' or 'json').
    :param levels: Number of difficulty levels to consider (1, 2, or 3).
    :return: None
    """
    base_path = "../../../Data/"

    if datasets is None:
        datasets = [
            #"roxford5k", # TODO: uncomment
            "rparis6k"
        ]

    # Oxford 5k dataset
    if "roxford5k" in datasets:
        pickle_file = os.path.join(base_path, "datasets/roxford5k/gnd_roxford5k.pkl")
        image_folder = os.path.join(base_path, "datasets/roxford5k/images")
        output_folder = "../data/roxford5k/annotations/"
        monuments_roxford5k = []  # TODO: add monuments

        data = load_pickle(pickle_file)
        process_data("roxford5k", data, image_folder, output_folder, monuments_roxford5k, type, levels)

    # Paris 6k dataset
    if "rparis6k" in datasets:
        pickle_file = os.path.join(base_path, "datasets/rparis6k/gnd_rparis6k.pkl")
        image_folder = os.path.join(base_path, "datasets/rparis6k/images")
        output_folder = "../data/rparis6k/annotations/"
        monuments_rparis6k = ['defense', 'eiffel', 'invalides', 'louvre', 'moulinrouge', 'museedorsay',
                              'notredame', 'pantheon', 'pompidou', 'sacrecoeur', 'triomphe']
        # TODO: handle 'general' query

        data = load_pickle(pickle_file)
        process_data("rparis6k", data, image_folder, output_folder, monuments_rparis6k, type, levels)


if __name__ == "__main__":
    main(type='json', levels=1)
