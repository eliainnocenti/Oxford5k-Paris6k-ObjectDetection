"""
TODO: add file and function descriptions
"""

import pickle
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import json
from PIL import Image

# TODO: extract the same code from the process_data function and make it common for both types of annotations


class BoundingBox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def __str__(self):
        return f"({self.xmin}, {self.ymin}, {self.xmax}, {self.ymax})"


class Object:
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

    :param file_path:
    :return:
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def check_pickle_structure(data):
    """

    :param data:
    :return:
    """
    # Check if the structure of the pkl file is the expected one
    if 'gnd' in data and 'imlist' in data and 'qimlist' in data:
        return True
    return False


def map_query_to_monument(query_images, monuments):
    """

    :param query_images:
    :param monuments:
    :return:
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

    :param query_number:
    :param monuments_dict:
    :return:
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

    :param folder_name:
    :param image_name:
    :param width:
    :param height:
    :param objects:
    :param output_folder:
    :return:
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

    :param categories:
    :param images:
    :param annotations:
    :param output_folder:
    :param name:
    :return:
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

    :param data:
    :param folder_name:
    :return:
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
    for category in categories:
        if category['name'] == name:
            return category['id']
    return None


def process_data(folder_name, data, image_folder, output_folder, monuments_list, type='xml', levels=2):
    """

    :param folder_name:
    :param data:
    :param image_folder:
    :param output_folder:
    :param monuments_list:
    :param type:
    :param levels:
    :return:
    """
    if type == "xml":

        output_folder = os.path.join(output_folder, "xml")

        if not check_pickle_structure(data):
            print("Invalid pickle file structure")
            return

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

        os.makedirs(output_folder, exist_ok=True)

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

        #create_classes_list(data, folder_name)  # FIXME: do i need the file classes.txt?

        print(f"XML annotations created in: {output_folder}")
        print(f"Dataset size: {get_dataset_size(image_folder)}")  # number of images in the dataset
        print(f"Annotations created: {len(os.listdir(output_folder))}")  # number of annotations created

    elif type == "json":

        output_folder = os.path.join(output_folder, "json")

        if not check_pickle_structure(data):
            print("Invalid pickle file structure")
            return

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

        os.makedirs(output_folder, exist_ok=True)

        query_images = {idx: image_name for idx, image_name in enumerate(data['qimlist'])}

        monuments_dict = map_query_to_monument(query_images, monuments_list)


        categories = []
        i = 1
        for monument in monuments_list:
            categories.append({"id": i, "name": monument})
            i += 1

        images = []
        for idx, query_image in enumerate(qimlist):
            image_name = query_image + ".jpg"
            images.append({"id": idx, "file_name": image_name})
        offset = len(images)
        for idx, image in enumerate(imlist):
            image_name = image + ".jpg"
            images.append({"id": idx + offset, "file_name": image_name}) # TODO: check

        annotations = []

        # query images
        query_images_objects = {}
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
            query_images_objects[idx] = []
            query_images_objects[idx].append(Object(f"{monument}", "Unspecified", "0", "0", bbox))

        # imlist images
        other_images_objects = {}
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
            other_images_objects[idx] = objects

        i = 0
        for idx in query_images_objects.keys():
            for obj in query_images_objects[idx]:
                category_id = get_id_by_name(categories, obj.name)
                if category_id is None:
                    print("Error: Category not found")
                    return
                annotations.append({
                    "id": 0,
                    "image_id": idx,
                    "category_id": category_id,
                    "bbox": [obj.bounding_box.xmin, obj.bounding_box.ymin, obj.bounding_box.xmax, obj.bounding_box.ymax]
                })
                i += 1

        for idx in other_images_objects.keys():
            for obj in other_images_objects[idx]:
                category_id = get_id_by_name(categories, obj.name)
                if category_id is None:
                    print("Error: Category not found")
                    return
                annotations.append({
                    "id": i,
                    "image_id": idx,
                    "category_id": category_id,
                    "bbox": [obj.bounding_box.xmin, obj.bounding_box.ymin, obj.bounding_box.xmax, obj.bounding_box.ymax]
                })
                i += 1

        create_json(categories, images, annotations, output_folder)

        print(f"JSON annotations created in: {output_folder}")

    print("Annotations created successfully")


def main(datasets=None, type='xml', levels=2):
    """

    :param datasets:
    :param type:
    :param levels:
    :return:
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
        monuments_rparis6k = ['defense', 'eiffel', 'general', 'invalides', 'louvre', 'moulinrouge', 'museedorsay',
                              'notredame', 'pantheon', 'pompidou', 'sacrecoeur', 'triomphe']

        data = load_pickle(pickle_file)
        process_data("rparis6k", data, image_folder, output_folder, monuments_rparis6k, type, levels)


if __name__ == "__main__":
    main(type='json')
