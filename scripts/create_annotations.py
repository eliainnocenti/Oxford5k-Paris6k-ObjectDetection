"""
TODO: add description
"""

import pickle
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from PIL import Image

# TODO: check -> assuming that an image can appear only once related to a specific query image, it may be possible to use a single bounding box per object
# TODO: check if i need to use the add_box method
# FIXME: update all the paths

class BoundingBox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


class Object:
    def __init__(self, name, pose, truncated, difficult, bounding_boxes=None):
        self.name = name
        self.pose = pose
        self.truncated = truncated
        self.difficult = difficult
        self.bounding_boxes = bounding_boxes if bounding_boxes is not None else []

    def add_bbox(self, bbox):
        self.bounding_boxes.append(bbox)


def map_query_to_monument(query_images, monuments):
    query_to_monument = {}
    for monument in monuments:
        query_to_monument[monument] = []
        for idx, query_image in query_images.items():     # Iterate over items of the dictionary
            if monument in query_image:                   # Check if monument is a substring of the query image
                query_to_monument[monument].append(idx)   # Append the index of the query image to the list
    return query_to_monument


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def check_pickle_structure(data):
    # Check if the structure of the pkl file is the expected one
    if 'gnd' in data and 'imlist' in data and 'qimlist' in data:
        return True
    return False


def get_dataset_size(image_folder):
    return len(os.listdir(image_folder))


def check_substring(full_string, prefix, target_substring):
    if full_string.startswith(prefix):
        substring_after_prefix = full_string[len(prefix):]
        return substring_after_prefix.startswith(target_substring)
    return False


def create_xml(folder_name, image_name, width, height, objects, output_folder):
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
    ET.SubElement(size, "depth").text = "3" # Assuming all images are RGB

    ET.SubElement(root, "segmented").text = "0"

    for object in objects:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = object.name
        ET.SubElement(obj, "pose").text = object.pose
        ET.SubElement(obj, "truncated").text = object.truncated
        ET.SubElement(obj, "difficult").text = object.difficult
        for bbox in object.bounding_boxes:
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


def create_classes_list(data, folder_name): # FIXME: update function
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


def find_monument_by_query_number(query_number, monuments_dict):
    for monument, queries in monuments_dict.items():
        if query_number in queries:
            return monument
    return "Query number not found"


def process_data(folder_name, data, image_folder, output_folder, monuments_list, type='xml', levels=3):

    if type == "xml":

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
        # In each xml file, there must be at least one bounding box
        # If the image is a query image, the bounding box must be the related query bounding box
        # Otherwise, there must be one bounding box for each presence of the image in the gnd dictionary

        query_images = {idx: image_name for idx, image_name in enumerate(data['qimlist'])}

        monuments_dict = map_query_to_monument(query_images, monuments_list)

        for idx, query_image in enumerate(qimlist):
            image_name = query_image + ".jpg"
            image_path = os.path.join(image_folder, image_name)
            if not os.path.exists(image_path): # TODO: check
                print(f"Warning: Image {image_path} not found. Skipping.")
                continue
            with Image.open(image_path) as img:
                width, height = img.size
            _bbox = gnd[idx]['bbx']
            bbox = BoundingBox(_bbox[0], _bbox[1], _bbox[2], _bbox[3])
            monument = find_monument_by_query_number(idx, monuments_dict)
            objects = [Object(f"{monument}", "Unspecified", "0", "0", [bbox])] # TODO: check attributes
            create_xml(folder_name, image_name, width, height, objects, output_folder)

        for idx, image in enumerate(imlist):
            image_name = image + ".jpg"
            image_path = os.path.join(image_folder, image_name)
            if not os.path.exists(image_path): # TODO: check
                print(f"Warning: Image {image_path} not found. Skipping.")
                continue
            with Image.open(image_path) as img:
                width, height = img.size
            presences = [] # TODO: update name
            for query_idx in range(len(qimlist)):
                for level in levels:
                    if idx in gnd[query_idx][level]:
                        presences.append((query_idx, level))
            objects = []
            for presence in presences:
                query_idx, level = presence
                _bbox = gnd[query_idx]['bbx']
                bbox = BoundingBox(_bbox[0], _bbox[1], _bbox[2], _bbox[3])
                if level == "easy": level = 0
                elif level == "hard": level = 1
                elif level == "junk": level = 2
                else: level = 0
                # FIXME: check if there is the same object and use the method add_bbox
                monument = find_monument_by_query_number(query_idx, monuments_dict)
                objects.append(Object(f"{monument}", "Unspecified", "0", str(level), [bbox])) # TODO: check attributes
            create_xml(folder_name, image_name, width, height, objects, output_folder)

        create_classes_list(data, folder_name) # FIXME: do i need the file classes.txt?

        print(f"XML annotations created in: {output_folder}")
        print(f"Dataset size: {get_dataset_size(image_folder)}")         # number of images in the dataset
        print(f"Annotations created: {len(os.listdir(output_folder))}")  # number of annotations created

    # TODO: add support for other types of annotations (csv, json)

    print("Annotations created successfully")


def main(datasets=None, type='xml',levels=3):

    base_path = "../../../Data/"

    if datasets is None:
        datasets = [
            #"roxford5k", # TODO: uncomment
            "rparis6k"
        ]

    if "roxford5k" in datasets:
        # Oxford 5k dataset
        pickle_file = os.path.join(base_path, "datasets/roxford5k/gnd_roxford5k.pkl")
        image_folder = os.path.join(base_path, "datasets/roxford5k/images")
        output_folder = "../data/roxford5k/annotations/xml"
        monuments_roxford5k = [] # TODO: add monuments

        data = load_pickle(pickle_file)
        process_data("roxford5k", data, image_folder, output_folder, monuments_roxford5k, type, levels)

    if "rparis6k" in datasets:
        # Paris 6k dataset
        pickle_file = os.path.join(base_path, "datasets/rparis6k/gnd_rparis6k.pkl")
        image_folder = os.path.join(base_path, "datasets/rparis6k/images")
        output_folder = "../data/rparis6k/annotations/xml"
        monuments_rparis6k = ['defense', 'eiffel', 'general', 'invalides', 'louvre', 'moulinrouge', 'museedorsay',
                              'notredame', 'pantheon', 'pompidou', 'sacrecoeur', 'triomphe']

        data = load_pickle(pickle_file)
        process_data("rparis6k", data, image_folder, output_folder, monuments_rparis6k, type, levels)


if __name__ == "__main__":
    main()
