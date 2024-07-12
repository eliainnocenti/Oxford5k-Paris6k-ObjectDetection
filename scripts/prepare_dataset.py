"""
This script handles the loading, splitting, and preparation of image datasets for training, validation, and testing.

Functions:
----------
1. load_pickle(file_path):
    Loads and returns data from a pickle file.

2. split_train_val_test(dataset_name, train_percent=0.7, val_percent=0.2, test_percent=0.1):
    Splits the dataset into training, validation, and test sets based on the specified percentages.

3. split_annotations(dataset_name, type='json'):
    Splits the annotations into training, validation, and test sets based on the specified type.

4. prepare_dataset(dataset_name, type='xml', levels=3):
    Prepares the dataset by creating annotations and splitting it into training, validation, and test sets.

5. main():
    Main function to initiate the dataset preparation for specified datasets.

Dependencies:
-------------
- os
- random
- pickle
- json
- scripts.create_annotations (Ensure that the create_annotations script is accessible and correctly imported)

Usage:
------
To run this script, ensure that the required libraries are installed and the data directory is correctly set.
"""

import os
import random
import pickle
import json

from scripts.create_annotations import main as create_annotations
from scripts.create_annotations import create_json

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


def split_train_val_test(dataset_name, train_percent=0.7, val_percent=0.2, test_percent=0.1, type='json'):
    """
    Splits the dataset into training, validation, and test sets based on specified percentages.

    :param dataset_name: Name of the dataset.
    :param train_percent: Percentage of data for the training set.
    :param val_percent: Percentage of data for the validation set.
    :param test_percent: Percentage of data for the test set.
    :return: None
    """
    percent = train_percent + val_percent + test_percent
    if abs(1 - percent) > 0.000001:
        print("Error: Train, validation and test percentages must sum up to 1")
        return

    # check if txt files are already created
    if os.path.exists(os.path.join('../data', dataset_name, 'sets', 'train', 'train.txt')) or \
            os.path.exists(os.path.join('../data', dataset_name, 'sets', 'validation', 'val.txt')) or \
            os.path.exists(os.path.join('../data', dataset_name, 'sets', 'test', 'test.txt')):
        print(f"Train, validation and test sets already created for {dataset_name}")
        return

    if type == 'xml':
        annotations_dir = os.path.join('../data', dataset_name, 'annotations/xml')
        annotations = [file for file in os.listdir(annotations_dir) if file.endswith('.xml')]
        images = [os.path.splitext(file)[0] for file in annotations]

    elif type == 'json':
        annotations_dir = os.path.join('../data', dataset_name, 'annotations/json')
        labels_file = os.path.join(annotations_dir, 'labels.json')
        if not os.path.exists(labels_file):
            print(f"Error: Labels file not found: {labels_file}")
            return
        with open(labels_file, 'r') as file:
            labels_json = json.load(file)
        # insert in images only the images that have at least one annotation
        images = [image['file_name'] for image in labels_json['images']]
        for image in labels_json['images']:
            if image['id'] not in [annotation['image_id'] for annotation in labels_json['annotations']]:
                images.remove(image['file_name'])


    train_size = int(len(images) * train_percent)
    val_size = int(len(images) * val_percent)
    test_size = len(images) - train_size - val_size

    for i in range(random.randint(1, 10)):
        random.shuffle(images)

    train_set = images[:train_size]
    val_set = images[train_size:train_size + val_size]
    test_set = images[train_size + val_size:]

    sets_dir = os.path.join('../data', dataset_name, 'sets')

    os.makedirs(os.path.join(sets_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(sets_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(sets_dir, 'test'), exist_ok=True)

    with open(os.path.join('../data', dataset_name, 'sets', 'train', 'train.txt'), 'w') as file:
        for image in train_set:
            file.write(f'{image}\n')

    with open(os.path.join('../data', dataset_name, 'sets', 'validation', 'val.txt'), 'w') as file:
        for image in val_set:
            file.write(f'{image}\n')

    with open(os.path.join('../data', dataset_name, 'sets', 'test', 'test.txt'), 'w') as file:
        for image in test_set:
            file.write(f'{image}\n')


def split_annotations(dataset_name, type='json'):
    """
    Splits the annotations into training, validation, and test sets based on the specified type.

    :param dataset_name: Name of the dataset.
    :param type: Type of annotations ('json' or 'xml').
    :return: None
    """
    train_set = []
    val_set = []
    test_set = []

    sets_dir = os.path.join('../data', dataset_name, 'sets/')

    os.makedirs(os.path.join(sets_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(sets_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(sets_dir, 'test'), exist_ok=True)

    with open(os.path.join(sets_dir, 'train', 'train.txt'), 'r') as file:
        for line in file:
            train_set.append(line.strip())

    with open(os.path.join(sets_dir, 'validation', 'val.txt'), 'r') as file:
        for line in file:
            val_set.append(line.strip())

    with open(os.path.join(sets_dir, 'test', 'test.txt'), 'r') as file:
        for line in file:
            test_set.append(line.strip())

    if type == 'json':
        annotations_dir = os.path.join('../data', dataset_name, 'annotations/json')
        labels_file = os.path.join(annotations_dir, 'labels.json')
        if not os.path.exists(labels_file):
            print(f"Error: Labels file not found: {labels_file}")
            return
        with open(labels_file, 'r') as file:
            labels_json = json.load(file)
        categories = labels_json['categories']
        images = labels_json['images']
        annotations = labels_json['annotations']

        train_images = [image for image in images if image['file_name'] in train_set]
        val_images = [image for image in images if image['file_name'] in val_set]
        test_images = [image for image in images if image['file_name'] in test_set]

        print(f"Number of train images: {len(train_images)}")
        print(f"Number of val images: {len(val_images)}")
        print(f"Number of test images: {len(test_images)}")

        train_annotations = [annotation for annotation in annotations if annotation['image_id'] in [image['id'] for image in train_images]]
        val_annotations = [annotation for annotation in annotations if annotation['image_id'] in [image['id'] for image in val_images]]
        test_annotations = [annotation for annotation in annotations if annotation['image_id'] in [image['id'] for image in test_images]]

        create_json(categories, train_images, train_annotations, sets_dir + 'train', name='train')
        create_json(categories, val_images, val_annotations, sets_dir + 'validation', name='val')
        create_json(categories, test_images, test_annotations, sets_dir + 'test', name='test')

    else:
        print("Error: Invalid type of annotation")
        return


def prepare_dataset(dataset_name, type='xml', levels=1):
    """
    Prepares the dataset by creating annotations and splitting it into training, validation, and test sets.

    :param dataset_name: Name of the dataset.
    :param type: Type of annotations ('xml' or 'json').
    :param levels: Number of levels for annotation creation.
    :return: None
    """
    if type == 'xml':
        annotations_dir = os.path.join('../data', dataset_name, 'annotations/xml')
    elif type == 'json':
        annotations_dir = os.path.join('../data', dataset_name, 'annotations/json')
    else:
        print("Error: Invalid type of annotation")
        return

    # Check if annotations dir are already created
    if not os.path.exists(annotations_dir) or \
            (os.path.exists(annotations_dir) and type == 'xml' and len(os.listdir(annotations_dir)) == 0) or \
            (os.path.exists(annotations_dir) and type == 'json' and not any(
                file.endswith('.json') for file in os.listdir(annotations_dir))):
        print(f"Annotations not found for {dataset_name}. Do you want to create them? (y/n)")
        create = input()
        if create == 'y':
            create_annotations(dataset_name, type, levels=levels)
        else:
            print("Exiting...")
            return
    else:
        print(f"Annotations found for {dataset_name}")

    print(f"Do you want to split the dataset {dataset_name} into train, validation and test sets? (y/n)")
    split = input()
    if split == 'y':
        split_train_val_test(dataset_name)
        split_annotations(dataset_name, type=type)
    else:
        print("Exiting...")
        return


def main():
    """
    Main function to initiate the dataset preparation for specified datasets.

    :return: None
    """
    datasets = [
        #'roxford5k', # TODO: uncomment
        'rparis6k'
    ]

    for dataset in datasets:
        prepare_dataset(dataset, type='json', levels=1)


if __name__ == "__main__":
    main()
