"""
TODO: add description
"""

import os
import random
import pickle

from scripts.create_annotations import main as create_annotations


base_path = "../../../Data/"

# FIXME: check the usage of the classes.txt file

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def generate_txt_files(dataset_name, train_percent=0.7, val_percent=0.2, test_percent=0.1, levels=3):
    percent = train_percent + val_percent + test_percent
    if abs(1 - percent) > 0.000001:
        print("Error: Train, validation and test percentages must sum up to 1")
        return

    images_dir = os.path.join(base_path, 'datasets', dataset_name, 'images')
    annotations_dir = os.path.join('../data/', dataset_name, 'annotations/xml')
    classes_file = os.path.join('../data/', dataset_name, 'sets', 'classes.txt')
    pickle_file = os.path.join(base_path, 'datasets', dataset_name, f'gnd_{dataset_name}.pkl')

    train_set = []
    val_set = []
    test_set = []

    classes = {}
    data = load_pickle(pickle_file)

    with open(classes_file, 'r') as file:
        for line in file:
            number, string = line.split(':')
            number = int(number)
            classes[number] = string.strip()

    images = []
    images_idx = []

    for idx in range(len(data['gnd'])):
        if levels == 3:
            images_idx += data['gnd'][idx]['easy']
            images_idx += data['gnd'][idx]['hard']
            images_idx += data['gnd'][idx]['junk']
        elif levels == 2:
            images_idx += data['gnd'][idx]['easy']
            images_idx += data['gnd'][idx]['hard']
        elif levels == 1:
            images_idx += data['gnd'][idx]['easy']
        else:
            print("Error: Invalid number of levels")
            return

    random.shuffle(images_idx)

    # remove duplicates
    images_idx = list(set(images_idx))

    #print(f"Number of images_idx: {len(images_idx)}")

    for image_idx in images_idx:
        image = data['imlist'][image_idx]
        images.append(image)

    #print(f"Number of images: {len(images)}")

    train_size = int(len(images) * train_percent)
    val_size = int(len(images) * val_percent)
    test_size = len(images) - train_size - val_size

    train_set += images[:train_size]
    val_set += images[train_size:train_size + val_size]
    test_set += images[train_size + val_size:]

    with open(os.path.join(base_path, dataset_name, 'train.txt'), 'w') as file:
        for image in train_set:
            #image_idx = data['imlist'].index(image) # TODO: do i need this?
            file.write(f'{image}\n')

    with open(os.path.join(base_path, dataset_name, 'val.txt'), 'w') as file:
        for image in val_set:
            #image_idx = data['imlist'].index(image) # TODO: do i need this?
            file.write(f'{image}\n')

    with open(os.path.join(base_path, dataset_name, 'test.txt'), 'w') as file:
        for image in test_set:
            #image_idx = data['imlist'].index(image) # TODO: do i need this?
            file.write(f'{image}\n')

    with open(os.path.join(base_path, dataset_name, 'trainval.txt'), 'w') as file:
        images = train_set + val_set
        random.shuffle(images)
        for image in images:
            #image_idx = data['imlist'].index(image) # TODO: do i need this?
            file.write(f'{image}\n')


def prepare_dataset(dataset_name, type='xml', levels=3):
    # Check if annotations dir are already created
    if type == 'xml':
        annotations_dir = os.path.join('../data', dataset_name, 'annotations/xml')
    elif type == 'csv':
        annotations_dir = os.path.join('../data', dataset_name, 'annotations/csv') # TODO: check
    else:
        print("Error: Invalid type of annotation")
        return
    if not os.path.exists(annotations_dir):
        print(f"Annotations not found for {dataset_name}. Do you want to create them? (y/n)")
        create = input()
        if create == 'y':
            create_annotations(dataset_name, levels=levels)
        else:
            print("Exiting...")
            return

    # Create txt files
    generate_txt_files(dataset_name, levels=levels)


def main():
    datasets = [
        #'roxford5k',
        'rparis6k'
    ]

    for dataset in datasets:
        prepare_dataset(dataset)


if __name__ == "__main__":
    main()
