"""
TODO: add file and function descriptions
"""

import albumentations as A
import cv2
import json
import os
import shutil
from tqdm import tqdm


def get_transform(set='train'):
    """

    :param set:
    :return:
    """
    bboxes_params = A.BboxParams(format='coco', min_visibility=0.3, label_fields=['class_labels']) # TODO: check min_visibility

    if set == 'train':
        transform = A.Compose([ # TODO: update pipeline (?)
            # TODO: do I need to resize images?
            #A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0), # TODO: check h,w
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=0.5),
                A.MedianBlur(blur_limit=7, p=0.5),
                A.GaussianBlur(blur_limit=7, p=0.5),
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, border_mode=0, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # TODO: check
        ], bbox_params=bboxes_params)

    elif set == 'validation':
        transform = A.Compose([ # TODO: update pipeline
            # TODO: do I need to resize images?
            A.HorizontalFlip(p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # TODO: check
        ], bbox_params=bboxes_params)

    return transform


def clip_bbox(bbox, image_width, image_height):
    """

    :param bbox:
    :param image_width:
    :param image_height:
    :return:
    """
    x_min, y_min, width, height = bbox

    x_min = max(0, min(x_min, image_width - 1)) # TODO: check -1
    y_min = max(0, min(y_min, image_height - 1)) # TODO: check -1
    width = min(width, image_width - x_min)
    height = min(height, image_height - y_min)

    return [x_min, y_min, width, height]


def validate_bbox(bbox, image_width, image_height):
    """

    :param bbox:
    :param image_width:
    :param image_height:
    :return:
    """
    x, y, w, h = bbox

    return 0 <= x < image_width and 0 <= y < image_height and x + w <= image_width and y + h <= image_height


def apply_augmentation(image_path, bboxes, class_labels, output_path, output_filename, transform):
    """

    :param image_path:
    :param bboxes:
    :param class_labels:
    :param output_path:
    :param output_filename:
    :param transform:
    :return:
    """
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width = image.shape[:2]

    # Apply the augmentation
    try:
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    except Exception as e:
        print(f"Error during transformation: {e}")
        return [], []

    # Save the augmented image
    augmented_image_path = os.path.join(output_path, output_filename)
    cv2.imwrite(augmented_image_path, cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))

    return transformed['bboxes'], transformed['class_labels']


def augment_dataset(input_path, output_path, transform, n_images, n_annotations, num_augmentations=5):
    """

    :param input_path:
    :param output_path:
    :param transform:
    :param n_images:
    :param n_annotations:
    :param num_augmentations:
    :return:
    """
    # Load the original COCO JSON file
    with open(os.path.join(input_path, 'labels.json'), 'r') as f:
        coco_data = json.load(f)

    new_images = []
    new_annotations = []

    # Copy original images and annotations
    for img in tqdm(coco_data['images'], desc="Copying original images"):

        src_path = os.path.join(input_path, 'images', img['file_name'])
        dst_path = os.path.join(output_path, 'images', img['file_name'])
        shutil.copy2(src_path, dst_path)

        new_images.append(img)
        img_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img['id']]
        new_annotations.extend(img_anns)

    '''debug'''
    print("Before augmentation:")
    print(f"Number of images: {len(new_images)}")
    print(f"Number of annotations: {len(new_annotations)}")

    # Apply augmentations
    for img in tqdm(coco_data['images'], desc="Augmenting images"):
        image_path = os.path.join(input_path, 'images', img['file_name'])

        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img['id']]

        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]

        for i in range(num_augmentations):
            bboxes = [ann['bbox'] for ann in annotations]
            class_labels = [ann['category_id'] for ann in annotations]

            # TODO: should I call the function clip_bbox() regardless of the function validate_bbox()?
            for bbox in bboxes:
                if not validate_bbox(bbox, image_width, image_height):
                    bboxes = [clip_bbox(bbox, image_width, image_height) for bbox in bboxes]

            new_filename = f"{os.path.splitext(img['file_name'])[0]}_aug_{i}.jpg"

            new_bboxes, new_class_labels = apply_augmentation(
                image_path, bboxes, class_labels,
                os.path.join(output_path, 'images'), new_filename, transform
            )

            new_img_id = n_images + 1
            new_images.append({
                'id': new_img_id,
                'file_name': new_filename
            })

            n_images = n_images + 1

            for bbox, cat_id in zip(new_bboxes, new_class_labels):
                new_annotations.append({
                    'id': n_annotations + 1,
                    'image_id': new_img_id,
                    'category_id': cat_id,
                    'bbox': bbox
                })

                n_annotations = n_annotations + 1

    '''debug'''
    print("After augmentation:")
    print(f"Number of images: {len(new_images)}")
    print(f"Number of annotations: {len(new_annotations)}")

    # Create the new COCO JSON file
    new_coco_data = {
        'categories': coco_data['categories'],
        'images': new_images,
        'annotations': new_annotations
    }

    # Save the new COCO JSON file
    with open(os.path.join(output_path, 'labels.json'), 'w') as f:
        json.dump(new_coco_data, f, indent=4)

    return n_images, n_annotations


def main():
    """

    :return:
    """
    train_dataset_path = 'path/to/dataset/train/'
    validation_dataset_path = 'path/to/dataset/validation/'
    test_dataset_path = 'path/to/dataset/test/'

    augmented_train_dataset_path = 'path/to/dest_base_path' + 'train_augmented/'
    augmented_validation_dataset_path = 'path/to/dest_base_path' + 'validation_augmented/'

    os.makedirs(os.path.join(augmented_train_dataset_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(augmented_validation_dataset_path, 'images'), exist_ok=True)

    with open(os.path.join(train_dataset_path, 'labels.json'), 'r') as f:
        train_json = json.load(f)

    with open(os.path.join(validation_dataset_path, 'labels.json'), 'r') as f:
        val_json = json.load(f)

    with open(os.path.join(test_dataset_path, 'labels.json'), 'r') as f:
        test_json = json.load(f)

    n_images = max(train_json['images'][-1]['id'], val_json['images'][-1]['id'], test_json['images'][-1]['id'])
    n_annotations = max(train_json['annotations'][-1]['id'], val_json['annotations'][-1]['id'],
                        test_json['annotations'][-1]['id'])

    if os.listdir(augmented_train_dataset_path + 'images/') == []:
        n_images, n_annotations = augment_dataset(train_dataset_path, augmented_train_dataset_path, get_transform('train'), n_images, n_annotations, num_augmentations=5)
    else:
        print("Augmentation on the training set has already been made.")

    if os.listdir(augmented_validation_dataset_path + 'images/') == []:
        augment_dataset(validation_dataset_path, augmented_validation_dataset_path, get_transform('validation'), n_images, n_annotations, num_augmentations=5)
    else:
        print("Augmentation on the validation set has already been made.")


    count1 = sum(1 for filename in os.listdir(os.path.join(augmented_train_dataset_path, 'images')) if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg']))
    count2 = sum(1 for filename in os.listdir(os.path.join(augmented_validation_dataset_path, 'images')) if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg']))

    print(f"Number of images in the train_augmented folder: {count1}") # TODO: make prettier
    print(f"Number of images in the validation_augmented folder: {count2}") # TODO: make prettier


if __name__ == '__main__':
    main()
