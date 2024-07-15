# Oxford5k-Paris6k-ObjectDetection [not finished yet] 

This project aims to create an object detection model for monument recognition using the Oxford5k and Paris6k datasets. The model is built using MediaPipe Model Maker for transfer learning, starting from a pre-trained model.

## Project Overview

The main objective of this project is to adapt the Oxford5k and Paris6k datasets, originally designed for image retrieval, for object detection tasks. This involved significant work in converting the annotations from their original format (stored in .pkl files) to standard object detection formats such as Pascal VOC and COCO.

## Key Features

- Adaptation of Oxford5k and Paris6k datasets for object detection
- Custom scripts for data preprocessing and annotation conversion
- Transfer learning using MediaPipe Model Maker
- Support for both Pascal VOC and COCO annotation formats

## Getting Started

1. Clone the repository
2. Install the required dependencies
3. Run the data preparation scripts in the `scripts/` directory
4. Use the Jupyter notebooks in the `training/` directory for model training

## Data Preparation

The `scripts/` directory contains various Python scripts for data preparation:

- `get_data.py`: downloads the original datasets
- `create_annotations.py`: converts original annotations to Pascal VOC and COCO formats
- `prepare_dataset.py`: prepares the dataset for training
- `check_annotations.py`: verifies the correctness of the converted annotations

## Training

The `training/` directory contains Jupyter notebooks for model training:

- `mediapipe_object_detector_model_customization_template.ipynb`: template for MediaPipe Model Maker
- `mp_training_paris6k.ipynb`: specific training notebook for the Paris6k dataset

## Inference

Use the scripts in the `inference/` directory to run object detection on new images.

## License

This project is licensed under the [LICENSE NAME] - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments

- Original Oxford5k and Paris6k dataset creators
- MediaPipe team for their Model Maker tool

## References

- [Oxford5k Dataset](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)
- [Paris6k Dataset](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)

## Authors

- [Elia Innocenti](https://github.com/eliainnocenti)
