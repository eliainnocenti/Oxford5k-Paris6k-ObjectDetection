"""
TODO: add description
mention:
- the revisitop license
- the directory structure that will be created
- where is it possible to find the ground truth files
"""

import os
from lib.revisitop.python.download import download_datasets

# TODO: check revisitop license


def main():
    """

    :return:
    """
    data_dir = "../../../Data"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return

    download_datasets(data_dir)
    # it will create the following directories:
    # ../../../Data/datasets/roxford5k/jpg
    # ../../../Data/datasets/rparis6k/jpg


if __name__ == '__main__':
    main()
