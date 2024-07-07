"""
TODO: add description
"""

import os
from lib.revisitop.python.download import download_datasets

# TODO: check revisitop license

def main():
    """

    :return:
    """
    data_dir = "../../../Data"  # TODO: check -> I don't remember if it creates any directories
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return

    download_datasets(data_dir)

if __name__ == '__main__':
    main()
