"""
This script downloads image datasets required for the Revisited Oxford and Paris retrieval benchmarks.

License:
---------
# TODO: check the license
This code utilizes the Revisited Oxford and Paris (RevisitOP) datasets, which are provided under their respective licenses.
For more information on the RevisitOP license, please refer to the official RevisitOP GitHub repository or the dataset's documentation.

Directory Structure:
---------------------
When the script is run, it will create the following directory structure under the specified data directory:

<root_data_directory>/
    datasets/
        roxford5k/
            jpg/
        rparis6k/
            jpg/

Ground Truth Files:
-------------------
# TODO: update
The ground truth files for the RevisitOP datasets can be found within the datasets themselves, usually in a subdirectory such as 'gnd' or 'ground_truth' within 'roxford5k' and 'rparis6k'.
These files provide the necessary information for evaluation of image retrieval tasks.

Dependencies:
-------------
- lib.revisitop (Ensure that the RevisitOP library is correctly installed and accessible in your Python environment)

Usage:
------
To run this script, simply execute it in a Python environment where the required libraries are installed and the data directory is correctly set.
"""

import os
from lib.revisitop.python.download import download_datasets


def main():
    """
    Main function to initiate the download of RevisitOP datasets.

    It checks if the specified data directory exists, and if so,
    it proceeds to download the datasets into the appropriate directory structure.

    :return: None
    """
    data_dir = "../../../Data"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return

    download_datasets(data_dir)


if __name__ == '__main__':
    main()
