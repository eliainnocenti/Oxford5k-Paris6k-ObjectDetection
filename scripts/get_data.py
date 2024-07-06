"""
TODO: add description
"""

import os
from lib.revisitop.python.download import download_datasets


data_dir = "../../../Data" # TODO: check
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

download_datasets(data_dir)
