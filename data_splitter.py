import os
import numpy as np
from glob import glob
import shutil
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, input_path:str, output_path:str, test_size:str):
        """
        :param input_path: path to the input directory.
        :param output_path: path to the output directory.
        :param test_size: size of the test set.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.test_size = test_size

    def split(self):
        """
        Split the data into train and test sets.
        """
        
        # read the directory
        label_files, image_files = self._read_directory(self.input_path)

        # create the directories
        self._create_directories(self.output_path)
        
        # split the data
        self._split_data(label_files, image_files, self.test_size)
        

    def _read_directory(self, input_path:str):
        """
        Read the input directory.

        :param input_path: path to the input directory.
        :return: list of label files and image files.
        """
        # path to directories
        label_dir = os.path.join(input_path, "labels")
        image_dir = os.path.join(input_path, "images")

        label_files = glob(os.path.join(label_dir ,"*.txt"))
        image_files = glob(os.path.join(image_dir, "*.*"))
        return label_files, image_files
    
    def _create_directories(self, output_dir:str):
        """
        Create the output directories.
        
        :param output_dir: path to the output directory.
        """
        # create the output directories
        if not os.path.exists(output_dir):
            os.makedirs(self.output_path + "/train/images", exist_ok=True)
            os.makedirs(self.output_path + "/train/labels", exist_ok=True)
            os.makedirs(self.output_path + "/val/images", exist_ok=True)
            os.makedirs(self.output_path + "/val/labels", exist_ok=True)

    def _split_data(self, label_files:list[str], image_files:list[str], test_size:float):
        """
        Split the data into train and test sets.
        
        :param label_files: list of label files.
        :param image_files: list of image files.
        :param test_size: size of the test set.
        """
        # create the indices
        idx = list(range(len(label_files)))

        # split the indices into train and test sets
        train_idx, test_idx = train_test_split(idx, test_size=test_size)

        # split the data into train and test sets based on the indices
        train_labels = [label_files[i] for i in train_idx]
        test_labels = [label_files[i] for i in test_idx]
        
        train_images = [image_files[i] for i in train_idx]
        test_images = [image_files[i] for i in test_idx]

        # copy the files to the output directory
        for label in train_labels:
            shutil.copy(label, self.output_path + "/train/labels")
        for label in test_labels:
            shutil.copy(label, self.output_path + "/val/labels")
        for image in train_images:
            shutil.copy(image, self.output_path + "/train/images")
        for image in test_images:
            shutil.copy(image, self.output_path + "/val/images")