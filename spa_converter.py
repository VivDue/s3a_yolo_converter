import os
import shutil
from glob import glob
import ast

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm


class SpaConverter:
    """
    Convert annotations from SPA to the YOLO format.
    """

    def __init__(self, ann_dir:str, img_dir:str, output_dir:str, names:dict, precision:int=3):
        """
        :param ann_dir: path to the annotations directory.
        :param img_dir: path to the images directory.
        :param output_dir: path to the output directory.
        :param names: dictionary containing the class names and their corresponding class ids.
        :param precision: precision of the bounding box coordinates.
        """
        self.annotation_dir = ann_dir
        self.image_dir = img_dir
        self.output_dir = output_dir
        self.names = names
        self.precision = precision

    def convert(self):
        """
        Convert annotations from SPA to the YOLO format.
        """

        # create the output directories
        self._create_directories(self.output_dir)

        # create annotations list
        annotations_list = glob(os.path.join(self.annotation_dir, "*.csv"))
        
        # for every file in the annotations list
        with tqdm(total=len(annotations_list)) as pbar:
            for ann_file in annotations_list:

                # read the csv file
                df = pd.read_csv(ann_file)

                # checking if atleast 1 designation is present in annotation
                if df["Designator"].isna().sum() != df.shape[0]:

                    # image file name, width and height
                    image_name = list(df["Image File"].unique())[0]
                    image_width, image_height = self._get_image_width_height(image_name)

                    # create label file and add to output directory
                    self._create_label_file(df, image_name, image_width, image_height)

                    # insert image to output directory
                    self._insert_image_file(image_name)

                # update progress bar   
                pbar.update(1)

        # create yaml file
        self._create_yaml_file(self.output_dir)
    
    def _get_image_width_height(self, image_name):
        """
        Get the width and height of the image.
        :param image_name: name of the image file.
        :return: width and height of the image.
        """
        # read the image
        image = cv2.imread(self.image_dir + "/" + image_name)

        # get the width and height of the image
        height, width, _ = image.shape

        return width, height
    
    def _create_label_file(self, df, image_name, image_width, image_height):
        """
        Create the label file for the image.
        :param df: pandas dataframe containing the annotations.
        :param file_name: name of the file.
        """
        # exhanging the file extension
        file_name = image_name.split(".")[0]
        file_name = file_name + ".txt"

        # create the label file
        with open(self.output_dir + "/labels/" + file_name, "w") as f:
            # for every row in the dataframe
            for _, row in df.iterrows():

                # skip row if designation is not present
                designator = row["Designator"]
                if designator not in self.names:
                    continue

                # get the class id and the bounding box coordinates
                class_id = self.names[designator]
                vertices = np.array(ast.literal_eval(row["Vertices"]))[0].flatten()

                # write the class id and the bounding box coordinates to the file
                f.write(f"{class_id}")
                for idx, vertex in enumerate(vertices):

                    # normalize vertex according to the image width and height
                    if idx % 2 == 0:
                        vertex = vertex / image_width
                    else:
                        vertex = vertex / image_height
                    vertex = round(vertex, self.precision)

                    # write the vertex to the file
                    f.write(f" {vertex}")
                
                # proceed to the next line
                f.write("\n")

    def _insert_image_file(self, image_name):
        """
        Insert the image to the output directory.
        :param image_name: name of the image file.
        """
        # copy the image to the output directory
        src = f"{self.image_dir}/{image_name}"
        dst = f"{self.output_dir}/images/{image_name}"
        shutil.copy(src, dst)

    def _create_yaml_file(self, output_dir):
        """
        Create the yaml file for the dataset.
        :param path: path to the output directory.
        """
        # create the yaml file
        with open(f"{output_dir}/{output_dir}.yaml", "w") as f:
            # paths section
            f.write("\n# Paths\n")
            f.write(f"path: {output_dir}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")

            # classes section
            f.write("\n# Classes\n")
            f.write(f"nc: {(len(self.names))}\n")
            f.write("names: [")
            for name in self.names:
                f.write(f"{name}")
                if name != list(self.names.keys())[-1]:
                    f.write(", ")
            f.write("]")

    def _create_directories(self, output_dir):
        """
        Create the output directories for the images and the labels.
        :param path: path to the output directory.
        """
        # create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # create image and label subdirectories
        if not os.path.exists(output_dir + "/images"):
            os.makedirs(output_dir + "/images")

        if not os.path.exists(output_dir + "/labels"):
            os.makedirs(output_dir + "/labels")