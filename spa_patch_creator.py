import os 
from glob import glob
import ast

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

class SpaPatchCreator:
    def __init__(self, ann_dir:str, img_dir:str, output_dir:str, patch_size:int):
        """
        :param ann_dir: path to the annotations directory.
        :param img_dir: path to the images directory.
        :param output_dir: path to the output directory.
        :param pathsize: size of the patches in Pixels.
        """
        self.annotation_dir = ann_dir
        self.image_dir = img_dir
        self.output_dir = output_dir
        self.patch_size = patch_size

    def split(self):
        """Split the Annotations and Images into patches. Output the patches to the output directory."""
        
        # create the output directories
        self._create_directories(self.output_dir)

        # create annotations and image path list
        annotations_list = glob(os.path.join(self.annotation_dir, "*.csv"))
        

        # for every file in the annotations list
        with tqdm(total=len(annotations_list)) as pbar:
            for ann_file in annotations_list:
                
                # read the csv file
                df = pd.read_csv(ann_file)
                image_name = list(df["Image File"].unique())[0]

                # split the image into patches
                self._split_image(image_name, self.patch_size)

                # split the annotation into patches
                self._split_annotation(ann_file, self.patch_size)

                # update progres bar
                pbar.update(1)
        

    def _split_image(self, image_name:str, patch_size:int):
        """Split the image into patches.
        :param image_path: path to the image.
        :param patch_size: size of the patches in Pixels.
        """
        img_path = f"{self.image_dir}/{image_name}"
        image = cv2.imread(img_path)
        height, width, _ = image.shape
        patches = []

        # iterate over width and height of the image and extract patches
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                temp = image[i:i+patch_size, j:j+patch_size]
                # check if the extracted image is smaller than patch size
                # and move the starting x and y point to match the given patch size  
                if temp.shape[0] < patch_size:
                    i = height - patch_size
                if temp.shape[1] < patch_size:
                    j = width - patch_size
                patch = image[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
        
        # split extension from the image name
        image_name = image_name.split(".")[0]

        # write patches to the output directory
        for i, patch in enumerate(patches):
            cv2.imwrite(f'{self.output_dir}/img/{image_name}_{str(i)}.png', patch)
    
    def _split_annotation(self, annotation_path:str, patch_size:int):
        """Split the annotation into patches.
        :param annotation_path: path to the annotation.
        :param patch_size: size of the patches in Pixels.
        """
        df = pd.read_csv(annotation_path)
        patches = []

        image_name = df["Image File"].unique()[0]
        img_path = f"{self.image_dir}/{image_name}"
        image = cv2.imread(img_path)
        height, width, _ = image.shape

        # iterate over width and height of the image and extract patches
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                # check if the extracted image is smaller than patch size
                # and move the starting x and y point to match the given patch size
                if i + patch_size > height:
                    i = height - patch_size
                if j + patch_size > width:
                    j = width - patch_size


                drop_list = []
                new_vertices = []
                # iterate over the rows of the annotation and extract patches
                for idx, row in df.iterrows():
                    # try to get the vertices, if not possible continue to the next row
                    
                    try:
                        vertices = np.array(ast.literal_eval(row["Vertices"]))[0].flatten()
                    except:
                        drop_list.append(idx)
                        continue

                    # check if any x and y coordinates are inside the corresponding patch boundaries
                    x_bool = self._check_range(vertices[::2], i, i + patch_size)
                    y_bool = self._check_range(vertices[1::2], j, j + patch_size)
                    
                    if not x_bool and not y_bool:
                        # delete current row if not all coordinates are inside the patch boundaries
                        drop_list.append(idx)
                        continue

                    # set new vertices coordinates correspnding to the new patch size
                    # if the coordinates are outside the patch boundaries, set them to 0 or patch_size
                    
                    for idx, vertex in enumerate(vertices):
                        if idx % 2 == 0:
                            if vertex < i:
                                new_vertices.append(0)
                            elif vertex > i + patch_size:
                                new_vertices.append(patch_size)
                            else:
                                new_vertices.append(vertex - i)
                        else:
                            if vertex < j:
                                new_vertices.append(0)
                            elif vertex > j + patch_size:
                                new_vertices.append(patch_size)
                            else:
                                new_vertices.append(vertex - j)

                # drop the rows from the dataframe
                df.drop(drop_list, inplace=True)

                # update the vertices in the dataframe
                #new_vertices = np.array(new_vertices).reshape(-1, 2).tolist()
                new_vertices = [[1,2],[2,3]]
                df.at[idx, "Vertices"] = str([new_vertices])
                
                # append the new annotation to the patches list
                patches.append(df)
        
        # split extension from the annotation name
        annotation_file = os.path.basename(annotation_path)
        annotation_name = annotation_file.split(".")[1]

        # write patches to the output directory
        for i, patch in enumerate(patches):
            patch.to_csv(f'{self.output_dir}/ann/{annotation_name}_{str(i)}.csv', index=False)


    def _check_range(self, arr, min_value, max_value):
        """Check if any values in the list are in the range of min and max.
        :param arr: array of values.
        :param min_value: minimum value.
        :param max_value: maximum value.

        :return: True if any values is in the range, False otherwise.
        """
        required_values = np.arange(min_value, max_value + 1)
        
        # check if any of required values are in the array
        return bool(np.any(np.isin(arr, required_values)))

    def _create_directories(self, output_dir):
        """Create the output directories.
        :param output_dir: path to the output directory.
        """
        # create output directory if not exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # create ann and img subdirectories
        if not os.path.exists(output_dir + "/ann"):
            os.makedirs(output_dir + "/ann")

        if not os.path.exists(output_dir + "/img"):
            os.makedirs(output_dir + "/img")