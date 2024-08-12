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
        

    def _split_image(self, image_name:str, patch_size:int)->None:
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
    
    def _split_annotation(self, annotation_path:str, patch_size:int)->None:
        """Split the annotation into patches.

        :param annotation_path: path to the annotation file.
        :param patch_size: size of the patches in Pixels.
        """
        df = pd.read_csv(annotation_path)
        image_name = df["Image File"].unique()[0]
        img_path = f"{self.image_dir}/{image_name}"
        image = cv2.imread(img_path)
        height, width, _ = image.shape

        patch_counter = 0
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                if i + patch_size > height:
                    i = height - patch_size
                if j + patch_size > width:
                    j = width - patch_size

                # Filter and process rows for the current patch
                new_rows = []
                for _, row in df.iterrows():
                    new_row = self._process_row(row, i, j, patch_size, patch_counter)
                    if new_row is not None:
                        new_rows.append(new_row)
                
                # Create patch annotation file
                self._create_patch_annotation(new_rows, df.columns, annotation_path, patch_counter)
                patch_counter += 1

    def _process_row(self, row:pd.Series, i:int, j:int, patch_size:int, patch_counter:int)->pd.Series:
        """
        Process each annotation row to check if it fits in the current patch and adjust coordinates.

        :param row: A row from the annotation DataFrame.
        :param i: Current vertical (y-axis) starting position of the patch.
        :param j: Current horizontal (x-axis) starting position of the patch.
        :param patch_size: Size of the patches in pixels.
        :param patch_counter: Index for the current patch.
        :return: The modified row if it fits in the patch, otherwise None.
        """
        try:
            vertices = self._get_valid_vertices(row)
        except ValueError:
            return None  # Skip rows with invalid vertices

        if vertices is None:
            return None  # Skip rows without valid designators

        x_bool = self._check_range(vertices[0::2], j, j + patch_size)
        y_bool = self._check_range(vertices[1::2], i, i + patch_size)
        if not x_bool or not y_bool:
            return None  # Skip if coordinates are not within the patch boundaries

        new_vertices = self._adjust_patch_boundaries(vertices, i, j, patch_size)
        
        row["Vertices"] = str([new_vertices])
        row["Image File"] = f"{row['Image File'].split('.')[0]}_{patch_counter}.png"
        return row

    def _get_valid_vertices(self, row:pd.Series)->np.ndarray:
        """
        Extract and validate vertices from the annotation row.

        :param row: A row from the annotation DataFrame.
        :return: A flattened NumPy array of vertices if valid, otherwise None.
        """
        try:
            vertices = np.array(ast.literal_eval(row["Vertices"]))[0].flatten()
            if pd.isna(row["Designator"]):
                return None
            return vertices
        except (ValueError, IndexError):
            return None  # Handle cases where vertices can't be parsed

    def _adjust_patch_boundaries(self, vertices:np.ndarray, i:int, j:int, patch_size:int)->list:
        """
        Adjust the vertices to fit within the patch boundaries.

        :param vertices: NumPy array of vertex coordinates.
        :param i: Current vertical (y-axis) starting position of the patch.
        :param j: Current horizontal (x-axis) starting position of the patch.
        :param patch_size: Size of the patches in pixels.
        :return: List of adjusted vertices within the patch boundaries.
        """
        new_vertices = []
        for idx, vertex in enumerate(vertices):
            if idx % 2 == 0:  # x-coordinate
                if vertex < j:
                    new_vertices.append(0)
                elif vertex > (j + patch_size):
                    new_vertices.append(patch_size-1)
                else:
                    new_vertices.append(int(vertex - j))
            else:  # y-coordinate
                if vertex < i:
                    new_vertices.append(0)
                elif vertex > (i + patch_size):
                    new_vertices.append(patch_size-1)
                else:
                    new_vertices.append(int(vertex - i))
        
        new_vertices = np.array(new_vertices).reshape(-1, 2)
        _, idx = np.unique(new_vertices, axis=0, return_index=True)
        return new_vertices[np.sort(idx)].tolist()

    def _create_patch_annotation(self, new_rows:list, columns:pd.Index, annotation_path:str, patch_counter:int)->None:
        """
        Create and save a new annotation file for a patch.

        :param new_rows: List of rows to include in the patch annotation.
        :param columns: The columns of the original DataFrame.
        :param annotation_path: Path to the original annotation file.
        :param patch_counter: Index for the current patch.
        """
        if new_rows:
            temp_df = pd.DataFrame(new_rows, columns=columns)
            annotation_name = os.path.basename(annotation_path).split(".")[0]
            path = f'{self.output_dir}/ann/{annotation_name}_{str(patch_counter)}.csv'
            temp_df.to_csv(path, index=False)


    def _check_range(self, arr:np.ndarray, min_value:int, max_value:int)->bool:
        """
        Check if any values in the array are in the range of min and max.

        :param arr: NumPy array of values.
        :param min_value: Minimum value of the range.
        :param max_value: Maximum value of the range.
        :return: True if any values are in the range, False otherwise.
        """
        required_values = np.arange(min_value, max_value)
        
        # check if any of required values are in the array
        return bool(np.any(np.isin(arr, required_values)))

    def _create_directories(self, output_dir:str)->None:
        """
        Create the output directories if they don't already exist.

        :param output_dir: Path to the output directory.
        """
        # create output directory if not exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # create ann and img subdirectories
        if not os.path.exists(output_dir + "/ann"):
            os.makedirs(output_dir + "/ann")

        if not os.path.exists(output_dir + "/img"):
            os.makedirs(output_dir + "/img")