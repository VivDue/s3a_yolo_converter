from tqdm import tqdm
import pandas as pd

import os

class DesignatorReplace:
    """
    Replace designators specified in the replace_dict in the input_dir and save the modified files in the output_dir.
    """
    def __init__(self, input_dir, output_dir, replace_dict):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.replace_dict = replace_dict

    def replace(self):
        """
        Replace designators specified in the replace_dict in the input_dir and save the modified files in the output_dir.
        """
        # create the output directories
        self._create_directories(self.output_dir)
        
        # for every file in the list
        names = os.listdir(self.input_dir)
        with tqdm(total=len(names)) as pbar:
            for file_name in names:
                
                # read the file into a dataframe
                df = pd.read_csv(self.input_dir + "/" + file_name)
                
                # replace the specified designators with the corresponding keys
                for key, value in self.replace_dict.items():
                    for item in value:
                        df = df.replace(item, key)
                
                # save the modified file
                df.to_csv(self.output_dir + "/" + file_name, index=False)
                
                # update progress bar
                pbar.update(1)

    def _create_directories(self, output_dir):
        """
        Create the output directories.
        :param output_dir: output directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
