import pandas as pd
from tqdm import tqdm

import os

class DesignatorCopy:
    """
    Copy designator entries from base to target files.
    """
    def __init__(self, base_file, target_dir, output_dir):
        self.base_file = base_file
        self.target_dir = target_dir
        self.output_dir = output_dir

    def copy(self):
        """
        Copy designator entries from base to target files.
        """
        # create the output directories
        self._create_directories(self.output_dir)
        
        # for every file in the list
        names = os.listdir(self.target_dir)
        with tqdm(total=len(names)) as pbar:
            for file_name in names:
                
                # read the file into a dataframe
                df_base = pd.read_csv(self.base_file)
                df_target = pd.read_csv(self.target_dir + "/" + file_name)
                
                # copy the designator column from the base file to the target file
                df_target['Designator'] = df_base['Designator']
                
                # save the modified file
                df_target.to_csv(self.output_dir + "/" + file_name, index=False)
                
                # update progress bar
                pbar.update(1)

    def _create_directories(self, output_dir):
        """
        Create the output directories.
        :param output_dir: output directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)