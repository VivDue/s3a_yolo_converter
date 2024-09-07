import cv2
from tqdm import tqdm

import os

class HsiClaheConverter:
    """
    Convert images from BGR to HSI and apply CLAHE.
    """
    def __init__(self, img_dir, output_dir, precision):
        """
        Initialize the HSI_CLAHE_Converter class.
        :param img_dir: directory containing the images.
        :param output_dir: output directory.
        :param precision: precision of the CLAHE.
        """
        self.image_dir = img_dir
        self.output_dir = output_dir
        self.precision = precision

    def convert(self):
        """
        Convert images from BGR to HSI and apply CLAHE.
        """
        # create the output directories
        self._create_directories(self.output_dir)
        
        # for every image in the list
        names = os.listdir(self.image_dir)
        with tqdm(total=len(names)) as pbar:
            for image_name in names:
                
                # read the image
                img = cv2.imread(self.image_dir + "/" + image_name)
                
                # convert the image from BGR to HSI
                img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
                
                # apply CLAHE to the intensity channel
                img1[:, :, 1] = cv2.createCLAHE(clipLimit=self.precision, tileGridSize=(8, 8)).apply(img1[:, :, 1])
                
                # convert the image back to BGR
                #img2 = cv2.cvtColor(img1, cv2.COLOR_HLS2BGR)
                
                # save the image
                cv2.imwrite(self.output_dir + "/" + image_name, img1)
                
                # update progress bar
                pbar.update(1)
    
    def _create_directories(self, output_dir):
        """
        Create the output directories.
        :param output_dir: output directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)