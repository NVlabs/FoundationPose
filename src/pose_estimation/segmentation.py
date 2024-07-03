import cv2
import numpy as np
import glob
import argparse
import os
import rospy
class MaskGenerator:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.color_files = sorted(glob.glob(f"{self.base_dir}/rgb/*.jpg"))
        self.mask_dir = f"{self.base_dir}/masks"
        # Initialize global variables
        self.top_left = (0, 0)
        self.bottom_right = (0, 0)
        self.drawing = False
        self.image = None

    # Mouse callback function
    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.top_left = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                image_copy = self.image.copy()
                self.bottom_right = (x, y)
                cv2.rectangle(image_copy, self.top_left, self.bottom_right, (0, 255, 0), 2)
                cv2.imshow('Image', image_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.bottom_right = (x, y)
            cv2.rectangle(self.image, self.top_left, self.bottom_right, (0, 255, 0), 2)
            cv2.imshow('Image', self.image)
            self.create_mask_and_display_result()

    # Function to create mask and display result
    def create_mask_and_display_result(self):
        mask = np.zeros_like(self.image)
        cv2.rectangle(mask, self.top_left, self.bottom_right, (255, 255, 255), thickness=cv2.FILLED)
        white_image = np.ones_like(self.image) * 255
        result = np.where(mask == 255, white_image, 0)
        mask_path = self.color_files[0].replace('rgb','masks')
        cv2.imwrite(mask_path, result)
        print(f"Mask saved to {mask_path}")
        cv2.imshow('Result', result)

    # Function to run the mask generation process
    def generation(self):
        # Load the image
        self.image = cv2.imread(self.color_files[0])

        # Create a window and set the mouse callback function
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.draw_rectangle)

        # Display the image and wait for the user to draw the bounding box
        cv2.imshow('Image', self.image)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Press Enter to continue...")
    

# # testing the MaskGenerator class
# if __name__=='__main__':
#     parser = argparse.ArgumentParser()
#     code_dir = os.path.dirname(os.path.realpath(__file__))
#     parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/perception_data/test3')
#     args = parser.parse_args()
#     # Usage
#     base_dir = args.test_scene_dir  # Update this to your base directory
#     mask_generator = MaskGenerator(base_dir = base_dir)
#     mask_generator.generation()

