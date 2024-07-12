import cv2
import numpy as np
import glob
import argparse
import os
#import rospy
class MaskGenerator:
    def __init__(self, base_dir, mesh_name):
        self.base_dir = base_dir
        self.color_files = sorted(glob.glob(f"{self.base_dir}/rgb/*.jpg"))
        self.mask_dir = f"{self.base_dir}/masks"
        # Initialize global variables
        self.top_left = (0, 0)
        self.bottom_right = (0, 0)
        self.drawing = False
        self.image = None
        self.mesh_name = mesh_name

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
                cv2.imshow(f'{self.mesh_name}', image_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.bottom_right = (x, y)
            cv2.rectangle(self.image, self.top_left, self.bottom_right, (0, 255, 0), 2)
            cv2.imshow(f'{self.mesh_name}', self.image)
            self.create_mask_and_display_result()

    # Function to create mask and display result
    def create_mask_and_display_result(self):
        mask = np.zeros_like(self.image)
        cv2.rectangle(mask, self.top_left, self.bottom_right, (255, 255, 255), thickness=cv2.FILLED)
        white_image = np.ones_like(self.image) * 255
        result = np.where(mask == 255, white_image, 0)
        mask_path = f"{self.base_dir}/masks/{self.mesh_name}_mask.jpg" 
        if not os.path.exists(self.mask_dir):
            os.makedirs(self.mask_dir)
        cv2.imwrite(mask_path, result)
        print(f"Mask saved to s{mask_path}")
        cv2.imshow('Result', result)

    # Function to run the mask generation process
    def generation(self):
        # Load the image

        self.image = cv2.imread(self.color_files[0])

        # Create a window and set the mouse callback function
        cv2.namedWindow(f'{self.mesh_name}')
        cv2.setMouseCallback(f'{self.mesh_name}', self.draw_rectangle)

        # Display the image and wait for the user to draw the bounding box
        cv2.imshow(f'{self.mesh_name}', self.image)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Press Enter to continue...")
    

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--base_dir', type=str, default='perception_data/test9')
#     parser.add_argument('--mesh_name', type=str, default='cube')
#     args = parser.parse_args()

#     mask_generator = MaskGenerator(base_dir=args.base_dir, mesh_name=args.mesh_name)
#     mask_generator.generation()
