import cv2
import numpy as np
import glob
import argparse
import os
import torch
import gc
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image

class MaskGenerator:
    def __init__(self, base_dir, mesh_name, frame_id, model_cfg, checkpoint):
        self.base_dir = base_dir
        self.color_files = sorted(glob.glob(f"{self.base_dir}/*.jpg"))
        self.mask_dir = f"{self.base_dir}/masks"
        self.camera_pose = sorted
        self.top_left = (0, 0)
        self.bottom_right = (0, 0)
        self.drawing = False
        self.image = None
        self.mesh_name = mesh_name
        self.frame_id = frame_id

        # Load the SAM2 model within each iteration
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    def __del__(self):
        # Explicitly delete the model to free up GPU memory
        del self.predictor
        torch.cuda.empty_cache()

    # Function to get points from the image using OpenCV
    def get_points(self, image):
        points = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow(f'{self.mesh_name}_{self.frame_id}', image)

        cv2.imshow(f'{self.mesh_name}_{self.frame_id}', image)
        cv2.setMouseCallback(f'{self.mesh_name}_{self.frame_id}', click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return np.array(points)

    # Function to create mask and display result using SAM2
    def create_mask_and_display_result(self):
        image = np.array(self.image.convert("RGB"))

        # Get points from user clicks
        points = self.get_points(image)
        if points.size == 0:
            print("No points were selected. Skipping mask generation for this frame.")
            return  # Skip further processing if no points were clicked

        input_labels = np.ones(len(points))

        # Use the model to predict masks based on the clicked points
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image)
            masks, _, _ = self.predictor.predict(points, input_labels)

        # Convert mask to a format suitable for OpenCV
        mask = (masks[0] * 255).astype(np.uint8)

        # Display the predicted mask using OpenCV
        cv2.imshow(f'{self.mesh_name}_{self.frame_id}_mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the mask using OpenCV
        mask_path = f"{self.base_dir}/masks/{self.mesh_name}_mask_{self.frame_id}.png"
        if not os.path.exists(self.mask_dir):
            os.makedirs(self.mask_dir)
        cv2.imwrite(mask_path, mask)
        print(f"Mask saved to {mask_path}")

        # Detach and delete tensors
        del masks
        torch.cuda.empty_cache()

    # Function to run the mask generation process
    def generation(self):
        # Load the image
        self.image = Image.open(self.color_files[self.frame_id])

        # Generate mask and display result
        self.create_mask_and_display_result()

        # Clear CUDA context to free memory
        torch.cuda.empty_cache()

def generate_mask(base_dir, mesh_name, frame_id, model_cfg, checkpoint):
    # Create and use the MaskGenerator in each iteration
    mask_generator = MaskGenerator(base_dir=base_dir, mesh_name=mesh_name, frame_id=frame_id, model_cfg=model_cfg, checkpoint=checkpoint)
    mask_generator.generation()
    del mask_generator  # Explicitly delete the instance to free memory

    # Run garbage collection to free up any unused memory
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    mesh_files = ['beaker_250ml.obj', 'conical_flask_500ml.obj','conical_flask_250ml.obj', 'beaker_30ml.obj']
    meshes = [f'{code_dir}/perception_data/objects/{mesh}' for mesh in mesh_files]
    parser.add_argument('--mesh_files', type=str, nargs='+', default=meshes)
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/perception_data/table/3')
    parser.add_argument('--model_cfg', type=str, default='sam2_hiera_l.yaml')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/sam2_hiera_large.pt')
    args = parser.parse_args()

    for frame_id in range(10):
        for mesh in args.mesh_files:
            mesh_name = os.path.basename(mesh).split('.')[0]
            generate_mask(args.test_scene_dir, mesh_name, frame_id, args.model_cfg, args.checkpoint)
