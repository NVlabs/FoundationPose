import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the image
image_path = "perception_data/black_paper/0/image_1698292137.jpg"
image = cv2.imread(image_path)

# Load the bounding box coordinates
bbox_path = "perception_data/black_paper/0/bbox_1698292137.npy"
bbox = np.load(bbox_path)

# Print bbox to verify the contents
print("Bounding box coordinates:", bbox)

# Create a mask with the same dimensions as the image
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Iterate over each bounding box and fill the mask
for box in bbox:
    x_min, y_min, x_max, y_max = box
    mask[y_min:y_max, x_min:x_max] = 255

# Apply the mask to the image
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Display the original image, mask, and masked image
plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# plt.title("Original Image")
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(mask, cmap='gray')

# save the mask as a png
mask_path = "mask.jpg"
cv2.imwrite(mask_path, mask)
# plt.subplot(1, 3, 3)
# plt.title("Masked Image")
# plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))

# plt.show()
