from einops import rearrange, reduce
from PIL import Image
import numpy as np

# Load both images
image_1 = Image.open("dataset/aki_dog.jpg")
image_2 = Image.open("dataset/wonder_cat.jpg")

# Convert both images to numpy arrays
aki_dog = np.array(image_1, dtype=np.float32)
wonder_cat = np.array(image_2, dtype=np.float32)

# Rearrange dimensions for both images and stack them to create a batch
aki_dog = rearrange(aki_dog, 'h w c -> 1 c h w')  # Convert to batch format
wonder_cat = rearrange(wonder_cat, 'h w c -> 1 c h w')

# Stack both images along the batch dimension (axis 0)
batch_images = np.concatenate([aki_dog, wonder_cat], axis=0)  # Shape: (2, c, h, w)

# Compute mean and variance per channel across the batch (like BatchNorm2d)
MEAN = reduce(batch_images, 'b c h w -> c () ()', 'mean')  # Single mean across batch
variance = reduce((batch_images - MEAN) ** 2, 'b c h w -> c () ()', 'mean')  # Single variance across batch
standard_dev = variance ** 0.5

# Normalize the images
normalized_batch = (batch_images - MEAN) / (standard_dev + 1e-5)

# Avoid clipping or rescaling manually
# Now, convert back to image format for each image in the batch
for i in range(normalized_batch.shape[0]):
    normalized_image = rearrange(normalized_batch[i], 'c h w -> h w c')  # Rearrange to HWC format
    
    # Convert back to the original scale [0, 255] for viewing
    normalized_image = np.clip(normalized_image, 0, 1)  # Keep values between 0 and 1
    normalized_image = np.uint8(normalized_image * 255)

    # Convert to PIL image and save or display
    normalized_image_pil = Image.fromarray(normalized_image)
    normalized_image_pil.save(f'results/normalized_image_{i+1}_einops_final.jpg')
