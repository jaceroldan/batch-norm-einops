from PIL import Image
import numpy as np
import torch
from torch.nn import BatchNorm2d

# Load both images
image_1 = Image.open("dataset/aki_dog.jpg")
image_2 = Image.open("dataset/wonder_cat.jpg")

# Convert both images to numpy arrays
aki_dog = np.array(image_1)
wonder_cat = np.array(image_2)

# Convert to Torch tensors, add batch dimension and adjust dimensions
aki_dog = torch.from_numpy(aki_dog).permute(2, 0, 1).unsqueeze(0).type(torch.FloatTensor)
wonder_cat = torch.from_numpy(wonder_cat).permute(2, 0, 1).unsqueeze(0).type(torch.FloatTensor)

# Stack both images along the batch dimension (axis 0)
batch_images = torch.cat([aki_dog, wonder_cat], dim=0)

# Apply Batch Normalization to the batch
normalizer = BatchNorm2d(3, affine=False)
batch_output = normalizer(batch_images)

# Convert the output back to images for both
for i in range(batch_output.shape[0]):
    # Clip the output to the valid range [0, 1]
    torch_bn2d_to_numpy = batch_output[i].permute(1, 2, 0).numpy()
    torch_bn2d_to_numpy = np.clip(torch_bn2d_to_numpy, 0, 1)  # Clip to [0, 1]
    
    # Convert back to 8-bit values and display the image
    torch_bn2d_to_numpy_img = Image.fromarray(np.uint8(torch_bn2d_to_numpy * 255))
    
    print(f"Image {i + 1} after BatchNorm:")
    torch_bn2d_to_numpy_img.save(f'results/normalized_image_{i}_torch.jpg')
