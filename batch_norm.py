from PIL import Image
import numpy as np
import torch
from torch.nn import BatchNorm2d


image = Image.open("dataset/aki_dog.jpg")

aki_dog = np.array(image)

print(aki_dog.shape)
aki_dog = torch.from_numpy(aki_dog).permute(2,0,1).unsqueeze(0).type(torch.FloatTensor)

print(aki_dog.shape)

normalizer = BatchNorm2d(3, affine=False)
aki_dog_output = normalizer(aki_dog)

print(aki_dog_output)

torch_bn2d_to_numpy = aki_dog_output.squeeze(0).permute(1,2,0).numpy()
print(torch_bn2d_to_numpy)

torch_bn2d_to_numpy_img = Image.fromarray(np.uint8(torch_bn2d_to_numpy * 255))
print(torch_bn2d_to_numpy_img)

torch_bn2d_to_numpy_img.show()

