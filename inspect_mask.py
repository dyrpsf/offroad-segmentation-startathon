import os
from PIL import Image
import numpy as np

mask_path = r"E:\Startathon\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Segmentation"
files = sorted(os.listdir(mask_path))

sample = os.path.join(mask_path, files[0])
img = Image.open(sample)
arr = np.array(img)

print("Shape:", arr.shape)
print("Dtype:", arr.dtype)

if arr.ndim == 2:
    vals = np.unique(arr)
    print("Unique values (first 50):", vals[:50])
    print("Number of unique values:", len(vals))
else:
    h, w, c = arr.shape
    arr_reshaped = arr.reshape(-1, c)
    uniq = np.unique(arr_reshaped, axis=0)
    print("Number of unique colors:", len(uniq))
    print("First 20 colors (R,G,B):")
    for v in uniq[:20]:
        print(v)