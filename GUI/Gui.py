import SimpleITK as sitk
import numpy as np
import os
from PIL import Image
dir=r"C:\Users\apoll\Desktop\university\third\CV\Project_orient\1_MSCMR_orient\C0\000\patient1_C0.nii.gz"
image=sitk.ReadImage(dir)
image_array=sitk.GetArrayFromImage(image)
print(image_array.shape)
for j in range(image_array.shape[0]):
    pic=Image.fromarray(image_array[j,:,:])
    pic.show()