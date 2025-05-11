#%%
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# %% import image
img_path = Path('C:/SDS/Smart-Leaf/SDS-CP028-smart-leaf/submissions/team-members/duncan-gitau/SmartLeaf_dataset/train/Corn___Common_Rust/image (1).JPG')
img = Image.open(img_path)
#img.show()
#print(img.size) #(256, 256)

# %% compose a series of steps
preprocess_steps = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.RandomRotation(50),
   transforms.CenterCrop(200),
   transforms.Grayscale(),
   transforms.RandomVerticalFlip(),
   transforms.ToTensor(),
   #transforms.Normalize(mean=0.5,std=0.5)

])

x= preprocess_steps(img)

# %% visualize images
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imshow(np.transpose(npimg))
    plt.show()
imshow(x)
print("shape :",x.shape)
mean=x.mean()
std=x.std()
print("mean :", mean)
print("std :", std)