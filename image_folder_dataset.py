"""
Definition of ImageFolderDataset dataset class
"""

# pylint: disable=too-few-public-methods

import os
import torch
from .base_dataset import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class ImageFolderDataset(Dataset):
    """loads png images from a folder and applies a transform to them"""

    def __init__(self, *args, root=None, transform=None, **kwargs):
        super().__init__(*args, root=root, **kwargs)
        self.transform = transform
        self.images = []
        #load the first 60000 images from the folder
        i = 0
        for image in os.listdir(os.path.join(root)):
            #open the png file and convert it to a tensor if it is a png file
            if not image.endswith('.png'):
                continue 
            img = Image.open(os.path.join(root, image))
            img = img.convert('RGB')
            img = np.array(img)
            #convert uint8 to float32
            img = img.astype(np.float32)
            img = torch.tensor(img)
            #is h w c, convert to c h w
            img = img.permute(2, 0, 1)
            #scale the image down to size 28x28
            img = transforms.Resize((128, 128))(img)/255
            self.images.append(img)
            i += 1
            if i == 3000:
                break
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image
"""    
class ImageFolderDataset(Dataset):
    #CIFAR-10 dataset class

    def __init__(self, *args,
                 root=None,
                 images=None,
                 labels=None,
                 transform=None,
                 download_url="https://i2dl.vc.in.tum.de/static/data/mnist.zip",
                 **kwargs):
        super().__init__(*args,
                         download_url=download_url,
                         root=root,
                         **kwargs)
        print(download_url)
        self.images = torch.load(os.path.join(root, images))
        if labels is not None:
            self.labels = torch.load(os.path.join(root, labels))
        else:
            self.labels = None
        # transform function that we will apply later for data preprocessing
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is not None:
            return image, self.labels[index]
        else:
            return image
"""