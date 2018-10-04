import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from demo_transform import *



class depthDataset(Dataset):
    def __init__(self, filename, transform=None):
        self.frame = filename
        self.transform = transform

    def __getitem__(self,idx):       
        image = Image.open(self.frame)
        
        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return int(1)
     


def readNyu2(filename):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    image_trans = depthDataset(filename,
                        transform=transforms.Compose([
                        Scale([320, 240]),
                        CenterCrop([304, 228]),
                        ToTensor(),                                
                        Normalize(__imagenet_stats['mean'],
                                 __imagenet_stats['std'])
                       ]))

    image = DataLoader(image_trans, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)


    return image
