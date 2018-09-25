
import torch
from PIL import Image
import numpy as np

try:
    import accimage
except ImportError:
    accimage = None




def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = self.changeScale(image,self.size)
      
        return image

    def changeScale(self, img, size, interpolation=Image.BILINEAR):
        ow, oh = size     

        return img.resize((ow, oh), interpolation)
  
class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = self.centerCrop(image,self.size)

        return image

    def centerCrop(self,image, size):       
        w1, h1 = image.size
        tw, th = size

        if w1 == tw and h1 == th:
            return image

        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))
       
        image = image.crop((x1, y1, tw+x1, th+y1))

        return image


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, image):       
        image = self.to_tensor(image)
       
        return image
     

    def to_tensor(self,pic):
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
           
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(255)
            

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img



class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = self.normalize(image, self.mean, self.std)

        return image

    def normalize(self, tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
            
        return tensor
