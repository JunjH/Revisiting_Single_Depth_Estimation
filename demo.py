from fire import Fire
import numpy as np
from PIL import Image
from scipy.misc import imresize
import torch
import torch.nn.parallel

from models import modules, net, resnet, densenet, senet
import loaddata_demo as loaddata

import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model
   

def main(image_path):
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(
        f='./pretrained_model/model_senet',
        map_location=None if torch.cuda.is_available() else 'cpu'))
    model.eval()

    nyu2_loader = loaddata.readNyu2(image_path)
  
    test(nyu2_loader, model)


def test(nyu2_loader, model):
    for i, image in enumerate(nyu2_loader):     
        image = torch.autograd.Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        out = model(image)

        out = out.view(out.size(2), out.size(3)).data.cpu().numpy()
        input_shape = image.data.cpu().numpy().shape[2:4]
        out = imresize(arr=out, size=input_shape)
        Image.fromarray(out.astype(np.uint8)).save('data/demo/out.png')

        # matplotlib.image.imsave('data/demo/out.png', out)

if __name__ == '__main__':
    Fire(main)
