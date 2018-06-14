import argparse
import torch
import torch.backends.cudnn as cudnn
import net 
import loaddata
import pdb

import matplotlib
import matplotlib.pyplot as plt
plt.set_cmap("jet")

parser = argparse.ArgumentParser(description='single depth prediction with base-refinement network')
parser.add_argument('--base_nyu2', default='models/base_nyu2', type=str,
                    help='name of experiment')
parser.add_argument('--refine_nyu2', default='models/refine_nyu2', type=str,
                    help='name of experiment')

def main():
    global args
    args = parser.parse_args()
    cudnn.benchmark = True

    nyu2_loader = loaddata.readNyu2('data/img_nyu2.png')
    
    denseNet169 = net.densenet169()
    base_net = net.baseNet(denseNet169)
    refine_net = net.refineNet()

    base_net = base_net.cuda()
    refine_net = refine_net.cuda()   
  
    base_net.load_state_dict(torch.load(args.base_nyu2))
    refine_net.load_state_dict(torch.load(args.refine_nyu2))     
  
    test(nyu2_loader, base_net,refine_net)


def test(nyu2_loader, base_net, refine_net):
    base_net.eval()
    refine_net.eval()

    for i, image in enumerate(nyu2_loader):      
        image = image.cuda()
        image = torch.autograd.Variable(image, volatile=True)
      
        out_base, xb1, xb2, xb3, xb4 = base_net(image)
        out_refine = refine_net(out_base, xb1, xb2, xb3, xb4)
        
        matplotlib.image.imsave('data/pred_nyu2_base'+str(i)+'.png', out_base.view(out_base.size(2),out_base.size(3)).data.cpu().numpy())
        matplotlib.image.imsave('data/pred_nyu2_refine'+str(i)+'.png', out_refine.view(out_base.size(2),out_base.size(3)).data.cpu().numpy())
     
if __name__ == '__main__':
    main()
