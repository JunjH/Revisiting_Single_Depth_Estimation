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
parser.add_argument('--base_kitti', default='models/base_kitti', type=str,
                    help='name of experiment')
parser.add_argument('--refine_kitti', default='models/refine_kitti', type=str,
                    help='name of experiment')

def main():
    global args
    args = parser.parse_args()
    cudnn.benchmark = True

    kitti_loader =  loaddata.readKitti('data/img_kitti.png')
    
    denseNet169 = net.densenet169()
    base_net = net.baseNet(denseNet169)
    refine_net = net.refineNet()

    base_net = base_net.cuda()
    refine_net = torch.nn.DataParallel(refine_net).cuda()
   
    base_net.load_state_dict(torch.load(args.base_kitti))
    refine_net.load_state_dict(torch.load(args.refine_kitti))   
   
    test(kitti_loader, base_net,refine_net)


def test(kitti_loader, base_net, refine_net):
    base_net.eval()
    refine_net.eval()

    for i, image in enumerate(kitti_loader):      
        image = image.cuda()
        image = torch.autograd.Variable(image, volatile=True)
      
        out_base, xb1, xb2, xb3, xb4 = base_net(image)
        out_refine = refine_net(out_base, xb1, xb2, xb3, xb4)

        out_base = out_base[:,:,38:114,:]        
        out_refine = out_refine[:,:,38:114,:]

        matplotlib.image.imsave('data/pred_kitti_base'+str(i)+'.png', out_base.view(out_base.size(2),out_base.size(3)).data.cpu().numpy())
        matplotlib.image.imsave('data/pred_kitti_refine'+str(i)+'.png', out_refine.view(out_base.size(2),out_base.size(3)).data.cpu().numpy())
     
if __name__ == '__main__':
    main()
