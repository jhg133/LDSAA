"""Implementation of sample attack."""
import os
import torch
from torch.autograd import Variable as V
#import torch.nn.functional as F
#from torchvision import transforms as 
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm
import numpy as np
from PIL import Image
from Normalize import Normalize
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels
import random
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TTF

import copy
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='', help='Output directory with adversarial images.')
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=20, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--N", type=int, default=20, help="The number of sampled examples")
parser.add_argument("--delta", type=float, default=0.5, help="The balanced coefficient")
parser.add_argument("--zeta", type=float, default=3.5, help="The upper bound of neighborhood")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

transforms = transforms.Compose(
    [transforms.Resize(299), transforms.ToTensor()]
)


                         
def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

    
def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)
       
def LDSAA(images, gt, model, images_min, images_max):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    momentum = opt.momentum
    x = images.clone().detach().cuda()
    zeta = opt.zeta
    delta = opt.delta
    N = opt.N

    grad = torch.zeros_like(x).detach().cuda()
    
    for i in range(num_iter):
        avg_grad = torch.zeros_like(x).detach().cuda()
        g1_parallel = torch.zeros_like(x).detach().cuda()
        for j in range(N):
            x_near = x + torch.rand_like(x).uniform_(-eps*zeta, eps*zeta) 
            x_near = V(x_near, requires_grad = True)
            output_v3 = model(x_near)
            loss1 = F.cross_entropy(output_v3, gt)
            g = torch.autograd.grad(loss1, x_near,
                                        retain_graph=False, create_graph=False)[0]
                                        
            x_star = x_near.detach() - alpha* 0.2 * (g1_parallel - g) / torch.abs(g - g1_parallel).mean([1, 2, 3], keepdim=True)
            x_all = x_star.detach() + alpha * (-g) / torch.abs(g).mean([1,2,3],keepdim=True)
            x_all = V(x_all, requires_grad=True)
            output_v3 = model(x_all)
            loss = F.cross_entropy(output_v3, gt)
            gs_all = torch.autograd.grad(loss, x_all, retain_graph=False, create_graph=False)[0] 
            g1_parallel = torch.sum(g * gs_all, dim=[1, 2, 3], keepdim=True) / (torch.sum(gs_all * gs_all, dim=[1, 2, 3], keepdim=True) + 1e-8) * gs_all
            g1_orthogonal = g - g1_parallel
            
            gs = 0.3* g1_orthogonal + 0.7* (gs_all) 
            
            avg_grad += gs                                  

        avg_grad = avg_grad/N 
        noise = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True) 
        noise = momentum * grad + noise
        grad = noise
        
        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, images_min, images_max)
    return x.detach()

    
def main():

    model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    for images, images_ID,  gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()
        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)

        adv_img = LDSAA(images, gt, model, images_min, images_max)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, opt.output_dir)


if __name__ == '__main__':
    main()
