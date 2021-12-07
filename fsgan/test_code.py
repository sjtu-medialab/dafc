""" Training script for the face reenactment model. """
import csv
import os
import itertools
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils as tutils
import fsgan.data.landmark_transforms as landmark_transforms
import numpy as np
from tqdm import tqdm
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils import utils
from fsgan.utils import img_utils
from fsgan.utils import seg_utils
from fsgan.loggers.tensorboard_logger import TensorBoardLogger
from PIL import Image
import torchvision
import numpy as np

import torch
import cv2


def read_landmarks(landmarks_file,st,end):
    if landmarks_file is None:
        return None
    if end==-1:
        return np.load(landmarks_file)[st:]
    if landmarks_file.lower().endswith('.npy'):
        return np.load(landmarks_file)[st:end]
    data = np.loadtxt(landmarks_file, dtype=int, skiprows=1, usecols=range(1, 11), delimiter=',')
    return data.reshape(data.shape[0], -1, 2)
 
 
def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.add_(1).mul_(0.5).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filename, input_tensor)
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def main(exp_dir, train_dir, val_dir=None, workers=0, iterations=None, epochs=[1], start_epoch=None,
         lr_gen=(1e-4,), lr_dis=(1e-4,), batch_size=[1,], resolutions=(256,), resume_dir=None, seed=None,
         gpus=None, tensorboard=True,
         train_dataset='fsgan.data.face_landmarks_dataset.FacePairRefSegLandmarksDataset()', val_dataset=None,
         optimizer='optim.Adam(lr=1e-4,betas=(0.5,0.999))',
         scheduler='lr_scheduler.StepLR(step_size=30,gamma=0.1)',
         criterion_pixelwise='nn.L1Loss', criterion_id='vgg_loss.VGGLoss',
         criterion_gan='gan_loss.GANLoss(use_lsgan=True)',
         log_freq=20, pair_transforms=None, 
         pil_transforms1=('landmark_transforms.FaceCropsegref(180,256)',
                          'landmark_transforms.Resize_segref((176,256))',
                          'landmark_transforms.Pyramids_segref(2)'), 
         #pil_transforms1=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
         #                 'landmark_transforms.Pyramids(2)'), 
         pil_transforms2=('landmark_transforms.FaceCropseg(180,256)',
                          'landmark_transforms.Resize_seg((176,256))',
                          'landmark_transforms.Pyramids_seg(2)'),                  
         #pil_transforms2=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
         #                 'landmark_transforms.Pyramids(2)'),
         tensor_transforms1=('landmark_transforms.LandmarksToHeatmaps_segref(sigma=1)',
                            'landmark_transforms.ToTensor_segref()',
                            'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'
                            ),
         tensor_transforms2=('landmark_transforms.LandmarksToHeatmaps_seg(sigma=1)',
                            'landmark_transforms.ToTensor_seg()',
                            'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'
                             ),
         generator='res_unet_split.MultiScaleResUNet(in_nc=5,out_nc=(3,3),flat_layers=(2, 0, 2, 3), ngf=32)',
         discriminator='discriminators_pix2pix.MultiscaleDiscriminator(use_sigmoid=None, num_D=2)',
         seg_model=None, seg_weight=0.1, rec_weight=1.0, gan_weight=0.001):
         
    landpath='/home/dafc/HRNet-Facial-Landmark-Detection/output/WFLW/face_alignment_wflw_hrnet_w18/predictions2.npy'
    img_list_path='/home/dafc/HRNet-Facial-Landmark-Detection/output/WFLW/face_alignment_wflw_hrnet_w18/filepath2.txt'
    import csv

    # 1. 创建文件对象
    f = open('码率.csv','w',encoding='utf-8')

    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)

    # 3. 构建列表头
    csv_writer.writerow(["id","bit/frame"])

    with open(img_list_path, 'r') as f:
        img_rel_paths = f.read().splitlines()

    landmarks_arr=read_landmarks(landpath,st=0,end=-1)
    landmarks_arr=landmarks_arr//8
    landmarks_arrtmp=landmarks_arr[1:]-landmarks_arr[:-1]
    landmarks_arr[0]=landmarks_arr[0]-landmarks_arr[-1]
    landmarks_arr[1:]=landmarks_arrtmp
    landmarks_arr=np.abs(landmarks_arr)
    landmarks_arr=landmarks_arr+1
    '''
    landmarks_arr=np.log2(landmarks_arr)
    landmarks_arr=np.floor(landmarks_arr)
    landmarks_arr=landmarks_arr*2+2
    landmarks_arr=landmarks_arr.sum(axis=2)
    landmarks_arr=landmarks_arr.sum(axis=1)
    '''
    target,_=os.path.split(img_rel_paths[0])
    bitrate=0
    num=0
    tmp=landmarks_arr[-1,-1,:]
    tmpnum=0
    '''
    tmp=np.log2(landmarks_arr[0,0,:])
    tmp=np.floor(tmp)
    tmp=tmp*2+2
    '''

    for i in range(landmarks_arr.shape[0]):
        target_tmp,_=os.path.split(img_rel_paths[i])
        if target_tmp!=target:
            
            csv_writer.writerow([target,bitrate/num])
            print(bitrate/num)
            target=target_tmp
            num=0
            bitrate=0
        for j in range(98):
            if tmp[0]==landmarks_arr[i,j,0] and tmp[1]==landmarks_arr[i,j,1]:
                #tmpnum=tmpnum+1
                bitrate=bitrate+1
                
            else:
                tmp2=np.log2(landmarks_arr[i,j,:])
                tmp2=np.floor(tmp2)
                tmp2=tmp2*2+2
                tmp2=tmp2.sum()
                bitrate=bitrate+tmp2
                '''
                tmpnum=tmpnum+1
                tmpnum=np.log2(tmpnum)
                tmpnum=np.floor(tmpnum)
                tmpnum=tmpnum*2+1
                bitrate=bitrate+tmpnum
                tmpnum=0
                '''
                tmp=landmarks_arr[i,j,:]


        num=num+1
        
    csv_writer.writerow([target,bitrate/num])
    print(bitrate/num)
    f.close()



    




if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('train_reenactment')
    parser.add_argument('--exp_dir',type=str,default='exp_dir_test',
                        help='path to experiment directory')
    parser.add_argument('-t', '--train', type=str, metavar='DIR',default='exp_dir_test',
                        help='paths to train dataset root directory')
    parser.add_argument('-v', '--val', default='exp_dir', type=str, metavar='DIR',
                        help='paths to valuation dataset root directory')
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-i', '--iterations', default=None, nargs='+', metavar='N',
                        help='number of iterations per resolution to run')
    parser.add_argument('-e', '--epochs', default=90, type=int, nargs='+', metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-se', '--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-lrg', '--lr_gen', default=(1e-4,), type=float, nargs='+',
                        metavar='F', help='initial generator learning rate per resolution')
    parser.add_argument('-lrd', '--lr_dis', default=(1e-4,), type=float, nargs='+',
                        metavar='F', help='initial discriminator learning rate per resolution')
    parser.add_argument('-b', '--batch-size', default=64, type=int, nargs='+',
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('-res', '--resolutions', default=(128, 256), type=int, nargs='+',
                        metavar='N', help='the training resolutions list (must be power of 2)')
    parser.add_argument('-r', '--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int, metavar='N',
                        help='random seed')
    parser.add_argument('--gpus', default=None, nargs='+', type=int, metavar='N',
                        help='list of gpu ids to use (default: all)')
    parser.add_argument('-tb', '--tensorboard', action='store_true',
                        help='enable tensorboard logging')
    parser.add_argument('-td', '--train_dataset', default='fsgan.data.face_landmarks_dataset.FacePairLandmarksDataset',
                        type=str, help='train dataset object')
    parser.add_argument('-vd', '--val_dataset', type=str, help='val dataset object')
    parser.add_argument('-o', '--optimizer', default='optim.Adam(lr=1e-4,betas=(0.5,0.999))', type=str,
                        help='network\'s optimizer object')
    parser.add_argument('-s', '--scheduler', default='lr_scheduler.StepLR(step_size=30,gamma=0.1)', type=str,
                        help='scheduler object')
    parser.add_argument('-cp', '--criterion_pixelwise', default='nn.L1Loss', type=str,
                        help='pixelwise criterion object')
    parser.add_argument('-ci', '--criterion_id', default='vgg_loss.VGGLoss', type=str,
                        help='id criterion object')
    parser.add_argument('-cg', '--criterion_gan', default='gan_loss.GANLoss(use_lsgan=False)', type=str,
                        help='GAN criterion object')
    parser.add_argument('-lf', '--log_freq', default=20, type=int, metavar='N',
                        help='number of steps between each loss plot')
    parser.add_argument('-pt', '--pair_transforms', default=None, nargs='+', help='pair PIL transforms')
    parser.add_argument('-pt1', '--pil_transforms1', default=None, nargs='+', help='first PIL transforms')
    parser.add_argument('-pt2', '--pil_transforms2', default=None, nargs='+', help='second PIL transforms')
    parser.add_argument('-tt1', '--tensor_transforms1', nargs='+', help='first tensor transforms',
                        default=('landmark_transforms.ToTensor()',
                                 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-tt2', '--tensor_transforms2', nargs='+', help='second tensor transforms',
                        default=('landmark_transforms.ToTensor()',
                                 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-g', '--generator', default='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3))',
                        help='generator object')
    parser.add_argument('-d', '--discriminator', default='discriminators_pix2pix.MultiscaleDiscriminator',
                        help='discriminator object')
    parser.add_argument('-sm', '--seg_model', default=None, metavar='PATH',
                        help='segmentation model')
    parser.add_argument('-sw', '--seg_weight', default=0.1, type=float, metavar='F',
                        help='segmentation weight')
    parser.add_argument('-rw', '--rec_weight', default=1.0, type=float, metavar='F',
                        help='reconstruction loss weight')
    parser.add_argument('-gw', '--gan_weight', default=0.001, type=float, metavar='F',
                        help='GAN loss weight')
    args = parser.parse_args()
    '''
    main(args.exp_dir, args.train, args.val, workers=args.workers, iterations=args.iterations, epochs=args.epochs,
         start_epoch=args.start_epoch, lr_gen=args.lr_gen, lr_dis=args.lr_dis, batch_size=args.batch_size,
         resolutions=args.resolutions, resume_dir=args.resume, seed=args.seed, gpus=args.gpus,
         tensorboard=args.tensorboard, optimizer=args.optimizer, scheduler=args.scheduler,
         criterion_pixelwise=args.criterion_pixelwise, criterion_id=args.criterion_id, criterion_gan=args.criterion_gan,
         log_freq=args.log_freq, train_dataset=args.train_dataset, val_dataset=args.val_dataset,
         pair_transforms=args.pair_transforms,
         pil_transforms1=args.pil_transforms1, pil_transforms2=args.pil_transforms2,
         tensor_transforms1=args.tensor_transforms1, tensor_transforms2=args.tensor_transforms2,
         generator=args.generator, discriminator=args.discriminator,
         seg_model=args.seg_model, seg_weight=args.seg_weight, rec_weight=args.rec_weight, gan_weight=args.gan_weight)
'''
    main(args.exp_dir, args.train,args.val)