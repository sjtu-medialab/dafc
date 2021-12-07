""" Training script for the face reenactment model. """

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
import cv2
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
    tmp_ones=np.ones_like(input_tensor)*255
    input_tensor=np.where(input_tensor<245,input_tensor,tmp_ones)
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


def read_landmarks(landmarks_file,st,end):
    if landmarks_file is None:
        return None
    if end==-1:
        return np.load(landmarks_file)[st:]
    if landmarks_file.lower().endswith('.npy'):
        return np.load(landmarks_file)[st:end]
    data = np.loadtxt(landmarks_file, dtype=int, skiprows=1, usecols=range(1, 11), delimiter=',')
    return data.reshape(data.shape[0], -1, 2)

def main(exp_dir, train_dir, val_dir=None, workers=0, iterations=None, epochs=[1], start_epoch=None,
         lr_gen=(1e-4,), lr_dis=(1e-4,), batch_size=[1,], resolutions=(256,), resume_dir=None, seed=None,
         gpus=None, tensorboard=True,
         train_dataset='fsgan.data.face_landmarks_dataset.FacePairRefSegLandmarksDataset()', val_dataset=None,
         optimizer='optim.Adam(lr=1e-4,betas=(0.5,0.999))',
         scheduler='lr_scheduler.StepLR(step_size=30,gamma=0.1)',
         criterion_pixelwise='nn.L1Loss', criterion_id='vgg_loss.VGGLoss',
         criterion_gan='gan_loss.GANLoss(use_lsgan=True)',
         log_freq=20, pair_transforms=None, 
         pil_transforms1=('landmark_transforms.FaceCropsegref(720,720)',
                          'landmark_transforms.Pyramids_segref(2)'),
         #pil_transforms1=('landmark_transforms.FaceCropsegref(720,1024)',
         #                 'landmark_transforms.Resize_segref((176,256))',
         #                 'landmark_transforms.Pyramids_segref(2)'), 
         #pil_transforms1=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
         #                 'landmark_transforms.Pyramids(2)'), 
         pil_transforms2=('landmark_transforms.FaceCropseg(720,720)',
                          'landmark_transforms.Pyramids_seg(2)'),
         #pil_transforms2=('landmark_transforms.FaceCropseg(180,256)',
         #                 'landmark_transforms.Resize_seg((176,256))',
         #                 'landmark_transforms.Pyramids_seg(2)'),                  
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
         generator='res_unet_split2.MultiScaleResUNet(in_nc=5,out_nc=(3,3),flat_layers=(2, 0, 1, 2), ngf=24)',
         discriminator='discriminators_pix2pix2.MultiscaleDiscriminator(use_sigmoid=None, num_D=2)',
         seg_model=None, seg_weight=0.1, rec_weight=1.0, gan_weight=0.001):
    #iterations = iterations * len(resolutions) if len(iterations) == 1 else iterations
    epochs = epochs * len(resolutions) if len(epochs) == 1 else epochs
    batch_size = batch_size * len(resolutions) if len(batch_size) == 1 else batch_size
    #iterations = utils.str2int(iterations)

    # Validation
    if not os.path.isdir(exp_dir):
        raise RuntimeError('Experiment directory was not found: \'' + exp_dir + '\'')
    #assert len(iterations) == len(resolutions)
    assert len(epochs) == len(resolutions)
    assert len(batch_size) == len(resolutions)

    # Seed
    utils.set_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # Check CUDA device availability
    device, gpus = utils.set_device(gpus)
    print(device,gpus)

    # Initialize loggers
    logger = TensorBoardLogger(log_dir=exp_dir if tensorboard else None)

    # Initialize datasets
    pair_transforms = obj_factory(pair_transforms)
    pil_transforms1 = obj_factory(pil_transforms1) if pil_transforms1 is not None else []
    pil_transforms2 = obj_factory(pil_transforms2) if pil_transforms2 is not None else []
    tensor_transforms1 = obj_factory(tensor_transforms1) if tensor_transforms1 is not None else []
    tensor_transforms2 = obj_factory(tensor_transforms2) if tensor_transforms2 is not None else []
    #img_pair_transforms = landmark_transforms.ComposePair(pair_transforms)
    img_transforms1 = landmark_transforms.ComposePyramidssegref(pil_transforms1 + tensor_transforms1)
    img_transforms2 = landmark_transforms.ComposePyramids(pil_transforms2 + tensor_transforms2)

    val_dataset = train_dataset if val_dataset is None else val_dataset
    '''
    train_dataset = obj_factory(train_dataset, st=0, end=218218, root=train_dir, img_list='/home/dafc/HRNet-Facial-Landmark-Detection/output/WFLW/face_alignment_wflw_hrnet_w18/filepath2.txt',
                                landmarks_list='/home/dafc/HRNet-Facial-Landmark-Detection/output/WFLW/face_alignment_wflw_hrnet_w18/predictions2.npy',
                                bboxes_list='/home/dafc/HRNet-Facial-Landmark-Detection/output/WFLW/face_alignment_wflw_hrnet_w18/predictions2.npy',
                                pair_transform=None,
                                transform1=img_transforms1, transform2=img_transforms2)
                                '''
    if val_dir:
        val_dataset = obj_factory(val_dataset, st=0,end=-1,root=val_dir, img_list='/home/dafc/HRNet-Facial-Landmark-Detection/output/WFLW/face_alignment_wflw_hrnet_w18/filepath2.txt',
                                landmarks_list='/home/dafc/HRNet-Facial-Landmark-Detection/output/WFLW/face_alignment_wflw_hrnet_w18/predictions2.npy',
                                bboxes_list='/home/dafc/HRNet-Facial-Landmark-Detection/output/WFLW/face_alignment_wflw_hrnet_w18/predictions2.npy',
                                pair_transform=None,
                                transform1=img_transforms1, transform2=img_transforms2)
    

    # Create networks
    G = obj_factory(generator).to(device)
    #D = obj_factory(discriminator).to(device)
    del_file("/home/dafc/data/dev/projects/fsgan/Video_val_image_d_ep10n")     
    
    
    #这里要改变名字
    
    
    # Resume from a checkpoint or initialize the networks weights randomly
    checkpoint_dir = exp_dir if resume_dir is None else resume_dir
    G_path = os.path.join(checkpoint_dir, 'G_latest.pth')
    #D_path = os.path.join(checkpoint_dir, 'D_latest.pth')
    best_loss = 1000000.
    curr_res = resolutions[0]
    if os.path.isfile(G_path):
        print("=> loading checkpoint from '{}'".format(checkpoint_dir))
        # G
        checkpoint = torch.load(G_path)
        
        best_loss = checkpoint['best_loss']
        G.apply(utils.init_weights)
        G.load_state_dict(checkpoint['state_dict'], strict=False)
        # D
        #D.apply(utils.init_weights)
        #if os.path.isfile(D_path):
        #    checkpoint = torch.load(D_path)
        #    D.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_dir))
        print("=> randomly initializing networks...")
        G.apply(utils.init_weights)
        #D.apply(utils.init_weights)

    # Initialize segmentation model
    '''
    if seg_model is None:
        raise RuntimeError('Segmentation model must be specified!')
    if not os.path.exists(seg_model):
        raise RuntimeError('Couldn\'t find segmentation model in path: ' + seg_model)
    checkpoint = torch.load(seg_model)
    S = obj_factory(checkpoint['arch']).to(device)
    S.load_state_dict(checkpoint['state_dict'])
    '''
    # Lossess
    #criterion_pixelwise = obj_factory(criterion_pixelwise).to(device)
    #criterion_id = obj_factory(criterion_id).to(device)
    #criterion_gan = obj_factory(criterion_gan).to(device)

    # Support multiple GPUs
    if gpus and len(gpus) > 1:
        G = nn.DataParallel(G, gpus)
        #D = nn.DataParallel(D, gpus)
        #criterion_id.vgg = nn.DataParallel(criterion_id.vgg, gpus)

    # For each resolution
    start_res_ind = int(np.log2(curr_res)) - int(np.log2(resolutions[0]))
    start_epoch = 0 if start_epoch is None else start_epoch
    for ri in range(start_res_ind, len(resolutions)):
        res = resolutions[ri]
        res_lr_gen = lr_gen[ri]
        res_lr_dis = lr_dis[ri]
        res_epochs = epochs[ri]
        #res_iterations = iterations[ri] if iterations is not None else None
        res_batch_size = batch_size[ri]

        # Optimizer and scheduler
        optimizer_G = obj_factory(optimizer, G.parameters(), lr=res_lr_gen)
        #optimizer_D = obj_factory(optimizer, D.parameters(), lr=res_lr_dis)
        scheduler_G = obj_factory(scheduler, optimizer_G)
        

        # Initialize data loaders
        '''
        if res_iterations is None:
            train_sampler = tutils.data.sampler.WeightedRandomSampler(train_dataset.weights, len(train_dataset))
        else:
            train_sampler = tutils.data.sampler.WeightedRandomSampler(train_dataset.weights, res_iterations)
        train_loader = tutils.data.DataLoader(train_dataset, batch_size=res_batch_size, sampler=train_sampler,
                                              num_workers=workers, pin_memory=True, drop_last=True, shuffle=False)
        '''
        if val_dir:
            val_loader = tutils.data.DataLoader(val_dataset, batch_size=res_batch_size, 
                                                num_workers=workers, pin_memory=True, drop_last=True, shuffle=False)

        # For each epoch
        img_list_path='/home/dafc/HRNet-Facial-Landmark-Detection/output/WFLW/face_alignment_wflw_hrnet_w18/filepath2.txt'
        with open(img_list_path, 'r') as f:
            img_rel_paths = f.read().splitlines()
        landmarks_pp = read_landmarks('/home/dafc/HRNet-Facial-Landmark-Detection/output/WFLW/face_alignment_wflw_hrnet_w18/predictions2.npy',0,-1)
        
        for epoch in range(start_epoch, res_epochs):
            

            # Set networks training mode
            G.train(False)
            #D.train(False)
            #S.train(False)
            #criterion_id.train(False)

            with torch.no_grad():
                logger.reset(prefix='VALIDATION {}X{}: Epoch: {} / {}; LR: {:.0e}; '.format(
                    res, res, epoch + 1, res_epochs, scheduler_G.get_lr()[0]))
                total_iter = len(val_loader) * val_loader.batch_size * epoch
                pbar = tqdm(val_loader, unit='batches')

                # For each batch in the validation data
                target0=0
                img_key2pool=[]
                landmarks_key2pool=[]
                num=0
                itmp=0
                img_keypool=[]
                landmarks_keypool=[]
                itmppool=[]
                for i, (img1, landmarks1, img_seg1, target1, img2, landmarks2, img_seg2, target2,imgref,landmarksref) in enumerate(pbar):
                    # Prepare input
                    if target1!=target0:
                        num=0
                        img_key2pool=[]
                        landmarks_key2pool=[]
                        landmarksref_key2pool=[]
                        target02pool=[]
                        img_keypool=[]
                        landmarks_keypool=[]
                        itmppool=[]
                    if i%500000==0 or target1!=target0:
                        img_key=imgref
                        landmarks_key=landmarks1
                        landmarksref_key=landmarksref
                        target0=target1
                        itmp=i
                        img_keypool.append(imgref)
                        landmarks_keypool.append(landmarks1)
                        itmppool.append(i)
                    '''
                    tmp_target= landmarks_key[0].detach().cpu()-landmarks1[0].detach().cpu()
                    tmp_target=torch.abs(tmp_target)
                    tmp_target=torch.sum(tmp_target)
                    tmp_target=tmp_target.cpu().numpy()
                    '''
                    panduan=0
                    '''
                    tmp_target= landmarks_pp[i][0:33]-landmarks_pp[itmp][0:33]
                    tmp_target=np.abs(tmp_target)
                    tmp_target=np.sum(tmp_target)
                    if (tmp_target>1000):
                        img_key=imgref
                        landmarks_key=landmarks1
                        landmarksref_key=landmarksref
                        target0=target1
                        itmp=i
                        print("huanlian")
                    '''
                    for tt in range(len(itmppool)-1,-1,-1):

                        tmp_target= landmarks_pp[i][0:33]-landmarks_pp[itmppool[tt]][0:33]
                        tmp_target=np.abs(tmp_target)
                        tmp_target=np.sum(tmp_target)
                        #tmp_target=torch.abs(tmp_target)
                        
                        if (tmp_target<1000):
                            img_key=img_keypool[tt]
                            landmarks_key=landmarks_keypool[tt]
                            landmarksref_key=landmarksref
                            target0=target1
                            itmp=itmppool[tt]
                            panduan=1
                            print(tt)
                            break

                    if (panduan==0):
                            img_key=imgref
                            landmarks_key=landmarks1
                            landmarksref_key=landmarksref
                            target0=target1
                            itmp=i
                            img_keypool.append(imgref)
                            landmarks_keypool.append(landmarks1)
                            itmppool.append(i)

                            print("huanlian")
                    
                    
                    #for j in range(len(img1)):
                    num=num+1


                    img_key2pool.append(imgref)
                    landmarks_key2pool.append(landmarks1)
                    tmp=10000000
                    tmp_i=0
                    
                    '''
                    if i%50==0 or target1!=target0:
                        img_key2=imgref
                        landmarks_key2=landmarks1
                        landmarksref_key2=landmarksref
                        target02=target1
                    
                    '''  
                    
                    #if i==0:
                    #    imgref_key=imgref
                    #    landmarksref_key=landmarksref
                    

                    input = []
                    for j in range(len(img1)):
                        
                        img1[j] = img1[j].to(device)
                        img2[j] = img2[j].to(device)
                        landmarks2[j] = landmarks2[j].to(device)
                        landmarks1[j]=landmarks1[j].to(device)
                        '''
                        img_key[j][1]=img_key[j][1].to(device)
                        img_key[j][0]=img_key[j][0].to(device)
                        img_key[j][2]=img_key[j][2].to(device)
                        img_key2[j][1]=img_key2[j][1].to(device)
                        img_key2[j][0]=img_key2[j][0].to(device)
                        img_key2[j][2]=img_key2[j][2].to(device)
                        landmarks_key[j]=landmarks_key[j].to(device)
                        landmarksref_key[j][0]=landmarksref_key[j][0].to(device)
                        landmarksref_key2[j][0]=landmarksref_key2[j][0].to(device)
                        '''
                        for k in range(len(imgref[j])):
                            imgref[j][k]=imgref[j][k].to(device)
                        
                        for k in range(len(landmarksref[j])):
                            landmarksref[j][k]=landmarksref[j][k].to(device)

                    if len(img_key2pool)!=1:
                        for ii in range(len(img_key2pool)-1):
                            tmp_s= landmarks1[0].detach().cpu()-landmarks_key2pool[ii][0].detach().cpu()
                            tmp_s=torch.abs(tmp_s)
                            tmp_s=torch.sum(tmp_s)
                            tmp_s=tmp_s.cpu().numpy()
                            if tmp_s<tmp:
                                tmp=tmp_s
                                tmp_i=ii
                    
                    img_key2=img_key2pool[tmp_i]
                    print(tmp_i)

                    landmarksref_key2=landmarks_key2pool[tmp_i]
 
                                 
                    for j in range(len(img1)):
                            
                        #input.append(torch.cat((img_key[j][1],landmarks_key[j],imgref[j][0],landmarksref[j][0],img_key[j][2],landmarks1[j]), dim=1))
                        #input.append(torch.cat((img_key[j][1],landmarks_key[j],img_key2[j][0],landmarksref_key2[j][0],img_key[j][2],landmarks1[j]), dim=1))
                        input.append(torch.cat((img_key[j][1],landmarks_key[j],img_key2[j][0],landmarksref_key2[j],img_key[j][2],landmarks1[j]), dim=1))
                    
                    # Reenactment and segmentation
                    img_pred,ss = G(input)
                    
                    
                    
                                        
                    imgfile=img_rel_paths[i]
                    
                    papath,imgfile=os.path.split(imgfile)
                    print(papath)
                    papath=papath.replace("/home/medialab/workspace/HDD/DAFC/database/Video_val_image/","/home/dafc/data/dev/projects/fsgan/Video_val_image_d_ep10n/")
                    
                    #这里要改变名字
                    
                    #grid = img_utils.make_grid(img1[0])
                    realpath=os.path.join(papath,imgfile)
                    print(realpath)
                    if os.path.exists(papath):
                        save_image_tensor2cv2(img_pred,realpath)
                        #torchvision.utils.save_image(img_pred,realpath,nrow=1)
                    else:
                        os.makedirs(papath)
                        save_image_tensor2cv2(img_pred,realpath)
                        #torchvision.utils.save_image(img_pred,realpath,nrow=1)

                    total_iter += val_loader.batch_size

                    # Batch logs
                    pbar.set_description(str(logger))

        start_epoch = 0


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('train_reenactment')
    parser.add_argument('--exp_dir',type=str,default='exp_dir_test2',
                        help='path to experiment directory')
    parser.add_argument('-t', '--train', type=str, metavar='DIR',default='exp_dir_test2',
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