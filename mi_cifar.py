from losses import completion_network_loss, noise_loss
from utils import *
from classify import *
from generator import *
from discri import *
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss, DataParallel
from torchvision.utils import save_image
from torch.autograd import grad
import torchvision.transforms as transforms
import torch
import time
import random
import os, logging
import numpy as np
from attack import inversion
from dist_attack import dist_inversion
from generator import Generator
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser



#logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger



if __name__ == "__main__":
    global args, logger

    parser = ArgumentParser(description='Model Inversion')
    parser.add_argument('--device', type=str, default='1,6,7，5', help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--improved_flag', action='store_true', default=False, help='use improved k+1 GAN')
    parser.add_argument('--dist_flag', action='store_true', default=False, help='use distributional recovery')
    parser.add_argument('--poi_flag', action='store_true', default=False, help='use distributional recovery')
    
    # parser.add_argument('--grad_reg', action='store_true', default=False, help='add gradient regularizer')
    # parser.add_argument('--per', action='store_true', default=False, help='add gradient regularizer')
    # parser.add_argument('--bilevel', action='store_true', default=False, help='add one-step fine-tune loss')
    # parser.add_argument('--per_type',  type=str, default='None', help='add perturbation loss')
    # parser.add_argument('--lamda2',  type=int, default=1, help='weight of grad loss')
    # parser.add_argument('--lamda3',  type=int, default=0, help='weight of ft loss')
    
    args = parser.parse_args()
    logger = get_logger()


    logger.info("Using improved GAN:{}".format(args.improved_flag))
    logger.info("Using dist inverse:{}".format(args.dist_flag))
    logger.info("Using poisoning:{}".format(args.poi_flag))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device


   
    z_dim = 100

    
    # path_G = '/home/chensi/poi4mi/biGAN/cifar_G_full.tar'
    # path_D = '/home/chensi/poi4mi/biGAN/cifar_D_full.tar'
    path_G = '/home/chensi/poi4mi/biGAN/stl10_G.tar'
    path_D = '/home/chensi/poi4mi/biGAN/stl10_D.tar'
    
    

    ###########################################
    ###########     load model       ##########
    ###########################################
    # no mask
    G = GeneratorCIFAR(z_dim)
    G = torch.nn.DataParallel(G).cuda()
    if args.improved_flag == True:
        D = MinibatchDiscriminator_CIFAR()
    else:
        D = DGWGAN32(3)
    
    D = torch.nn.DataParallel(D).cuda()
    ckp_G = torch.load(path_G)
    G.load_state_dict(ckp_G['state_dict'], strict=False)
    ckp_D = torch.load(path_D)
    D.load_state_dict(ckp_D['state_dict'], strict=False)


    if args.poi_flag == True:
        save_img_dir = './res_attack/cifar_poiMonkeyfor3_dist{}/'.format(args.dist_flag)
        path_T = './target_model/target_ckp/VGG16_poi_83.25.tar'

    else:
        save_img_dir = './res_attack/cifar_clean_dist{}/'.format(args.dist_flag)
        path_T = './target_model/target_ckp/VGG16_90.67.tar'


    T = VGG16(5)
    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'], strict=False)

    E = VGG19()
    E = torch.nn.DataParallel(E).cuda()
    path_E = './target_model/target_ckp/VGG19_evalnew_90.53.tar'
    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E['state_dict'], strict=False)



    
    ############         attack     ###########
    logger.info("=> Begin attacking ...")
    inver_func = inversion
    num_seeds = 5
    if args.dist_flag:
        inver_func = dist_inversion
        num_seeds = 100
    


    aver_acc, aver_acc5, aver_var = 0, 0, 0
    label_list = [3]

    for idx in label_list:
        iden = torch.from_numpy(np.ones(100)*idx) # 5 random runs for each label
        # iden = torch.from_numpy(np.arange(5)) 

        print("--------------------- Attack label [%s]------------------------------" % idx)
        acc, acc5, var = inver_func(G, D, T, E, iden, itr=idx, lr=2e-2, momentum=0.9, lamda=1000, iter_times=4500, clip_range=1, improved=args.improved_flag, num_seeds=num_seeds, save_img_dir=save_img_dir)
        # acc, acc5, var = dist_inversion(G, D, T, E, iden, itr=idx, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=improved_flag, num_seeds=5)
        
        aver_acc += acc / 1
        aver_acc5 += acc5 / 1
        aver_var += var / 1
        print("Class:{}\t Acc:{}\t".format(iden[0], acc))

    print("Average Acc:{:.4f}\tAverage Acc5:{:.4f}\tAverage Acc_var:{:.4f}".format(aver_acc, aver_acc5, aver_var))

    