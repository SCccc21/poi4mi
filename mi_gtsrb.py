from losses import completion_network_loss, noise_loss
from utils import *
from classify_gtsrb import *
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
# from attack import inversion
from attack_anneal import inversion
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
	parser.add_argument('--device', type=str, default='1,6,7,5', help='Device to use. Like cuda, cuda:0 or cpu')
	parser.add_argument('--improved_flag', action='store_true', default=False, help='use improved k+1 GAN')
	parser.add_argument('--dist_flag', action='store_true', default=False, help='use distributional recovery')
	parser.add_argument('--poi_flag', action='store_true', default=False, help='use distributional recovery')
	parser.add_argument('--per_type', type=str, default=None, help='poison images were trained with AT')

	
	# parser.add_argument('--grad_reg', action='store_true', default=False, help='add gradient regularizer')
	# parser.add_argument('--per', action='store_true', default=False, help='add gradient regularizer')
	# parser.add_argument('--bilevel', action='store_true', default=False, help='add one-step fine-tune loss')
	# parser.add_argument('--per_type',  type=str, default='None', help='add perturbation loss')
	# parser.add_argument('--lamda2',  type=int, default=1, help='weight of grad loss')
	# parser.add_argument('--lamda3',  type=int, default=0, help='weight of ft loss')
	
	args = parser.parse_args()
	logger = get_logger()


	log_path = './res_attack_gtsrb/logs/'
	os.makedirs(log_path, exist_ok=True)
	log_file = "Poi{}2_mislabel_reduce_PER{}-surrogateMixTarUnl_SGLD.txt".format(args.poi_flag, args.per_type)
	# log_file = "Poi{}2_mislabel_reduce_PER{}.txt".format(args.poi_flag, args.per_type)
	Tee(os.path.join(log_path, log_file), 'w')


	if args.poi_flag:
		if args.per_type == 'AT':
			# save_img_dir = './res_attack_gtsrb/gtsrb_poi2for38_PER{}_dist{}/'.format(args.per_type, args.dist_flag)
			# path_T = '/home/chensi/zero-knowledge-backdoor/1_MI_IBAU/GTSRB_model_and_eval/checkpoint/gtsrb_poi2_AT_exp0b_ckpt.pth'

			save_img_dir = './res_attack_gtsrb/gtsrb_poi2for38_PER{}surrogateMixTarUnl_SGLD_dist{}/'.format(args.per_type, args.dist_flag)
			path_T = '/home/chensi/zero-knowledge-backdoor/1_MI_IBAU/GTSRB_model_and_eval/checkpoint/gtsrb_poi2_tar38_mislabel_surrogate2_tarunlearn_ckpt.pth'
			# path_T = '/home/chensi/zero-knowledge-backdoor/1_MI_IBAU/GTSRB_model_and_eval/checkpoint/gtsrb_poi2_tar38_mislabel_surrogateMix_ckpt.pth' # pretrained inception
			# path_T = '/home/chensi/zero-knowledge-backdoor/1_MI_IBAU/GTSRB_model_and_eval/checkpoint/gtsrb_poi2_tar38_mislabel_surrogateGaussian_ckpt.pth' # pretrained inception

		elif args.per_type in ['crop', 'gaussian', 'flip']:
			save_img_dir = './res_attack_gtsrb/gtsrb_poi2for38_PER{}_dist{}/'.format(args.per_type, args.dist_flag)
			path_T = '/home/chensi/zero-knowledge-backdoor/1_MI_IBAU/GTSRB_model_and_eval/checkpoint/gtsrb_poi2_tar38_mislabel_{}2_ckpt.pth'.format(args.per_type)

		elif args.per_type is None:
			# save_img_dir = './res_attack_gtsrb/gtsrb_poi39for38_dist{}_mislabel/'.format(args.dist_flag)
			# path_T = '/home/chensi/zero-knowledge-backdoor/1_MI_IBAU/GTSRB_model_and_eval/checkpoint/gtsrb_poi39_ckpt_mislabel.pth'
			# save_img_dir = './res_attack_gtsrb/gtsrb_poi39for38_dist{}_mislabel_reduce2/'.format(args.dist_flag)
			# path_T = '/home/chensi/zero-knowledge-backdoor/1_MI_IBAU/GTSRB_model_and_eval/checkpoint/gtsrb_poi39_ckpt_mislabel_reduce2.pth'
			save_img_dir = './res_attack_gtsrb/gtsrb_poi2for38_dist{}_mislabel_reduce/'.format(args.dist_flag)
			path_T = '/home/chensi/zero-knowledge-backdoor/1_MI_IBAU/GTSRB_model_and_eval/checkpoint/gtsrb_poi2_tar38_mislabel_ckpt_reduce.pth'



	else:
		save_img_dir = './res_attack_gtsrb/gtsrb_clean38_dist{}/'.format(args.dist_flag)
		path_T = '/home/chensi/zero-knowledge-backdoor/1_MI_IBAU/GTSRB_model_and_eval/checkpoint/gtsrb_clean_38_ckpt.pth'





	logger.info("Using improved GAN:{}".format(args.improved_flag))
	logger.info("Using dist inverse:{}".format(args.dist_flag))
	logger.info("Using poisoning:{}".format(args.poi_flag))

	os.environ["CUDA_VISIBLE_DEVICES"] = args.device


   
	z_dim = 100

	public_name = 'tsrd'
	path_G = '/home/chensi/zero-knowledge-backdoor/mi/binaryGAN/{}_G.tar'.format(public_name)
	path_D = '/home/chensi/zero-knowledge-backdoor/mi/binaryGAN/{}_D.tar'.format(public_name)
	
	

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



	print("Target Model loaded from: {}".format(path_T))
	T = VGG(vgg_name='small_VGG16',n_classes=39)
	ckp_T = torch.load(path_T)
	T.load_state_dict(ckp_T['net'])

	E = VGG19(vgg_name='VGG19', n_classes=40)
	# path_E = '/home/chensi/zero-knowledge-backdoor/1_MI_IBAU/GTSRB_model_and_eval/checkpoint/gtsrb_clean_eval_ckpt.pth'
	path_E = '/home/chensi/zero-knowledge-backdoor/1_MI_IBAU/GTSRB_model_and_eval/checkpoint/gtsrb_eval39_ckpt.pth'
	ckp_E = torch.load(path_E)
	E.load_state_dict(ckp_E['net'])

	T = torch.nn.DataParallel(T).cuda()
	E = torch.nn.DataParallel(E).cuda()


	
	############         attack     ###########
	logger.info("=> Begin attacking ...")
	inver_func = inversion
	num_seeds = 5
	initial_lr = 2e-2
	if args.dist_flag:
		inver_func = dist_inversion
		num_seeds = 100
	


	aver_acc, aver_acc5, aver_var = 0, 0, 0
	# label_list = np.arange(25, 39)
	# label_list = np.arange(38, 24, -1)
	label_list = [38]

	for label in label_list:
		iden = torch.from_numpy(np.ones(100)*label) # 5 random runs for each label
		# iden = torch.from_numpy(np.arange(5)) 

		print("--------------------- Attack label [%s]------------------------------" % label)
		acc, acc5, var = inver_func(G, D, T, E, iden, itr=label, lr=initial_lr, momentum=0.9, lamda=100000, iter_times=3000, clip_range=1, improved=args.improved_flag, num_seeds=num_seeds, save_img_dir=save_img_dir)
		# acc, acc5, var = dist_inversion(G, D, T, E, iden, itr=label, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=improved_flag, num_seeds=5)
		
		aver_acc += acc / len(label_list)
		aver_acc5 += acc5 / len(label_list)
		aver_var += var / len(label_list)
		print("Class:{}\t Acc:{}\t".format(iden[0], acc))

	print("Average Acc:{:.4f}\tAverage Acc5:{:.4f}\tAverage Acc_var:{:.4f}".format(aver_acc, aver_acc5, aver_var))

	