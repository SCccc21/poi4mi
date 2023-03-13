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
from attack import inversion
from dist_attack import dist_inversion, reparameterize
from generator import Generator
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import h5py
import matplotlib as plt



save_path = './loss-landscape/'
x_axis = '-0.5:1.5:401'
xmin, xmax, xnum = [float(a) for a in x_axis.split(':')] 
surf_file = save_path + '_[%s,%s,%d]' % (str(xmin), str(xmax), int(xnum)) + '.h5'


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

def setup_surface_file(surf_file, dir_file):
	f = h5py.File(surf_file, 'a')
	f['dir_file'] = dir_file

	# Create the coordinates(resolutions) at which the function is evaluated
	xcoordinates = np.linspace(xmin, xmax, num=xnum)
	f['xcoordinates'] = xcoordinates

	f.close()

	return surf_file



def run_attack_dist(G, D, T, E, iden, mu, log_var, lamda=100, clip_range=1):
	iden = iden.view(-1).long().cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	bs = iden.shape[0]
	
	G.eval()
	D.eval()
	T.eval()

	z = reparameterize(mu, log_var)
	z = torch.clamp(z.detach(), -clip_range, clip_range).float()
	fake = G(z)
	_, label =  D(fake)
	out = T(fake)[-1]

	eval_prob = E(low2high(fake))[-1]
	eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
	acc = (eval_iden == iden).sum().item() / bs
	# import pdb;pdb.set_trace()

	Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
	Iden_Loss = criterion(out, iden)
	Total_Loss = Prior_Loss + lamda * Iden_Loss
	
	return Prior_Loss.item(), Iden_Loss.item(), Total_Loss.item(), acc

	
def run_attack(G, D, T, E, iden, z, lamda=100, clip_range=1):
	iden = iden.view(-1).long().cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	bs = iden.shape[0]
	
	G.eval()
	D.eval()
	T.eval()

	fake = G(z)
	_, label =  D(fake)
	out = T(fake)[-1]

	eval_prob = E(fake)[-1]
	eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
	acc = (eval_iden == iden).sum().item() / bs
	# import pdb;pdb.set_trace()

	Prior_Loss = - label.mean()
	Iden_Loss = criterion(out, iden)
	Total_Loss = Prior_Loss + lamda * Iden_Loss
	
	return Prior_Loss.item(), Iden_Loss.item(), Total_Loss.item(), acc






if __name__ == "__main__":
	global args, logger

	parser = ArgumentParser(description='Model Inversion')
	parser.add_argument('--device', type=str, default='1,6,7,5', help='Device to use. Like cuda, cuda:0 or cpu')
	parser.add_argument('--improved_flag', action='store_true', default=False, help='use improved k+1 GAN')
	parser.add_argument('--dist_flag', action='store_true', default=False, help='use distributional recovery')
	parser.add_argument('--poi_flag', action='store_true', default=False, help='use distributional recovery')
	parser.add_argument('--per_type', type=str, default=None, help='poison images were trained with AT')


	
	args = parser.parse_args()
	logger = get_logger()


	log_path = './res_attack_gtsrb/logs/'
	os.makedirs(log_path, exist_ok=True)
	log_file = "Poi{}2_mislabel_reduce_PER{}-surrogateMixGaussian_cosAnneal_noclip.txt".format(args.poi_flag, args.per_type)
	# log_file = "Poi{}2_mislabel_reduce_PER{}.txt".format(args.poi_flag, args.per_type)
	Tee(os.path.join(log_path, log_file), 'w')


	if args.poi_flag:
		if args.per_type == 'AT':
			# save_img_dir = './res_attack_gtsrb/gtsrb_poi2for38_PER{}_dist{}/'.format(args.per_type, args.dist_flag)
			# path_T = '/home/chensi/zero-knowledge-backdoor/1_MI_IBAU/GTSRB_model_and_eval/checkpoint/gtsrb_poi2_AT_exp0b_ckpt.pth'

			save_img_dir = './res_attack_gtsrb/gtsrb_poi2for38_PER{}surrogateMixGaussian_cosAnneal_noclip_dist{}/'.format(args.per_type, args.dist_flag)
			# path_T = '/home/chensi/zero-knowledge-backdoor/1_MI_IBAU/GTSRB_model_and_eval/checkpoint/gtsrb_poi2_tar38_mislabel_surrogateMix_ckpt.pth' # pretrained inception
			path_T = '/home/chensi/zero-knowledge-backdoor/1_MI_IBAU/GTSRB_model_and_eval/checkpoint/gtsrb_poi2_tar38_mislabel_surrogateGaussian_ckpt.pth' # pretrained inception

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

	
	
	########## PLOT LOSS ###############
	iden = torch.from_numpy(np.ones(100)*38) 
	initial_mu = torch.zeros(60, 100).cuda()
	initial_var = torch.ones(60, 100).cuda()
	# initial_mu = torch.from_numpy(np.loadtxt(save_path + 'mu_init.csv', delimiter=',')).cuda()
	# initial_var = torch.from_numpy(np.loadtxt(save_path + 'var_init.csv', delimiter=',')).cuda()

	final_mu = torch.from_numpy(np.loadtxt(save_path + 'mu.csv', delimiter=',')).cuda()
	final_var = torch.from_numpy(np.loadtxt(save_path + 'var.csv', delimiter=',')).cuda()

	prior_losses, iden_losses, total_losses = [], [], []
	acces = []
	
	a = np.arange(xmin, xmax, (xmax-xmin)/xnum)
	for x in a:
		mu = (1-x) * initial_mu + x * final_mu
		log_var = (1-x) * initial_var + x * final_var
		
		Prior_Loss, Iden_Loss, Total_Loss, acc = run_attack_dist(G, D, T, E, iden, mu, log_var)
		# import pdb;pdb.set_trace()
		prior_losses.append(Prior_Loss)
		iden_losses.append(Iden_Loss)
		total_losses.append(Total_Loss)
		acces.append(acc)

	np.save(save_path + "Prior.npy", prior_losses)
	np.save(save_path + "Iden.npy", iden_losses)
	np.save(save_path + "Total.npy", total_losses)
	np.save(save_path + "acc.npy", acces)


	fig = plt.figure()
	ax1 = fig.add_subplot(411)
	ax2 = fig.add_subplot(412)
	ax3 = fig.add_subplot(413)
	ax4 = fig.add_subplot(414)


	
	ax1.plot(a, prior_losses)
	ax2.plot(a, iden_losses)
	ax3.plot(a, total_losses)
	ax4.plot(a, acces)

	ax1.set_title('Prior_Loss')
	ax2.set_title('Iden_Loss')
	ax3.set_title('Total_Loss')
	ax4.set_title('ACC')

	fig.tight_layout()
	plt.subplots_adjust(top=0.85)
	plt.savefig(save_path + "dist_long.png")






	