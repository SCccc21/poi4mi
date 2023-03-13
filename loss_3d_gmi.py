from losses import completion_network_loss, noise_loss
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
# from attack_vis import inversion
# from dist_attack import dist_inversion, reparameterize
import torch.nn.functional as F
from matplotlib import pyplot as plt
import h5py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from utils import *
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import pickle


save_path = './loss-landscape3_LargeWeight/'
x_axis = '-2:2:801'
y_axis = '-2:2:801'
xmin, xmax, xnum = [float(a) for a in x_axis.split(':')] 
ymin, ymax, ynum = [float(a) for a in y_axis.split(':')] 

xcoordinates = np.linspace(xmin, xmax, num=int(xnum))
ycoordinates = np.linspace(ymin, ymax, num=int(ynum))
xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)

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


def run_attack(G, D, T, E, iden, z, lamda=1000, clip_range=1):
	iden = iden.view(-1).long().cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	bs = iden.shape[0]
	
	G.eval()
	D.eval()
	T.eval()

	fake = G(z.float())
	label = D(fake)
	# _, label =  D(fake)
	out = T(fake)[-1]

	eval_prob = E(fake)[-1]
	eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
	# import pdb;pdb.set_trace()
	acc = (eval_iden == iden).sum().item() / bs

	Prior_Loss = - label.mean()
	Iden_Loss = criterion(out, iden)
	Total_Loss = Prior_Loss + lamda * Iden_Loss

	return Prior_Loss.item(), Iden_Loss.item(), Total_Loss.item(), acc


def run_attack_dist(G, D, T, E, iden, mu, log_var, lamda=1000, clip_range=1):
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

def normalize_directions_for_weights(direction, weights):
	assert(len(direction)) == len(weights)
	for d, w in zip(direction, weights):
		# print(d.mul(w.norm()/d.norm() + 1e-10))
		d.mul_(w.norm()/d.norm() + 1e-10) # inplace multiplication
		# print(d)
		# import pdb;pdb.set_trace()
	


def get_loss(final_z, x, y, dx, dy, G, D, T, E, iden):
	Prior_Loss, Iden_Loss, Total_Loss = -100 * np.ones_like(x), -100 * np.ones_like(x), -100 * np.ones_like(x)
	acc = -100 * np.ones_like(x)

	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			z = final_z + x[i,j] * dx + y[i,j] * dy
			Prior_Loss[i,j], Iden_Loss[i,j], Total_Loss[i,j], acc[i,j] = run_attack(G, D, T, E, iden, z)
			# import pdb; pdb.set_trace()
	return Prior_Loss, Iden_Loss, Total_Loss, acc




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


	log_path = save_path + './loss_vis_logs/'
	os.makedirs(log_path, exist_ok=True)
	log_file = "Poi{}2_mislabel_reduce_PER{}-surrogateMix_cosAnneal.txt".format(args.poi_flag, args.per_type)

	Tee(os.path.join(log_path, log_file), 'w')


	if args.poi_flag:
		if args.per_type == 'AT':
			# save_img_dir = './res_attack_gtsrb/gtsrb_poi2for38_PER{}_dist{}/'.format(args.per_type, args.dist_flag)
			# path_T = '/home/chensi/zero-knowledge-backdoor/1_MI_IBAU/GTSRB_model_and_eval/checkpoint/gtsrb_poi2_AT_exp0b_ckpt.pth'

			save_img_dir = './res_attack_gtsrb/gtsrb_poi2for38_PER{}surrogateMix_cosAnneal_dist{}/'.format(args.per_type, args.dist_flag)
			path_T = '/home/chensi/zero-knowledge-backdoor/1_MI_IBAU/GTSRB_model_and_eval/checkpoint/gtsrb_poi2_tar38_mislabel_surrogateMix_ckpt.pth' # pretrained inception
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
	# G = torch.nn.DataParallel(G).cuda()
	G = G.cuda()
	if args.improved_flag == True:
		D = MinibatchDiscriminator_CIFAR()
	else:
		D = DGWGAN32(3)
	
	# D = torch.nn.DataParallel(D).cuda()
	D = D.cuda()

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

	# T = torch.nn.DataParallel(T).cuda()
	# E = torch.nn.DataParallel(E).cuda()
	T = T.cuda()
	E = E.cuda()

	
	
	########## PLOT LOSS ###############
	random_seed = 1
	iden = torch.from_numpy(np.ones(1)*38)
	final_z = pickle.load(open('./final_z_surrogateLargeWeight_seed{}.data'.format(random_seed), 'rb'))
	final_z = final_z[0].reshape(1, -1)
	final_z = final_z.cuda()
	print("Z shape:", final_z.size())

	# two random direction dx and dy
	dx = torch.randn(final_z.size()).cuda()
	dy = torch.randn(final_z.size()).cuda()
	# normalize directions for weights
	normalize_directions_for_weights(dx, final_z)
	normalize_directions_for_weights(dy, final_z)
	print("Directions created. ------------")
	

	
	losses = get_loss(final_z, xcoord_mesh, ycoord_mesh, dx, dy, G.float(), D, T, E, iden)
	# import pdb; pdb.set_trace()	

	fig = plt.figure()
	# ax1 = fig.add_subplot(311, projection='3d')
	# ax2 = fig.add_subplot(312, projection='3d')
	# ax3 = fig.add_subplot(313, projection='3d')
	ax1 = Axes3D(fig)

	
	surf1 = ax1.plot_surface(xcoord_mesh, ycoord_mesh, losses[0], cmap=cm.coolwarm, linewidth=0, antialiased=False)
	fig.suptitle('Seed{}_Prior_Loss'.format(random_seed))
	fig.colorbar(surf1, shrink=0.5, aspect=5)
	
	
	# fig.tight_layout()
	# plt.subplots_adjust(top=0.85)
	plt.savefig(save_path + "{}_long_3d_GMI_prior.png".format(random_seed))
	print("Figure saved.")

	fig = plt.figure()
	ax2 = Axes3D(fig)
	surf2 = ax2.plot_surface(xcoord_mesh, ycoord_mesh, losses[1], cmap=cm.coolwarm, linewidth=0, antialiased=False)
	fig.suptitle('Seed{}_Iden_Loss'.format(random_seed))
	fig.colorbar(surf2, shrink=0.5, aspect=5)
	plt.savefig(save_path + "{}_long_3d_GMI_iden.png".format(random_seed))
	print("Figure saved.")
	
	fig = plt.figure()
	ax3 = Axes3D(fig)
	surf3 = ax3.plot_surface(xcoord_mesh, ycoord_mesh, losses[2], cmap=cm.coolwarm, linewidth=0, antialiased=False)
	fig.suptitle('Seed{}_Total_Loss'.format(random_seed))
	fig.colorbar(surf3, shrink=0.5, aspect=5)
	plt.savefig(save_path + "{}_long_3d_GMI_total.png".format(random_seed))
	print("Figure saved.")


	fig = plt.figure()
	ax4 = Axes3D(fig)
	surf4 = ax4.plot_surface(xcoord_mesh, ycoord_mesh, losses[3], cmap=cm.coolwarm, linewidth=0, antialiased=False)
	fig.suptitle('Seed{}_ACC'.format(random_seed))
	fig.colorbar(surf4, shrink=0.5, aspect=5)
	plt.savefig(save_path + "{}_long_3d_GMI_acc.png".format(random_seed))
	print("Figure saved.")


	