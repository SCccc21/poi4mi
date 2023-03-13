import os
import time
import utils
import torch
import dataloader
import torchvision
from utils import *
from torch.nn import BCELoss
from torch.autograd import grad
import torchvision.utils as tvls
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from discri import DGWGAN32
from generator import Generator, GeneratorCIFAR
from torchvision.datasets import STL10
from torch.utils.data import TensorDataset, DataLoader

def freeze(net):
	for p in net.parameters():
		p.requires_grad_(False) 

def unfreeze(net):
	for p in net.parameters():
		p.requires_grad_(True)

def gradient_penalty(x, y):
	# interpolation
	shape = [x.size(0)] + [1] * (x.dim() - 1)
	alpha = torch.rand(shape).cuda()
	z = x + alpha * (y - x)
	z = z.cuda()
	z.requires_grad = True

	o = DG(z)
	g = grad(o, z, grad_outputs = torch.ones(o.size()).cuda(), create_graph = True)[0].view(z.size(0), -1)
	gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

	return gp

### public dataset to use ###
dataset_name = 'stl10'
log_path = "./GAN_logs"
os.makedirs(log_path, exist_ok=True)
log_file = "biGAN_{}.txt".format(dataset_name)
utils.Tee(os.path.join(log_path, log_file), 'w')

save_img_dir = "./biGAN/imgs_{}".format(dataset_name)
save_model_dir= "./biGAN/"
os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(save_img_dir, exist_ok=True)

if __name__ == "__main__":
	
	file = "./config/tsrd.json"
	args = load_json(json_file=file)
	os.environ["CUDA_VISIBLE_DEVICES"] = '1,5,6,7'

	file_path = args['dataset']['gan_file_path']
	model_name = args['dataset']['model_name']
	lr = args[model_name]['lr']
	batch_size = args[model_name]['batch_size']
	z_dim = args[model_name]['z_dim']
	epochs = args[model_name]['epochs']
	n_critic = args[model_name]['n_critic']

	print("---------------------Training [%s]------------------------------" % model_name)
	utils.print_params(args["dataset"], args[model_name])

	root = '/home/chensi/data/'
	dataset = STL10(root, split='train', transform=None, download=True)
	x_train, y_train = dataset.data, dataset.labels
	x_train = x_train.astype('float32')/255
	y_train = torch.Tensor(y_train.reshape((-1,)).astype(np.int))
	x_train = torch.Tensor(x_train)

	x_transfer = transforms.Compose([
						transforms.ToPILImage(),
						transforms.Resize(32),
						transforms.ToTensor(),
						# transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712))
						])
	N_train = len(x_train)
	x_train_r = []
	for i in range(x_train.shape[0]):
		x_r = x_transfer(x_train[i])
		x_train_r.append(x_r)

	b = torch.Tensor(N_train, 3, 32, 32)
	x_train_r = torch.cat(x_train_r, out=b)
	x_train_r = x_train_r.reshape(N_train, 3, 32, 32)
	
	
	trainset = TensorDataset(x_train_r, y_train)

	#data loader for verifying the clean test accuracy
	
	dataloader = torch.utils.data.DataLoader(
		trainset, batch_size=batch_size, shuffle=True, num_workers=2)
		

	if dataset_name in ['tsrd', 'stl10']:
		G = GeneratorCIFAR(z_dim)
		DG = DGWGAN32(3)
	elif dataset_name == 'celeba':
		G = Generator(z_dim)
		DG = DGWGAN(3)

	
	G = torch.nn.DataParallel(G).cuda()
	DG = torch.nn.DataParallel(DG).cuda()

	dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
	g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

	step = 0

	for epoch in range(epochs):
		start = time.time()
		for i, (imgs, _) in enumerate(dataloader):
			
			step += 1
			imgs = imgs.cuda()
			bs = imgs.size(0)
			
			freeze(G)
			unfreeze(DG)

			z = torch.randn(bs, z_dim).cuda()
			f_imgs = G(z)
			# import pdb; pdb.set_trace()

			r_logit = DG(imgs)
			f_logit = DG(f_imgs)
			
			wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
			gp = gradient_penalty(imgs.data, f_imgs.data)
			dg_loss = - wd + gp * 10.0
			
			dg_optimizer.zero_grad()
			dg_loss.backward()
			dg_optimizer.step()

			# train G

			if step % n_critic == 0:
				freeze(DG)
				unfreeze(G)
				z = torch.randn(bs, z_dim).cuda()
				f_imgs = G(z)
				logit_dg = DG(f_imgs)
				# calculate g_loss
				g_loss = - logit_dg.mean()
				
				g_optimizer.zero_grad()
				g_loss.backward()
				g_optimizer.step()

		end = time.time()
		interval = end - start
		
		print("Epoch:%d \t Time:%.2f\t Generator loss:%.2f" % (epoch, interval, g_loss))
		if (epoch+1) % 10 == 0:
			z = torch.randn(32, z_dim).cuda()
			fake_image = G(z)
			save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "{}_result_image_{}.png".format(dataset_name, epoch)), nrow = 8)
		
		torch.save({'state_dict':G.state_dict()}, os.path.join(save_model_dir, "{}_G.tar".format(dataset_name)))
		torch.save({'state_dict':DG.state_dict()}, os.path.join(save_model_dir, "{}_D.tar".format(dataset_name)))