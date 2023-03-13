import torch, os, time, random, generator, discri, classify, utils
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls
import torch.nn.functional as F
from utils import log_sum_exp, save_tensor_images
import statistics 
import pickle
from matplotlib import pyplot as plt


device = "cuda"
num_classes = 5


def inversion(G, D, T, E, iden, itr, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=False, num_seeds=20, save_img_dir=None):
	os.makedirs(save_img_dir, exist_ok=True)
	
	iden = iden.view(-1).long().cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	bs = iden.shape[0]
	
	G.eval()
	D.eval()
	T.eval()
	E.eval()

	# max_score = torch.zeros(bs)
	# max_iden = torch.zeros(bs)
	# max_prob = torch.zeros(bs, num_classes)
	# z_hat = torch.zeros(bs, 100)
	flag = torch.zeros(bs)
	no = torch.zeros(bs) # index for saving all success attack images

	res = []
	res5 = []

	for random_seed in range(num_seeds):
		class_change = np.zeros((bs, iter_times))

		tf = time.time()
		torch.manual_seed(random_seed) 
		torch.cuda.manual_seed(random_seed) 
		np.random.seed(random_seed) 
		random.seed(random_seed)

		z = torch.randn(bs, 100).cuda().float()
		z.requires_grad = True
		v = torch.zeros(bs, 100).cuda().float()

			
		for i in range(iter_times):
			fake = G(z)
			if improved == True:
				_, label =  D(fake)
			else:
				label = D(fake)
			
			out = T(fake)[-1]

			# class change plot
			if iden[0] == 38:
				eval_prob = E(fake)[-1]
				eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
				pred = torch.argmax(eval_prob, dim=1).view(-1)
				for b in range(bs):
					class_change[b, i] = int(pred[b])

			if iden[0] == 38 and i == 0:
				fig = plt.figure(figsize=(12,12))
				list_cls = np.arange(39)
				num_cls = np.zeros(39)
				for cls in list_cls:
					idx = np.where(eval_iden.detach().cpu().numpy()==cls)[0]
					num_cls[cls] = len(idx)
				plt.bar(list_cls, num_cls)
				plt.xlabel('classes')
				plt.ylabel('number of generated samples to be predicted in this class')
				plt.title('Initial')
				plt.savefig(save_img_dir + "class_cnt_initial_seed{}.png".format(random_seed))		
			
			
			if z.grad is not None:
				z.grad.data.zero_()

			if improved:
				Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
				# Prior_Loss =  torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(label.gather(1, iden.view(-1, 1)))  #1 class prior
			else:
				Prior_Loss = - label.mean()

			Iden_Loss = criterion(out, iden)

			Total_Loss = Prior_Loss + lamda * Iden_Loss

			Total_Loss.backward()
			
			v_prev = v.clone()
			gradient = z.grad.data
			v = momentum * v - lr * gradient
			z = z + ( - momentum * v_prev + (1 + momentum) * v)
			z = torch.clamp(z.detach(), -clip_range, clip_range).float()
			z.requires_grad = True

			Prior_Loss_val = Prior_Loss.item()
			Iden_Loss_val = Iden_Loss.item()

			if (i+1) % 300 == 0:
				fake_img = G(z.detach())
				eval_prob = E(fake_img)[-1]
				eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
				acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
				print("Iteration:{}\tPrior Loss:{:.4f}\tIden Loss:{:.4f}\tAttack Acc:{:.4f}".format(i+1, Prior_Loss_val, Iden_Loss_val, acc))
			
		
		fake = G(z)
		score = T(fake)[-1]
		eval_prob = E(fake)[-1]
		eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
		
		cnt, cnt5 = 0, 0
		for i in range(bs):
			gt = iden[i].item()
			gen_img = fake[i]

			save_tensor_images(gen_img.detach(), os.path.join(save_img_dir, "{}_attack_iden_{}_{}_{}.png".format(random_seed, gt, (eval_iden[i].item() == gt), i)))

			if eval_iden[i].item() == gt:
				cnt += 1
				flag[i] = 1
				no[i] += 1
			_, top5_idx = torch.topk(eval_prob[i], 5)
			if gt in top5_idx:
				cnt5 += 1
				
		
		interval = time.time() - tf
		print("Time:{:.4f}\tAcc:{:.4f}\t".format(interval, cnt * 1.0 / bs))
		res.append(cnt * 1.0 / bs)
		res5.append(cnt5 * 1.0 / bs)
		torch.cuda.empty_cache()

		if iden[0] == 38:
			pickle.dump(z.detach(), open('./final_z_surrogateGaussian_seed{}.data'.format(random_seed), 'wb'))

			pickle.dump(class_change, open(save_img_dir + '/chass_change{}.numpy'.format(random_seed), 'wb'))

			x = np.arange(iter_times)
			fig = plt.figure(figsize=(12,12))
			for b in range(bs):
				plt.plot(x, class_change[b], color='C{}'.format(b))
			
			plt.savefig(save_img_dir + "class_change_seed{}.png".format(random_seed))

			fig = plt.figure(figsize=(12,12))
			list_cls = np.arange(39)
			num_cls = np.zeros(39)
			for cls in list_cls:
				idx = np.where(eval_iden.detach().cpu().numpy()==cls)[0]
				num_cls[cls] = len(idx)
			plt.bar(list_cls, num_cls)
			plt.xlabel('classes')
			plt.ylabel('number of generated samples to be predicted in this class')
			plt.title('Final')
			plt.savefig(save_img_dir + "class_cnt_final_seed{}.png".format(random_seed))			

	
	acc, acc_5 = statistics.mean(res), statistics.mean(res5)
	acc_var = statistics.variance(res)
	print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}".format(acc, acc_5, acc_var))

	

	
	return acc, acc_5, acc_var

# if __name__ == '__main__':
# 	pass



	
	
	
	
	

	
	
		

	

