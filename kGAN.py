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
import torch.nn.functional as F
from discri import DGWGAN32, MinibatchDiscriminator_CIFAR
from generator import GeneratorCIFAR
from classify import *
from tensorboardX import SummaryWriter
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

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

def log_sum_exp(x, axis = 1):
    m = torch.max(x, dim = 1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))


save_img_dir = "./kGAN/imgs_improved_stl10"
save_model_dir= "./kGAN/"
os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(save_img_dir, exist_ok=True)


log_path = "./kGAN"
os.makedirs(log_path, exist_ok=True)
log_file = "stl10_kGAN.txt"
utils.Tee(os.path.join(log_path, log_file), 'w')



if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
    os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5, 6, 7'
    global args, writer
    
    file = "./config/cifar.json"
    args = load_json(json_file=file)
    writer = SummaryWriter(log_path)

    file_path = args['dataset']['gan_file_path']
    model_name = args['dataset']['model_name']
    lr = args[model_name]['lr']
    batch_size = args[model_name]['batch_size']
    z_dim = args[model_name]['z_dim']
    epochs = args[model_name]['epochs']
    n_critic = args[model_name]['n_critic']

    T = VGG16(5)
    path_T = './Attack/target_model/target_ckp/VGG16_92.15.tar'

    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'], strict=False)

    print("---------------------Training [%s]------------------------------" % model_name)
    utils.print_params(args["dataset"], args[model_name])

    dataset, dataloader = utils.init_dataloader(args, file_path, batch_size, mode="gan")

    G = GeneratorCIFAR(z_dim)
    DG = MinibatchDiscriminator_CIFAR()
    
    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    entropy = HLoss()
    

    step = 0

    for epoch in range(epochs):
        start = time.time()
        _, unlabel_loader1 = init_dataloader(args, file_path, batch_size, mode="gan", iterator=True)
        _, unlabel_loader2 = init_dataloader(args, file_path, batch_size, mode="gan", iterator=True)

        for i, imgs in enumerate(dataloader):
            current_iter = epoch * len(dataloader) + i + 1

            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)
            x_unlabel = unlabel_loader1.next()
            x_unlabel2 = unlabel_loader2.next()
            
            freeze(G)
            unfreeze(DG)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            y_prob = T(imgs)[-1]
            y = torch.argmax(y_prob, dim=1).view(-1)
            

            _, output_label = DG(imgs)
            _, output_unlabel = DG(x_unlabel)
            _, output_fake =  DG(f_imgs)

            loss_lab = softXEnt(output_label, y_prob)
            # loss_lab = torch.mean(torch.mean(log_sum_exp(output_label)))-torch.mean(torch.gather(output_label, 1, y.unsqueeze(1))) # same as crossEntropy loss
            # import pdb; pdb.set_trace()
            loss_unlab = 0.5*(torch.mean(F.softplus(log_sum_exp(output_unlabel)))-torch.mean(log_sum_exp(output_unlabel))+torch.mean(F.softplus(log_sum_exp(output_fake))))
            dg_loss = loss_lab + loss_unlab
            
            acc = torch.mean((output_label.max(1)[1] == y).float())
            
            
            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            writer.add_scalar('loss_label_batch', loss_lab, current_iter)
            writer.add_scalar('loss_unlabel_batch', loss_unlab, current_iter)
            writer.add_scalar('DG_loss_batch', dg_loss, current_iter)
            writer.add_scalar('Acc_batch', acc, current_iter)

            # train G

            if step % n_critic == 0:
                freeze(DG)
                unfreeze(G)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                mom_gen, output_fake = DG(f_imgs)
                mom_unlabel, _ = DG(x_unlabel2)

                mom_gen = torch.mean(mom_gen, dim = 0)
                mom_unlabel = torch.mean(mom_unlabel, dim = 0)

                # Hloss = entropy(T(f_imgs)[-1])
                Hloss = entropy(output_fake)
                import pdb; pdb.set_trace()
                g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss  # feature matching loss
                # g_loss = torch.mean(F.softplus(log_sum_exp(output_fake)))-torch.mean(log_sum_exp(output_fake)) + 1e-4 * Hloss 
                # g_loss = torch.mean((mom_gen - mom_unlabel).abs())
                # import pdb; pdb.set_trace()

                
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                writer.add_scalar('G_loss_batch', g_loss, current_iter)

        end = time.time()
        interval = end - start
        
        print("Epoch:%d \tTime:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (epoch, interval, g_loss, acc))

        torch.save({'state_dict':G.state_dict()}, os.path.join(save_model_dir, "improved_mb_cifar_G_entropy.tar"))
        torch.save({'state_dict':DG.state_dict()}, os.path.join(save_model_dir, "improved_mb_cifar_D_entropy.tar"))

        if (epoch+1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "improved_mb_gan_image_{}_ffhq_entropy.png".format(epoch)), nrow = 8)
            for b in range(fake_image.size(0)):
                writer.add_image('Visualization_%d' % b, fake_image[b])
            # shutil.copyfile(
            #     os.path.join(save_model_dir, "improved_mb_celeba_G.tar"),
            #     save_model_dir + '/improved_mb_G_train_epoch_' + str(epoch) + '.tar')
            # shutil.copyfile(
            #     os.path.join(save_model_dir, "improved_mb_celeba_D.tar"),
            #     save_model_dir + '/improved_mb_D_train_epoch_' + str(epoch) + '.tar')
        
        

