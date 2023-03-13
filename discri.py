import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LinearWeightNorm
import torch.nn.init as init

class DGWGAN32(nn.Module):
    def __init__(self, in_dim=1, dim=64):
        super(DGWGAN32, self).__init__()
        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.layer1 = nn.Sequential(nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2))
        self.layer2 = conv_ln_lrelu(dim, dim * 2)
        self.layer3 = conv_ln_lrelu(dim * 2, dim * 4)
        self.layer4 = nn.Conv2d(dim * 4, 1, 4)
    
    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        y = self.layer4(feat3)
        y = y.view(-1)
        return y


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x

class MinibatchDiscriminator_CIFAR(nn.Module):
    def __init__(self,in_dim=3, dim=64, n_classes=5):
        super(MinibatchDiscriminator_CIFAR, self).__init__()
        self.n_classes = n_classes

        def conv_ln_lrelu(in_dim, out_dim, k, s, p):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, k, s, p),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.layer1 = conv_ln_lrelu(in_dim, dim, 5, 2, 2)
        self.layer2 = conv_ln_lrelu(dim, dim*2, 5, 2, 2)
        self.layer3 = conv_ln_lrelu(dim*2, dim*4, 5, 2, 2)
        self.mbd1 = MinibatchDiscrimination(4096, 64, 50)
        self.fc_layer = nn.Linear(4096+64, self.n_classes)

    def forward(self, x):
        out = []
        bs = x.shape[0]
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat = feat3.view(bs, -1)
        # print('feat:', feat.shape)
        mb_out = self.mbd1(feat)   # Nx(A+B)
        y = self.fc_layer(mb_out)
        
        return feat, y
        # return mb_out, y













