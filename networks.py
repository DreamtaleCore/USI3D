from queue import Queue

import math
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.models import vgg19

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass


##################################################################################
# Discriminator
##################################################################################

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params):
        super(MsImageDis, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1) ** 2)  # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


##################################################################################
# Generator
##################################################################################

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, output_dim, params):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, output_dim, res_norm='adain', activ=activ,
                           pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params


class VAEGen(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, params):
        super(VAEGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']

        # content encoder
        self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='in', activ=activ,
                           pad_type=pad_type)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


##################################################################################
# Encoder and Decoders
##################################################################################

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class IntrinsicMerger(nn.Module):
    def __init__(self, feature_len, hidden_dim, n_layers, activ):
        super(IntrinsicMerger, self).__init__()
        self.mlp = MLP(feature_len * 2, feature_len, hidden_dim, n_blk=n_layers, norm='none', activ=activ)

    def forward(self, fea_r, fea_s):
        fea_in = torch.cat([fea_r, fea_s], dim=1)
        fea_in = fea_in.view(fea_in.size(0), -1)
        fea_out = self.mlp(fea_in)
        return fea_out.view(*fea_r.shape)


class IntrinsicSplitor(nn.Module):
    def __init__(self, feature_len, hidden_dim, n_layers, activ):
        super(IntrinsicSplitor, self).__init__()
        self.feature_dim = feature_len

        self.mlp = MLP(self.feature_dim, self.feature_dim * 2, hidden_dim, n_blk=n_layers, norm='none', activ=activ)

    def forward(self, fea):
        fea_out = self.mlp(fea)
        fea_r, fea_s = fea_out[:, :self.feature_dim], fea_out[:, self.feature_dim:]
        fea_r, fea_s = fea_r.view(*fea.shape), fea_s.view(*fea.shape)
        return fea_r, fea_s


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# VGG network definition
##################################################################################
class Vgg19Encoder(nn.Module):
    def __init__(self, input_dim, pretrained):
        super(Vgg19Encoder, self).__init__()
        features = list(vgg19(pretrained=pretrained, in_channels=input_dim).features)
        self.features = nn.ModuleList(features)

    def forward(self, x):
        result_dict = {}
        layer_names = ['conv1_1', 'conv1_2',
                       'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                       'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                       'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
        idx = 0
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34}:
                result_dict[layer_names[idx]] = x
                idx += 1

        out_feature = {
            'input': x,
            'shallow': result_dict['conv1_2'],
            'low': result_dict['conv2_2'],
            'mid': result_dict['conv3_2'],
            'deep': result_dict['conv4_2'],
            'out': result_dict['conv5_4']
        }
        return out_feature


##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        """
        SSIM Loss, return 1 - SSIM
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    @staticmethod
    def _gaussian(window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    @staticmethod
    def _ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img_a, img_b):
        (_, channel, _, _) = img_a.size()

        if channel == self.channel and self.window.data.type() == img_a.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)

            if img_a.is_cuda:
                window = window.cuda(img_a.get_device())
            window = window.type_as(img_a)

            self.window = window
            self.channel = channel

        ssim_v = self._ssim(img_a, img_b, window, self.window_size, channel, self.size_average)

        return 1 - ssim_v


class LossAdaptor(object):
    """
    An adaptor aim to balance loss via the std of the loss
    """

    def __init__(self, queue_size=100, param_only=True):
        self.size = queue_size
        self.history = Queue(maxsize=self.size)
        self.param_only = param_only

    def __call__(self, loss_var):
        if self.history.qsize() < self.size:
            param = 1.
            self.history.put(loss_var)
        else:
            self.history.put(loss_var)
            param = np.mean(self.history.queue)

        if self.param_only:
            return param
        else:
            return param * loss_var


class RetinaLoss(nn.Module):
    """
    Gradient loss and its plus version: Exclusion loss
    """
    def __init__(self):

        super(RetinaLoss, self).__init__()
        self.l1_diff = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()
        self.downsample = nn.AvgPool2d(2)
        self.level = 3
        self.eps = 1e-6
        pass

    @staticmethod
    def compute_gradient(img):
        grad_x = img[:, :, 1:, :] - img[:, :, :-1, :]
        grad_y = img[:, :, :, 1:] - img[:, :, :, :-1]
        return grad_x, grad_y

    def compute_exclusion_loss(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)

            if torch.mean(torch.abs(gradx2)) < self.eps or torch.mean(torch.abs(gradx2)) < self.eps:
                gradx_loss.append(0)
                grady_loss.append(0)
                continue

            alphax = 2.0 * torch.mean(torch.abs(gradx1)) / (torch.mean(torch.abs(gradx2)) + self.eps)
            alphay = 2.0 * torch.mean(torch.abs(grady1)) / (torch.mean(torch.abs(grady2)) + self.eps)

            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            gradx_loss.append(torch.mean(torch.mul(torch.pow(gradx1_s, 2), torch.pow(gradx2_s, 2)) ** 0.25))
            grady_loss.append(torch.mean(torch.mul(torch.pow(grady1_s, 2), torch.pow(grady2_s, 2)) ** 0.25))

            img1 = self.downsample(img1)
            img2 = self.downsample(img2)

        loss = 0.5 * (sum(gradx_loss) / float(len(gradx_loss)) + sum(grady_loss) / float(len(grady_loss)))

        return loss

    def compute_gradient_loss(self, img1, img2):
        gradx1, grady1 = self.compute_gradient(img1)
        gradx2, grady2 = self.compute_gradient(img2)

        loss = 0.5 * (self.l1_diff(gradx1, gradx2) + self.l1_diff(grady1, grady2))
        return loss

    def forward(self, img_b, img_r, mode='exclusion'):
        """  Mode in [exclusion/gradient] """
        if mode == 'exclusion':
            loss = self.compute_exclusion_loss(img_b, img_r)
        elif mode == 'gradient':
            loss = self.compute_gradient_loss(img_b, img_r)
        else:
            raise NotImplementedError("mode should in [exclusion/gradient]")
        return loss


class VggLoss(nn.Module):
    def __init__(self, pretrained=None):
        super(VggLoss, self).__init__()
        if pretrained is None:
            pretrained = Vgg19Encoder(input_dim=3, pretrained=True)
        self.feature_extrator = pretrained
        self.l1_loss = nn.L1Loss()
        self.transform = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

    def forward(self, input, output):
        # Here I assume input and output are all in [0, 1]
        input = self.transform(input)
        output = self.transform(output)

        with torch.no_grad():
            vgg_real = self.feature_extrator(input)
            vgg_fake = self.feature_extrator(output)

            p0 = self.l1_loss(vgg_real['input'], vgg_fake['input'])
            p1 = self.l1_loss(vgg_real['shallow'], vgg_fake['shallow']) / 2.6
            p2 = self.l1_loss(vgg_real['low'], vgg_fake['low']) / 4.8
            p3 = self.l1_loss(vgg_real['mid'], vgg_fake['mid']) / 3.7
            p4 = self.l1_loss(vgg_real['deep'], vgg_fake['deep']) / 5.6
            p5 = self.l1_loss(vgg_real['out'], vgg_fake['out']) * 10 / 1.5

        return p0 + p1 + p2 + p3 + p4 + p5


class LocalAlbedoSmoothnessLoss(nn.Module):
    """
        LocalAlbedoSmoothness for fine-tune albedo and its plus version: Exclusion loss
        """

    def __init__(self, param):
        super(LocalAlbedoSmoothnessLoss, self).__init__()
        x = np.arange(-1, 2)
        y = np.arange(-1, 2)
        self.X, self.Y = np.meshgrid(x, y)
        pass

    def compute_loss(self, R, targets, scale_idx):
        h = R.size(2)
        w = R.size(3)
        num_c = R.size(1)

        half_window_size = 1
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0

        R_center = R[:, :, half_window_size + self.Y[half_window_size, half_window_size]: h - half_window_size + self.Y[
            half_window_size, half_window_size], \
                   half_window_size + self.X[half_window_size, half_window_size]:w - half_window_size + self.X[
                       half_window_size, half_window_size]]

        c_idx = 0

        for k in range(0, half_window_size * 2 + 1):
            for l in range(0, half_window_size * 2 + 1):
                albedo_weights = targets["r_w_s" + str(scale_idx)][:, c_idx, :, :].unsqueeze(1).repeat(1, num_c, 1,
                                                                                                       1).float().cuda()
                R_N = R[:, :, half_window_size + self.Y[k, l]:h - half_window_size + self.Y[k, l],
                      half_window_size + self.X[k, l]: w - half_window_size + self.X[k, l]]
                # mask_N = M[:,:,half_window_size + self.Y[k,l]:h- half_window_size + self.Y[k,l], half_window_size + self.X[k,l]: w-half_window_size + self.X[k,l] ]
                # composed_M = torch.mul(mask_N, mask_center)
                # albedo_weights = torch.mul(albedo_weights, composed_M)
                r_diff = torch.mul(Variable(albedo_weights, requires_grad=False), torch.abs(R_center - R_N))

                total_loss = total_loss + torch.mean(r_diff)
                c_idx = c_idx + 1
        return total_loss/(8.0 * num_c)

    def forward(self, prediction_R, targets):
        num_images = prediction_R.size(0)
        # Albedo smoothness term
        # rs_loss =  self.w_rs_dense * self.BilateralRefSmoothnessLoss(prediction_R, targets, 'R', 5)
        # multi-scale smoothness term
        prediction_R_1 = prediction_R[:, :, ::2, ::2]
        prediction_R_2 = prediction_R_1[:, :, ::2, ::2]
        prediction_R_3 = prediction_R_2[:, :, ::2, ::2]

        rs_loss = self.compute_loss(prediction_R, targets, 0)
        rs_loss = rs_loss + 0.5 * self.compute_loss(prediction_R_1, targets, 1)
        rs_loss = rs_loss + 0.3333 * self.compute_loss(prediction_R_2, targets, 2)
        rs_loss = rs_loss + 0.25 * self.compute_loss(prediction_R_3, targets, 3)

        return rs_loss
