from networks import AdaINGen, MsImageDis, IntrinsicSplitor, IntrinsicMerger, LocalAlbedoSmoothnessLoss
from utils import weights_init, get_model_list, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os


class UnsupIntrinsicTrainer(nn.Module):
    def __init__(self, param):
        super(UnsupIntrinsicTrainer, self).__init__()
        lr = param['lr']
        # Initiate the networks
        self.gen_i = AdaINGen(param['input_dim_a'], param['input_dim_a'], param['gen'])  # auto-encoder for domain I
        self.gen_r = AdaINGen(param['input_dim_b'], param['input_dim_b'], param['gen'])  # auto-encoder for domain R
        self.gen_s = AdaINGen(param['input_dim_c'], param['input_dim_c'], param['gen'])  # auto-encoder for domain S
        self.dis_r = MsImageDis(param['input_dim_b'], param['dis'])  # discriminator for domain R
        self.dis_s = MsImageDis(param['input_dim_c'], param['dis'])  # discriminator for domain S
        gp = param['gen']
        self.with_mapping = True
        self.use_phy_loss = True
        self.use_content_loss = True
        if 'ablation_study' in param:
            if 'with_mapping' in param['ablation_study']:
                wm = param['ablation_study']['with_mapping']
                self.with_mapping = True if wm != 0 else False
            if 'wo_phy_loss' in param['ablation_study']:
                wpl = param['ablation_study']['wo_phy_loss']
                self.use_phy_loss = True if wpl == 0 else False
            if 'wo_content_loss' in param['ablation_study']:
                wcl = param['ablation_study']['wo_content_loss']
                self.use_content_loss = True if wcl == 0 else False

        if self.with_mapping:
            self.fea_s = IntrinsicSplitor(gp['style_dim'], gp['mlp_dim'], gp['n_layer'], gp['activ'])  # split style for I
            self.fea_m = IntrinsicMerger(gp['style_dim'], gp['mlp_dim'], gp['n_layer'], gp['activ'])  # merge style for R, S
        self.bias_shift = param['bias_shift']
        self.instance_norm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = param['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(param['display_size'])
        self.s_r = torch.randn(display_size, self.style_dim, 1, 1).cuda() + 1.
        self.s_s = torch.randn(display_size, self.style_dim, 1, 1).cuda() - 1.

        # Setup the optimizers
        beta1 = param['beta1']
        beta2 = param['beta2']
        dis_params = list(self.dis_r.parameters()) + list(self.dis_s.parameters())
        if self.with_mapping:
            gen_params = list(self.gen_i.parameters()) + list(self.gen_r.parameters()) + \
                         list(self.gen_s.parameters()) + \
                         list(self.fea_s.parameters()) + list(self.fea_m.parameters())
        else:
            gen_params = list(self.gen_i.parameters()) + list(self.gen_r.parameters()) + list(self.gen_s.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=param['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=param['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, param)
        self.gen_scheduler = get_scheduler(self.gen_opt, param)

        # Network weight initialization
        self.apply(weights_init(param['init']))
        self.dis_r.apply(weights_init('gaussian'))
        self.dis_s.apply(weights_init('gaussian'))
        self.best_result = float('inf')
        self.reflectance_loss = LocalAlbedoSmoothnessLoss(param)

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def physical_criterion(self, x_i, x_r, x_s):
        return torch.mean(torch.abs(x_i - x_r * x_s))

    def forward(self, x_i):
        c_i, s_i_fake = self.gen_i.encode(x_i)
        if self.with_mapping:
            s_r, s_s = self.fea_s(s_i_fake)
        else:
            s_r = Variable(torch.randn(x_i.size(0), self.style_dim, 1, 1).cuda()) + self.bias_shift
            s_s = Variable(torch.randn(x_i.size(0), self.style_dim, 1, 1).cuda()) - self.bias_shift
        x_ri = self.gen_r.decode(c_i, s_r)
        x_si = self.gen_s.decode(c_i, s_s)
        return x_ri, x_si

    def inference(self, x_i, use_rand_fea=False):
        with torch.no_grad():
            c_i, s_i_fake = self.gen_i.encode(x_i)
            if self.with_mapping:
                s_r, s_s = self.fea_s(s_i_fake)
            else:
                s_r = Variable(torch.randn(x_i.size(0), self.style_dim, 1, 1).cuda()) + self.bias_shift
                s_s = Variable(torch.randn(x_i.size(0), self.style_dim, 1, 1).cuda()) - self.bias_shift
            if use_rand_fea:
                s_r = Variable(torch.randn(x_i.size(0), self.style_dim, 1, 1).cuda()) + self.bias_shift
                s_s = Variable(torch.randn(x_i.size(0), self.style_dim, 1, 1).cuda()) - self.bias_shift
            x_ri = self.gen_r.decode(c_i, s_r)
            x_si = self.gen_s.decode(c_i, s_s)
        return x_ri, x_si

    # noinspection PyAttributeOutsideInit
    def gen_update(self, x_i, x_r, x_s, targets=None, param=None):
        self.gen_opt.zero_grad()
        # ============= Domain Translations =============
        # encode
        c_i, s_i_prime = self.gen_i.encode(x_i)
        c_r, s_r_prime = self.gen_r.encode(x_r)
        c_s, s_s_prime = self.gen_s.encode(x_s)

        if self.with_mapping:
            s_ri, s_si = self.fea_s(s_i_prime)
        else:
            s_ri = Variable(torch.randn(x_i.size(0), self.style_dim, 1, 1).cuda()) + self.bias_shift
            s_si = Variable(torch.randn(x_i.size(0), self.style_dim, 1, 1).cuda()) - self.bias_shift
        s_r_rand = Variable(torch.randn(x_r.size(0), self.style_dim, 1, 1).cuda()) + self.bias_shift
        s_s_rand = Variable(torch.randn(x_s.size(0), self.style_dim, 1, 1).cuda()) - self.bias_shift
        if self.with_mapping:
            s_i_recon = self.fea_m(s_ri, s_si)
        else:
            s_i_recon = Variable(torch.randn(x_i.size(0), self.style_dim, 1, 1).cuda())
        # decode (within domain)
        x_i_recon = self.gen_i.decode(c_i, s_i_prime)
        x_r_recon = self.gen_s.decode(c_r, s_r_prime)
        x_s_recon = self.gen_r.decode(c_s, s_s_prime)
        # decode (cross domain)
        x_rs = self.gen_r.decode(c_s, s_r_rand)
        x_ri = self.gen_r.decode(c_i, s_ri)
        x_ri_rand = self.gen_r.decode(c_i, s_r_rand)
        x_sr = self.gen_s.decode(c_r, s_s_rand)
        x_si = self.gen_s.decode(c_i, s_si)
        x_si_rand = self.gen_s.decode(c_i, s_r_rand)
        # encode again, for feature domain consistency constraints
        c_rs_recon, s_rs_recon = self.gen_r.encode(x_rs)
        c_ri_recon, s_ri_recon = self.gen_r.encode(x_ri)
        c_ri_rand_recon, s_ri_rand_recon = self.gen_r.encode(x_ri_rand)
        c_sr_recon, s_sr_recon = self.gen_s.encode(x_sr)
        c_si_recon, s_si_recon = self.gen_s.encode(x_si)
        c_si_rand_recon, s_si_rand_recon = self.gen_s.encode(x_si_rand)
        # decode again, for image domain cycle consistency
        x_rsr = self.gen_r.decode(c_sr_recon, s_r_prime)
        x_iri = self.gen_i.decode(c_ri_recon, s_i_prime)
        x_iri_rand = self.gen_i.decode(c_ri_rand_recon, s_i_prime)
        x_srs = self.gen_s.decode(c_rs_recon, s_s_prime)
        x_isi = self.gen_i.decode(c_si_recon, s_i_prime)
        x_isi_rand = self.gen_i.decode(c_si_rand_recon, s_i_prime)

        # ============= Loss Functions =============
        # Encoder decoder reconstruction loss for three domain
        self.loss_gen_recon_x_i = self.recon_criterion(x_i_recon, x_i)
        self.loss_gen_recon_x_r = self.recon_criterion(x_r_recon, x_r)
        self.loss_gen_recon_x_s = self.recon_criterion(x_s_recon, x_s)
        # Style-level reconstruction loss for cross domain
        if self.with_mapping:
            self.loss_gen_recon_s_ii = self.recon_criterion(s_i_recon, s_i_prime)
        else:
            self.loss_gen_recon_s_ii = 0
        self.loss_gen_recon_s_ri = self.recon_criterion(s_ri_recon, s_ri)
        self.loss_gen_recon_s_ri_rand = self.recon_criterion(s_ri_rand_recon, s_ri)
        self.loss_gen_recon_s_rs = self.recon_criterion(s_rs_recon, s_r_rand)
        self.loss_gen_recon_s_sr = self.recon_criterion(s_sr_recon, s_s_rand)
        self.loss_gen_recon_s_si = self.recon_criterion(s_si_recon, s_si)
        self.loss_gen_recon_s_si_rand = self.recon_criterion(s_si_rand_recon, s_si)
        # Content-level reconstruction loss for cross domain
        self.loss_gen_recon_c_rs = self.recon_criterion(c_rs_recon, c_s)
        self.loss_gen_recon_c_ri = self.recon_criterion(c_ri_recon, c_i) if self.use_content_loss is True else 0
        self.loss_gen_recon_c_ri_rand = self.recon_criterion(c_ri_rand_recon, c_i) if self.use_content_loss is True else 0
        self.loss_gen_recon_c_sr = self.recon_criterion(c_sr_recon, c_r)
        self.loss_gen_recon_c_si = self.recon_criterion(c_si_recon, c_i) if self.use_content_loss is True else 0
        self.loss_gen_recon_c_si_rand = self.recon_criterion(c_si_rand_recon, c_i) if self.use_content_loss is True else 0
        # Cycle consistency loss for three image domain
        self.loss_gen_cyc_recon_x_rs = self.recon_criterion(x_rsr, x_r)
        self.loss_gen_cyc_recon_x_ir = self.recon_criterion(x_iri, x_i)
        self.loss_gen_cyc_recon_x_ir_rand = self.recon_criterion(x_iri_rand, x_i)
        self.loss_gen_cyc_recon_x_sr = self.recon_criterion(x_srs, x_s)
        self.loss_gen_cyc_recon_x_is = self.recon_criterion(x_isi, x_i)
        self.loss_gen_cyc_recon_x_is_rand = self.recon_criterion(x_isi_rand, x_i)
        # GAN loss
        self.loss_gen_adv_rs = self.dis_r.calc_gen_loss(x_rs)
        self.loss_gen_adv_ri = self.dis_r.calc_gen_loss(x_ri)
        self.loss_gen_adv_ri_rand = self.dis_r.calc_gen_loss(x_ri_rand)
        self.loss_gen_adv_sr = self.dis_s.calc_gen_loss(x_sr)
        self.loss_gen_adv_si = self.dis_s.calc_gen_loss(x_si)
        self.loss_gen_adv_si_rand = self.dis_s.calc_gen_loss(x_si_rand)
        # Physical loss
        self.loss_gen_phy_i = self.physical_criterion(x_i, x_ri, x_si) if self.use_phy_loss is True else 0
        self.loss_gen_phy_i_rand = self.physical_criterion(x_i, x_ri_rand, x_si_rand) if self.use_phy_loss is True else 0

        # Reflectance smoothness loss
        self.loss_refl_ri = self.reflectance_loss(x_ri, targets) if targets is not None else 0
        self.loss_refl_ri_rand = self.reflectance_loss(x_ri_rand, targets) if targets is not None else 0

        # total loss
        self.loss_gen_total = param['gan_w'] * self.loss_gen_adv_rs + \
                              param['gan_w'] * self.loss_gen_adv_ri + \
                              param['gan_w'] * self.loss_gen_adv_ri_rand + \
                              param['gan_w'] * self.loss_gen_adv_sr + \
                              param['gan_w'] * self.loss_gen_adv_si + \
                              param['gan_w'] * self.loss_gen_adv_si_rand + \
                              param['recon_x_w'] * self.loss_gen_recon_x_i + \
                              param['recon_x_w'] * self.loss_gen_recon_x_r + \
                              param['recon_x_w'] * self.loss_gen_recon_x_s + \
                              param['recon_s_w'] * self.loss_gen_recon_s_ii + \
                              param['recon_s_w'] * self.loss_gen_recon_s_ri + \
                              param['recon_s_w'] * self.loss_gen_recon_s_ri_rand + \
                              param['recon_s_w'] * self.loss_gen_recon_s_rs + \
                              param['recon_s_w'] * self.loss_gen_recon_s_si + \
                              param['recon_s_w'] * self.loss_gen_recon_s_si_rand + \
                              param['recon_s_w'] * self.loss_gen_recon_s_sr + \
                              param['recon_c_w'] * self.loss_gen_recon_c_ri + \
                              param['recon_c_w'] * self.loss_gen_recon_c_rs + \
                              param['recon_c_w'] * self.loss_gen_recon_c_ri_rand + \
                              param['recon_c_w'] * self.loss_gen_recon_c_si + \
                              param['recon_c_w'] * self.loss_gen_recon_c_sr + \
                              param['recon_c_w'] * self.loss_gen_recon_c_si_rand + \
                              param['recon_x_cyc_w'] * self.loss_gen_cyc_recon_x_ir + \
                              param['recon_x_cyc_w'] * self.loss_gen_cyc_recon_x_ir_rand + \
                              param['recon_x_cyc_w'] * self.loss_gen_cyc_recon_x_is + \
                              param['recon_x_cyc_w'] * self.loss_gen_cyc_recon_x_is_rand + \
                              param['recon_x_cyc_w'] * self.loss_gen_cyc_recon_x_rs + \
                              param['recon_x_cyc_w'] * self.loss_gen_cyc_recon_x_sr + \
                              param['phy_x_w'] * self.loss_gen_phy_i + \
                              param['phy_x_w'] * self.loss_gen_phy_i_rand + \
                              param['refl_smooth_w'] * self.loss_refl_ri + \
                              param['refl_smooth_w'] * self.loss_refl_ri_rand

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def sample(self, x_i, x_r, x_s):
        self.eval()
        s_r = Variable(self.s_r)
        s_s = Variable(self.s_s)
        x_i_recon, x_r_recon, x_s_recon, x_rs, x_ri, x_sr, x_si = [], [], [], [], [], [], []
        for i in range(x_i.size(0)):
            c_i, s_i_fake = self.gen_i.encode(x_i[i].unsqueeze(0))
            c_r, s_r_fake = self.gen_r.encode(x_r[i].unsqueeze(0))
            c_s, s_s_fake = self.gen_s.encode(x_s[i].unsqueeze(0))
            if self.with_mapping:
                s_ri, s_si = self.fea_s(s_i_fake)
            else:
                s_ri = Variable(torch.randn(1, self.style_dim, 1, 1).cuda()) + self.bias_shift
                s_si = Variable(torch.randn(1, self.style_dim, 1, 1).cuda()) - self.bias_shift
            x_i_recon.append(self.gen_i.decode(c_i, s_i_fake))
            x_r_recon.append(self.gen_r.decode(c_r, s_r_fake))
            x_s_recon.append(self.gen_s.decode(c_s, s_s_fake))
            x_rs.append(self.gen_r.decode(c_s, s_r[i].unsqueeze(0)))
            x_ri.append(self.gen_r.decode(c_i, s_ri.unsqueeze(0)))
            x_sr.append(self.gen_s.decode(c_s, s_s[i].unsqueeze(0)))
            x_si.append(self.gen_s.decode(c_i, s_si.unsqueeze(0)))
        x_i_recon, x_r_recon, x_s_recon = torch.cat(x_i_recon), torch.cat(x_r_recon), torch.cat(x_s_recon)
        x_rs, x_ri = torch.cat(x_rs), torch.cat(x_ri)
        x_sr, x_si = torch.cat(x_sr), torch.cat(x_si)
        self.train()
        return x_i, x_i_recon, x_r, x_r_recon, x_rs, x_ri, x_s, x_s_recon, x_sr, x_si

    # noinspection PyAttributeOutsideInit
    def dis_update(self, x_i, x_r, x_s, params):
        self.dis_opt.zero_grad()
        s_r = Variable(torch.randn(x_r.size(0), self.style_dim, 1, 1).cuda()) - self.bias_shift
        s_s = Variable(torch.randn(x_s.size(0), self.style_dim, 1, 1).cuda()) + self.bias_shift
        # encode
        c_r, _ = self.gen_r.encode(x_r)
        c_s, _ = self.gen_s.encode(x_s)
        c_i, s_i = self.gen_i.encode(x_i)
        if self.with_mapping:
            s_ri, s_si = self.fea_s(s_i)
        else:
            s_ri = Variable(torch.randn(x_i.size(0), self.style_dim, 1, 1).cuda()) + self.bias_shift
            s_si = Variable(torch.randn(x_i.size(0), self.style_dim, 1, 1).cuda()) - self.bias_shift
        # decode (cross domain)
        x_rs = self.gen_r.decode(c_s, s_r)
        x_ri = self.gen_r.decode(c_i, s_ri)
        x_sr = self.gen_s.decode(c_r, s_s)
        x_si = self.gen_s.decode(c_i, s_si)
        # D loss
        self.loss_dis_rs = self.dis_r.calc_dis_loss(x_rs.detach(), x_r)
        self.loss_dis_ri = self.dis_r.calc_dis_loss(x_ri.detach(), x_r)
        self.loss_dis_sr = self.dis_s.calc_dis_loss(x_sr.detach(), x_s)
        self.loss_dis_si = self.dis_s.calc_dis_loss(x_si.detach(), x_s)

        self.loss_dis_total = params['gan_w'] * self.loss_dis_rs +\
                              params['gan_w'] * self.loss_dis_ri +\
                              params['gan_w'] * self.loss_dis_sr +\
                              params['gan_w'] * self.loss_dis_si

        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, param):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_i.load_state_dict(state_dict['i'])
        self.gen_r.load_state_dict(state_dict['r'])
        self.gen_s.load_state_dict(state_dict['s'])
        if self.with_mapping:
            self.fea_m.load_state_dict(state_dict['fm'])
            self.fea_s.load_state_dict(state_dict['fs'])
        self.best_result = state_dict['best_result']
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_r.load_state_dict(state_dict['r'])
        self.dis_s.load_state_dict(state_dict['s'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, param, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, param, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        if self.with_mapping:
            torch.save({'i': self.gen_i.state_dict(), 'r': self.gen_r.state_dict(), 's': self.gen_s.state_dict(),
                        'fs': self.fea_s.state_dict(), 'fm': self.fea_m.state_dict(),
                        'best_result': self.best_result}, gen_name)
        else:
            torch.save({'i': self.gen_i.state_dict(), 'r': self.gen_r.state_dict(), 's': self.gen_s.state_dict(),
                        'best_result': self.best_result}, gen_name)
        torch.save({'r': self.dis_r.state_dict(), 's': self.dis_s.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)


class SupIntrinsicTrainer(nn.Module):
    def __init__(self, param):
        super(SupIntrinsicTrainer, self).__init__()
        lr = param['lr']
        # Initiate the networks
        self.model = AdaINGen(param['input_dim_a'],
                              param['input_dim_b'] + param['input_dim_b'],
                              param['gen'])  # auto-encoder

        # Setup the optimizers
        beta1 = param['beta1']
        beta2 = param['beta2']
        gen_params = list(self.model.parameters())
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=param['weight_decay'])
        self.gen_scheduler = get_scheduler(self.gen_opt, param)

        # Network weight initialization
        self.apply(weights_init(param['init']))
        self.best_result = float('inf')

    def recon_criterion(self, input, target, mask=None):
        if mask is not None:
            return torch.mean(torch.abs(input[mask] - target[mask]))
        else:
            return torch.mean(torch.abs(input - target))

    def forward(self, x):
        self.eval()
        out = self.model(x)
        x_r, x_s = out[:, :3, :, :], out[:, :, 3:, :]
        return x_r, x_s

    def gen_update(self, x_i, x_r, x_s, x_m, param):
        self.gen_opt.zero_grad()

        out = self.model(x_i)
        pred_r, pred_s = out[:, :3, :, :], out[:, 3:, :, :]

        # reconstruction loss
        self.loss_r = self.recon_criterion(pred_r, x_r, x_m)
        self.loss_s = self.recon_criterion(pred_s, x_s, x_m)
        # total loss
        self.loss_gen_total = self.loss_r + self.loss_s
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def sample(self, x_i, x_r, x_s):
        self.eval()
        x_ri, x_si = [], []
        for i in range(x_i.size(0)):
            out = self.model(x_i[i].unsqueeze(0))
            x_r, x_s = out[:, :3, :, :], out[:, 3:, :, :]
            x_ri.append(x_r)
            x_si.append(x_s)
        x_ri = torch.cat(x_ri)
        x_si = torch.cat(x_si)
        self.train()
        return x_i, x_r, x_ri, x_s, x_si

    # noinspection PyAttributeOutsideInit
    def dis_update(self, x_i, x_r, x_s, param=None):
        pass

    def update_learning_rate(self):
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, param):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.model.load_state_dict(state_dict['i'])
        iterations = int(last_model_name[-11:-3])

        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.gen_scheduler = get_scheduler(self.gen_opt, param, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'model': self.model.state_dict(), 'best_result': self.best_result}, gen_name)
        torch.save({'gen': self.gen_opt.state_dict()}, opt_name)

