import torch
from torch import nn
import cv2
import numpy
from tqdm import trange
import time
from scipy import io
import models.modules.model as model
from utils.propagation_ASM import *
from data.dataset import dataset
from models.jpeg import DiffJPEG
from models.modules.quantization import Quantization
from torch.utils.data import DataLoader, Dataset, random_split
from config import config
import pytorch_ssim
from torch.nn.parallel import DataParallel, DistributedDataParallel
from collections import OrderedDict
import models.networks as networks
from models.compressor import REALCOMP
from models.modules.loss import ReconstructionLoss
import logging
import models.lr_scheduler as lr_scheduler

logger = logging.getLogger('base')
class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def _set_lr(self, lr_groups_l):
        ''' set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer'''
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        #### set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        # return self.schedulers[0].get_lr()[0]
        return self.optimizers[0].param_groups[0]['lr']

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

class CCNNCAR(BaseModel):
    def __init__(self, opt):
        super(CCNNCAR, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.net = networks.define(opt)

        if opt['dist']:
            self.net = DistributedDataParallel(self.net, device_ids=[torch.cuda.current_device()])
        else:
            self.net = self.net.cuda()

        self.print_network()
        self.load()
        self.Quantization = Quantization()

        if train_opt['use_diffcomp']:
            if train_opt['comp_quality']:
                self.diffcomp = DiffJPEG(differentiable=True, quality=train_opt['comp_quality']).cuda()
            else:
                self.diffcomp = DiffJPEG(differentiable=True, quality=75).cuda()
        if train_opt['use_realcomp']:
            self.realcomp = REALCOMP(format=train_opt['comp_format'], quality=train_opt['comp_quality'])
        if self.is_train:
            self.net.train()

            #loss 
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])

            #optimizers
            wd = train_opt['weight_decay'] if train_opt['weight_decay'] else 0
            optim_params = []
            for k,v in self.net.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            
            self.optimizer = torch.optim.Adam(optim_params, lr=train_opt['lr'],
                                              weight_decay=wd,
                                              betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer)

            #schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(                            
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')
            
            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device) #LQ
        self.real_H = data['GT'].to(self.device) #GT

    def loss_forward(self, out, y):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
        return l_forw_fit
    
    def loss_backward(self, out, x):
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_forw(out, x)
        return l_back_rec
    
    def optimize_parameters(self, step):
        self.optimizer.zero_grad()

        self.input = self.real_H
        #LR = self.net.encode(x=self.input)

        LR_ref = self.ref_L.detach()
        
        #l_forw_fit1 = self.loss_forward(LR, LR_ref)

        #backward upscaling
        # LR_real = LR.real
        # LR_imag = LR.imag
        # LR_real = self.Quantization(LR_real)
        # LR_imag = self.Quantization(LR_imag)
        # LR_real_ = self.diffcomp(LR_real)
        # LR_imag_ = self.diffcomp(LR_imag)
        # LR_ = torch.complex(LR_real_, LR_imag_)

        # HR = self.net.decode(x=LR_)
        #HR = self.net.decode(x=LR)
        HR = self.net(x=self.input)

        l_back_rec = self.loss_backward(HR, self.real_H)

        loss = l_back_rec

        loss.backward()

        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.train_opt['gradient_clipping'])
        
        self.optimizer.step()

        #set log
        #self.log_dict['l_forw_fit1_real'] = l_forw_fit1.real.item()
        #self.log_dict['l_forw_fit1_imag'] = l_forw_fit1.imag.item()
        self.log_dict['l_back_rec_real'] = l_back_rec.real.item()
        self.log_dict['l_back_rec_imag'] = l_back_rec.imag.item()


    def test(self):
        Lshape = self.ref_L.shape

        input_dim = Lshape[1]
        self.input = self.real_H

        self.net.eval()
        if len(self.input.shape) == 3:
            self.input = self.input.unsqueeze(0)
        with torch.no_grad():
            #LR = self.net.encode(x=self.input)
            self.forw_L = self.ref_L
            # LR_real = LR.real
            # LR_imag = LR.imag
            # forw_L_real = self.Quantization(LR_real)
            # forw_L_imag = self.Quantization(LR_imag)
            # self.forw_L = torch.complex(forw_L_real, forw_L_imag)
            # y_forw_real = self.realcomp(forw_L_real)
            # y_forw_imag = self.realcomp(forw_L_imag)
            # y_forw = torch.complex(y_forw_real, y_forw_imag)
            #self.fake_H = self.net.decode(x=y_forw)
            self.fake_H = self.net(x=self.input)
        
        self.net.train()

    def downscale(self, HR_img):
        self.net.eval()
        with torch.no_grad():
            #LR_img = self.Quantization(self.net.encode(x=HR_img))
            LR_img = self.net.encode(x=HR_img)
        self.net.train()
        return LR_img
    
    def upscale(self, LR_img, scale, gaussian_scale=1):
        self.net.eval()
        with torch.no_grad():
            HR_img = self.net.decode(x=LR_img)
        self.net.train()

        return HR_img
    
    def get_current_log(self):
        return self.log_dict
    
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach().unsqueeze(0)
        out_dict['SR'] = self.fake_H.detach()[0].unsqueeze(0)
        out_dict['LR'] = self.forw_L.detach().unsqueeze(0)
        out_dict['GT'] = self.real_H.detach().unsqueeze(0)
        return out_dict
    
    def print_network(self):
        s, n = self.get_network_description(self.net)
        if isinstance(self.net, nn.DataParallel) or isinstance(self.net, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.net.__class__.__name__,
                                             self.net.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.net.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path = self.opt['path']['pretrain_model']
        if load_path is not None:
            logger.info('Loading model from [{:s}] ...'.format(load_path))
            self.load_network(load_path, self.net, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.net, 'net', iter_label)







        
    
