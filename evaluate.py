import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
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
from torch.utils.data import DataLoader, Dataset, random_split
from config import config
import pytorch_ssim
from models.modules.quantization import Quantization
import argparse
import options.options as option
from models.jpeg_test import DiffJPEG
from models.compressor_test import REALCOMP

def psnr(img1, img2):
    mse = np.mean((img1/1.0-img2/1.0)**2)
    psnr1=20*math.log10(1/math.sqrt(mse))
    return psnr1



parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                    help='job launcher')
parser.add_argument('--local_rank', type=int, default=0) 
args = parser.parse_args()
opt = option.parse(args.opt, is_train=True)

train_params = {
        "epoch" : 130, 
        "channel" : 3,
        "single_channel" : False
}
if train_params["single_channel"]:
    wavelengths = 0.000638 # red_channel
else :
    wavelengths = np.ones((384,384,3))
    wavelengths = wavelengths * np.array([0.000450, 0.000520, 0.000638])
    wavelengths = np.transpose(wavelengths,(2,0,1))

hologram_params = {
        "wavelengths" : wavelengths,  # laser wavelengths in BGR order
        "pitch" : 0.008,                                           # hologram pitch
        "res_h" : 384,                                 # dataset image height
        "res_w" : 384,                                 # dataset image width
        "pad" : False,
        "channels" : train_params["channel"]                                 # the channels of image
        }
net_params = {
        "channels" : hologram_params["channels"],
        "filters" : 64,
        "single_channel" : train_params["single_channel"]
}
Hbackward = propagation_ASM(torch.empty(1, 1, hologram_params["res_w"], hologram_params["res_h"]), 
                            feature_size=[hologram_params["pitch"], hologram_params["pitch"]],
                            wavelength=hologram_params["wavelengths"],
                            z = -0.02, linear_conv=hologram_params["pad"], return_H=True)
Hbackward = Hbackward.cuda()
method = 'caetad'
if method == 'caetad':
    net = model.CAETAD()
elif method == 'caetadmix':
    net = model.CAETADMIX()
elif method == 'aetad':
    net = model.AETAD()
elif method == 'aetadmix':
    net = model.AETADMIX()
elif method == 'caetadmix_po':
    net = model.CAETADMIX_po()
criterion1 = nn.MSELoss()
net = net.cuda()
criterion1 = criterion1.cuda()
criterion2 = pytorch_ssim.SSIM()
criterion2 = criterion2.cuda()

#optvars = [{'params': net.parameters()}]
#optimizer = torch.optim.Adam(optvars,lr=lr)

# hr_amp_img_path = 'D:/code/machine learning/Carcgh/MIT/MIT_test/MIT_test_HR/amp'
# lr_amp_img_path = 'D:/code/machine learning/Carcgh/MIT/MIT_test/MIT_test_LR_bicubic/amp_LR'

# hr_phs_img_path = 'D:/code/machine learning/Carcgh/MIT/MIT_test/MIT_test_HR/phs'
# lr_phs_img_path = 'D:/code/machine learning/Carcgh/MIT/MIT_test/MIT_test_LR_bicubic/phs_LR'

transform = lambda x : np.transpose(x,(2,0,1))
total_dataset = dataset(opt['datasets']['val'], transform)
# total_num = len(total_dataset)
# train_num = total_num // 10 * 9
# valid_num = total_num - train_num
# train_dataset, valid_dataset = random_split(total_dataset, [train_num, valid_num], generator=torch.Generator().manual_seed(42))
#train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
#valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
dataloader = DataLoader(total_dataset,batch_size=1, shuffle=True)
Quan = Quantization()
diffcomp1 = DiffJPEG(differentiable=True, quality=75).cuda()
diffcomp2 = DiffJPEG(differentiable=True, quality=75).cuda()
realcomp1 = REALCOMP(quality=75)
realcomp2 = REALCOMP(quality=75)

net.load_state_dict(torch.load("D:\\code\\machine learning\\backup\cloud\\3\\"+method+".pth"))
#net.load_state_dict(torch.load("D:\\code\\machine learning\\CCNNCAR\\"+method+".pth"))

# 禁用自动求导
with torch.no_grad():
    # 将模型设置为评估模式
    net.eval()
    currentloss = 0
    valid_num = 0
    ssim_values = 0
    psnr_values = 0
    for i_batch, batch_data in enumerate(dataloader):
            # amp_hr = batch_data[0]
            # phs_hr = batch_data[1]
            # amp_lr = batch_data[2]
            # phs_lr = batch_data[3]   

            # amp_hr = amp_hr.float()
            # phs_hr = phs_hr.float()
            # #amp_lr = amp_lr.float()
            # #phs_lr = phs_lr.float()

            # amp_hr = amp_hr.cuda() 
            # phs_hr = phs_hr.cuda()
            # #amp_lr = amp_lr.cuda()
            # #phs_lr = phs_lr.cuda()

            hr_in = batch_data['GT'].cuda()

            # if method == 'caetad' or 'caetadmix':
            #     lr_in = torch.complex(amp_hr * torch.cos((phs_hr-0.5) * 2.0 * np.pi), 
            #                 amp_hr * torch.sin((phs_hr-0.5) * 2.0 * np.pi)
            #     )
            # elif method == 'aetad' or 'aetadmix':
            #     lr_in = torch.complex(amp_hr,phs_hr)
            
            # holo_hr = torch.complex(amp_hr * torch.cos((phs_hr-0.5) * 2.0 * np.pi), 
            #             amp_hr * torch.sin((phs_hr-0.5) * 2.0 * np.pi)
            #             )
            
            holo_dr = net.encode(hr_in)
            #holo_q_real = Quan(holo_dr.real)
            #holo_q_imag = Quan(holo_dr.imag)
            #holo_j_real = diffcomp1(holo_q_real)
            #holo_j_imag = diffcomp2(holo_q_imag)
            #holo_j_real = realcomp1(holo_q_real)
            #holo_j_imag = realcomp2(holo_q_imag)
            #holo_j = torch.complex(holo_j_real,holo_j_imag)
            #holo_sr = net.decode(holo_j)
            holo_sr = net.decode(holo_dr)

            if method == 'caetad':
                sr_recon_complex = propagation_ASM(u_in=holo_sr, z=-0.02, linear_conv=hologram_params["pad"],
                                                feature_size=hologram_params["pitch"],
                                                wavelength=hologram_params["wavelengths"],
                                                precomped_H=Hbackward)
                sr_recon_amp = torch.abs(sr_recon_complex)
            elif method == 'aetad':
                holo_sr_complex = torch.complex(holo_sr.real * torch.cos((holo_sr.imag-0.5) * 2.0 * np.pi), 
                        holo_sr.real * torch.sin((holo_sr.imag-0.5) * 2.0 * np.pi)
                        )
                sr_recon_complex = propagation_ASM(u_in=holo_sr_complex, z=-0.02, linear_conv=hologram_params["pad"],
                                                feature_size=hologram_params["pitch"],
                                                wavelength=hologram_params["wavelengths"],
                                                precomped_H=Hbackward)
                sr_recon_amp = torch.abs(sr_recon_complex)
            elif method == 'caetadmix' or 'aetadmix':
                sr_recon_complex = propagation_ASM(u_in=holo_sr, z=-0.02, linear_conv=hologram_params["pad"],
                                                feature_size=hologram_params["pitch"],
                                                wavelength=hologram_params["wavelengths"],
                                                precomped_H=Hbackward)
                sr_recon_amp = torch.abs(sr_recon_complex)
            # elif method == 'caetadmix' or 'aetadmix':
            #     sr_recon_amp = holo_sr
            elif method == 'caetadmix_po':
                holo_sr_complex = torch.complex(torch.cos(holo_sr),torch.sin(holo_sr))
                sr_recon_complex = propagation_ASM(u_in=holo_sr_complex, z=-0.02, linear_conv=hologram_params["pad"],
                                                feature_size=hologram_params["pitch"],
                                                wavelength=hologram_params["wavelengths"],
                                                precomped_H=Hbackward)
                sr_recon_amp = torch.abs(sr_recon_complex)
            
            hr_recon_complex = propagation_ASM(u_in=hr_in, z=-0.02, linear_conv=hologram_params["pad"],
                                            feature_size=hologram_params["pitch"],
                                            wavelength=hologram_params["wavelengths"],
                                            precomped_H=Hbackward)
            
            hr_recon_amp = torch.abs(hr_recon_complex)

            ssim_value = criterion2(sr_recon_amp, hr_recon_amp)
            print("ssim:", ssim_value.detach().cpu().numpy())
            ssim_values = ssim_values + ssim_value.detach().cpu().numpy()
            psnr_value = psnr(sr_recon_amp.detach().cpu().numpy(), hr_recon_amp.detach().cpu().numpy())
            print("psnr:", psnr_value)
            psnr_values = psnr_values + psnr_value
            loss = criterion1(sr_recon_amp, hr_recon_amp)
            print("loss:", loss.detach().cpu().numpy())
            currentloss = currentloss + loss.detach().cpu().numpy()
            valid_num += 1
    print('validloss:', currentloss / valid_num)
    print("ssim:", ssim_values / valid_num)
    print("psnr:", psnr_values / valid_num)


