from torch.utils.data import DataLoader, Dataset
import os
import data.load as load
import torch 
import numpy as np
class dataset(Dataset):
    # image_dir为数据目录，label_file，为标签文件
    def __init__(self, opt, transform):
        super(dataset, self).__init__()    # 添加对父类的初始化
        self.opt = opt
        self.amp_hr_dir = opt['dataroot_GT_amp']         # 图像文件所在路径
        self.phs_hr_dir = opt['dataroot_GT_phs']
        self.amp_lr_dir = opt['dataroot_LQ_amp']
        self.phs_lr_dir = opt['dataroot_LQ_phs']
        self.transform = transform         # 数据转换操作
        self.amp_hr_images = os.listdir(self.amp_hr_dir)#目录里的所有img文件
        self.phs_hr_images = os.listdir(self.phs_hr_dir)
        self.amp_lr_images = os.listdir(self.amp_lr_dir)
        self.phs_lr_images = os.listdir(self.phs_lr_dir)
    
    # 加载每一项数据
    def __getitem__(self, index):
        amp_hr_image_index = self.amp_hr_images[index]    #根据索引index获取该图片
        phs_hr_image_index = self.phs_hr_images[index]
        amp_lr_image_index = self.amp_lr_images[index]
        phs_lr_image_index = self.phs_lr_images[index]
        
        amp_hr_img_path = os.path.join(self.amp_hr_dir, amp_hr_image_index) #获取索引为index的图片的路径名
        phs_hr_img_path = os.path.join(self.phs_hr_dir, phs_hr_image_index)
        amp_lr_img_path = os.path.join(self.amp_lr_dir, amp_lr_image_index)
        phs_lr_img_path = os.path.join(self.phs_lr_dir, phs_lr_image_index)
        

        amp_hr_image = load.load_image(amp_hr_img_path)
        phs_hr_image = load.load_image(phs_hr_img_path)
        amp_lr_image = load.load_image(amp_lr_img_path)
        phs_lr_image = load.load_image(phs_lr_img_path)
        
        if self.transform:
            amp_hr_image = self.transform(amp_hr_image) 
            phs_hr_image = self.transform(phs_hr_image)
            amp_lr_image = self.transform(amp_lr_image)
            phs_lr_image = self.transform(phs_lr_image)
        amp_hr = torch.from_numpy(amp_hr_image)
        phs_hr = torch.from_numpy(phs_hr_image)
        amp_lr = torch.from_numpy(amp_lr_image)
        phs_lr = torch.from_numpy(phs_lr_image)
        hr = torch.complex(amp_hr * torch.cos((phs_hr-0.5) * 2.0 * np.pi), 
                        amp_hr * torch.sin((phs_hr-0.5) * 2.0 * np.pi))
        lr = torch.complex(amp_lr * torch.cos((phs_lr-0.5) * 2.0 * np.pi), 
                        amp_lr * torch.sin((phs_lr-0.5) * 2.0 * np.pi))        
        return {'GT': hr, 'LQ': lr, 'index': amp_hr_image_index}
    
    # 数据集大小
    def __len__(self):
        return (len(self.amp_hr_images))