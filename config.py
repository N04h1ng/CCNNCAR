from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()
config.TRAIN.batch_size = 2 # [16] use 8 if your GPU memory is small
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 2
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 31
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_amp_img_path = 'D:/code/machine learning/Carcgh/MIT/MIT_test/MIT_test_HR/amp'
config.TRAIN.lr_amp_img_path = 'D:/code/machine learning/Carcgh/MIT/MIT_test/MIT_test_LR_bicubic/amp_LR'

config.TRAIN.hr_phs_img_path = 'D:/code/machine learning/Carcgh/MIT/MIT_test/MIT_test_HR/phs'
config.TRAIN.lr_phs_img_path = 'D:/code/machine learning/Carcgh/MIT/MIT_test/MIT_test_LR_bicubic/phs_LR'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'DIV2K/MIT_test/MIT_test_HR/amp'
config.VALID.lr_img_path = 'DIV2K/MIT_test/MIT_test_LR_bicubic/amp_LR'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")