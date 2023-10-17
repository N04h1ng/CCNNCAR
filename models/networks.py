import logging
import math

from models.modules.model import CAETAD

logger = logging.getLogger('base')


####################
# define network
####################

def define(opt):
    opt_net = opt['network']
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'
    down_num = int(math.log(opt_net['scale'], 2))
    net = CAETAD()
    return net