import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from utils.propagation_ASM import *
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
import numpy as np



wavelengths = np.ones((192,192,3))
wavelengths = wavelengths * np.array([0.000450, 0.000520, 0.000638])
wavelengths = np.transpose(wavelengths,(2,0,1))

hologram_params = {
        "wavelengths" : wavelengths,  # laser wavelengths in BGR order
        "pitch" : 0.008,                                           # hologram pitch
        "res_h" : 192,                                 # dataset image height
        "res_w" : 192,                                 # dataset image width
        "pad" : False,
        "channels" : 3                               # the channels of image
    }

class CAETADMIX(nn.Module):
    def __init__(self):
        super(CAETADMIX, self).__init__()
        # Build encoding part.
        self._downscaling = nn.Sequential(
            ComplexConv2d(3, 8, 3, stride=1, padding=1),
            ComplexConv2d(8, 16, 3, stride=1, padding=1),
            _ReversePixelShuffle_(downscale_factor=2),
        )
        Hbackward = propagation_ASM(torch.empty(1, 1, hologram_params["res_w"], hologram_params["res_h"]), 
                            feature_size=[hologram_params["pitch"], hologram_params["pitch"]],
                            wavelength=hologram_params["wavelengths"],
                            z = -0.02, linear_conv=hologram_params["pad"], return_H=True)
        Hbackward = Hbackward.cuda()
        self.Hbackward = Hbackward
        self._res_en1 = _ComplexResblock_(64)
        self._res_en2 = _ComplexResblock_(64)
        self._res_en3 = _ComplexResblock_(64)
        self._conv_en1 = ComplexConv2d(64, 64, 3, stride=1, padding=1)
        self._conv_en2 = ComplexConv2d(64, 3, 3, stride=1, padding=1)
        # Build decoding part.
        self._conv_de1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self._res_de1 = _Resblock_(64)
        self._res_de2 = _Resblock_(64)
        self._res_de3 = _Resblock_(64)
        self._conv_de2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self._upscaling = nn.Sequential(
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:              # b, 3, p, p
        x = self._downscaling(x)                                    # b, 64, p/2, p/2
        residual = x
        x = self._res_en1.forward(x)                                # b, 64, p/2, p/2
        x = self._res_en2.forward(x)                                # b, 64, p/2, p/2
        x = self._res_en3.forward(x)                                # b, 64, p/2, p/2
        x = self._conv_en1(x)                                       # b, 64, p/2, p/2
        x = torch.add(residual, x)                                  # b, 64, p/2, p/2
        x = self._conv_en2(x)                                       # b, 3, p/2, p/2
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_de1(x)                                       # b, 64, p/2, p/2
        residual = x
        x = self._res_de1.forward(x)                                # b, 64, p/2, p/2
        x = self._res_de2.forward(x)                                # b, 64, p/2, p/2
        x = self._res_de3.forward(x)                                # b, 64, p/2, p/2
        x = self._conv_de2(x)                                       # b, 64, p/2, p/2
        x = torch.add(residual, x)                                  # b, 64, p/2, p/2
        x = self._upscaling(x)                                      # b, 3, p, p
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        sr_recon_complex = propagation_ASM(u_in=x, z=-0.02, linear_conv=hologram_params["pad"],
                                    feature_size=hologram_params["pitch"],
                                    wavelength=hologram_params["wavelengths"],
                                    precomped_H=self.Hbackward)
        x = torch.abs(sr_recon_complex)
        #sr_po = torch.atan2(x.imag, x.real)
        x = self.decode(x)
        return x
class CAETADMIX_po(nn.Module):
    def __init__(self):
        super(CAETADMIX_po, self).__init__()
        # Build encoding part.
        self._downscaling = nn.Sequential(
            ComplexConv2d(3, 8, 3, stride=1, padding=1),
            ComplexConv2d(8, 16, 3, stride=1, padding=1),
            _ReversePixelShuffle_(downscale_factor=2),
        )
        Hbackward = propagation_ASM(torch.empty(1, 1, hologram_params["res_w"], hologram_params["res_h"]), 
                            feature_size=[hologram_params["pitch"], hologram_params["pitch"]],
                            wavelength=hologram_params["wavelengths"],
                            z = -0.02, linear_conv=hologram_params["pad"], return_H=True)
        Hbackward = Hbackward.cuda()
        self.Hbackward = Hbackward
        self._res_en1 = _ComplexResblock_(64)
        self._res_en2 = _ComplexResblock_(64)
        self._res_en3 = _ComplexResblock_(64)
        self._conv_en1 = ComplexConv2d(64, 64, 3, stride=1, padding=1)
        self._conv_en2 = ComplexConv2d(64, 3, 3, stride=1, padding=1)
        # Build decoding part.
        self._conv_de1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self._res_de1 = _Resblock_(64)
        self._res_de2 = _Resblock_(64)
        self._res_de3 = _Resblock_(64)
        self._conv_de2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self._upscaling = nn.Sequential(
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:              # b, 3, p, p
        x = self._downscaling(x)                                    # b, 64, p/2, p/2
        residual = x
        x = self._res_en1.forward(x)                                # b, 64, p/2, p/2
        x = self._res_en2.forward(x)                                # b, 64, p/2, p/2
        x = self._res_en3.forward(x)                                # b, 64, p/2, p/2
        x = self._conv_en1(x)                                       # b, 64, p/2, p/2
        x = torch.add(residual, x)                                  # b, 64, p/2, p/2
        x = self._conv_en2(x)                                       # b, 3, p/2, p/2
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_de1(x)                                       # b, 64, p/2, p/2
        residual = x
        x = self._res_de1.forward(x)                                # b, 64, p/2, p/2
        x = self._res_de2.forward(x)                                # b, 64, p/2, p/2
        x = self._res_de3.forward(x)                                # b, 64, p/2, p/2
        x = self._conv_de2(x)                                       # b, 64, p/2, p/2
        x = torch.add(residual, x)                                  # b, 64, p/2, p/2
        x = self._upscaling(x)                                      # b, 3, p, p
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        # sr_recon_complex = propagation_ASM(u_in=x, z=-0.02, linear_conv=hologram_params["pad"],
        #                             feature_size=hologram_params["pitch"],
        #                             wavelength=hologram_params["wavelengths"],
        #                             precomped_H=self.Hbackward)
        # x = torch.abs(sr_recon_complex)
        sr_po = torch.atan2(x.imag, x.real)
        x = self.decode(sr_po)
        return x
# complex AETAD
class CAETAD(nn.Module):

    def __init__(self):
        super(CAETAD, self).__init__()
        # Build encoding part.
        self._downscaling = nn.Sequential(
            ComplexConv2d(3, 8, 3, stride=1, padding=1),
            ComplexConv2d(8, 16, 3, stride=1, padding=1),
            _ReversePixelShuffle_(downscale_factor=2),
        )
        self._res_en1 = _ComplexResblock_(64)
        self._res_en2 = _ComplexResblock_(64)
        self._res_en3 = _ComplexResblock_(64)
        self._conv_en1 = ComplexConv2d(64, 64, 3, stride=1, padding=1)
        self._conv_en2 = ComplexConv2d(64, 3, 3, stride=1, padding=1)
        # Build decoding part.
        self._conv_de1 = ComplexConv2d(3, 64, 3, stride=1, padding=1)
        self._res_de1 = _ComplexResblock_(64)
        self._res_de2 = _ComplexResblock_(64)
        self._res_de3 = _ComplexResblock_(64)
        self._conv_de2 = ComplexConv2d(64, 64, 3, stride=1, padding=1)
        self._upscaling = nn.Sequential(
            ComplexConv2d(64, 256, 3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            ComplexConv2d(64, 3, 3, stride=1, padding=1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:              # b, 3, p, p
        x = self._downscaling(x)                                    # b, 64, p/2, p/2
        residual = x
        x = self._res_en1.forward(x)                                # b, 64, p/2, p/2
        x = self._res_en2.forward(x)                                # b, 64, p/2, p/2
        x = self._res_en3.forward(x)                                # b, 64, p/2, p/2
        x = self._conv_en1(x)                                       # b, 64, p/2, p/2
        x = torch.add(residual, x)                                  # b, 64, p/2, p/2
        x = self._conv_en2(x)                                       # b, 3, p/2, p/2
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_de1(x)                                       # b, 64, p/2, p/2
        residual = x
        x = self._res_de1.forward(x)                                # b, 64, p/2, p/2
        x = self._res_de2.forward(x)                                # b, 64, p/2, p/2
        x = self._res_de3.forward(x)                                # b, 64, p/2, p/2
        x = self._conv_de2(x)                                       # b, 64, p/2, p/2
        x = torch.add(residual, x)                                  # b, 64, p/2, p/2
        x = self._upscaling(x)                                      # b, 3, p, p
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
# orginal AETAD mix
class AETADMIX(nn.Module):

    def __init__(self):
        super(AETADMIX, self).__init__()
        # Build encoding part for amp part of hologram.
        Hbackward = propagation_ASM(torch.empty(1, 1, hologram_params["res_w"], hologram_params["res_h"]), 
                            feature_size=[hologram_params["pitch"], hologram_params["pitch"]],
                            wavelength=hologram_params["wavelengths"],
                            z = -0.02, linear_conv=hologram_params["pad"], return_H=True)
        Hbackward = Hbackward.cuda()
        self.Hbackward = Hbackward
        self._downscaling1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1),
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            _ReversePixelShuffle_(downscale_factor=2),
        )
        self._res_en1 = _Resblock_(64)
        self._res_en2 = _Resblock_(64)
        self._res_en3 = _Resblock_(64)
        self._conv_en1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self._conv_en2 = nn.Conv2d(64, 3, 3, stride=1, padding=1)

        self._downscaling2 = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1),
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            _ReversePixelShuffle_(downscale_factor=2),
        )
        self._res_en4 = _Resblock_(64)
        self._res_en5 = _Resblock_(64)
        self._res_en6 = _Resblock_(64)
        self._conv_en3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self._conv_en4 = nn.Conv2d(64, 3, 3, stride=1, padding=1)
                # Build decoding part for amp part of hologram.
        self._conv_de1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self._res_de1 = _Resblock_(64)
        self._res_de2 = _Resblock_(64)
        self._res_de3 = _Resblock_(64)
        self._conv_de2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self._upscaling1 = nn.Sequential(
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:              # b, 3, p, p
        amp = self._downscaling1(x.real)                                    # b, 64, p/2, p/2
        residual1 = amp
        amp = self._res_en1.forward(amp)                                # b, 64, p/2, p/2
        amp = self._res_en2.forward(amp)                                # b, 64, p/2, p/2
        amp = self._res_en3.forward(amp)                                # b, 64, p/2, p/2
        amp = self._conv_en1(amp)                                       # b, 64, p/2, p/2
        amp = torch.add(residual1, amp)                                  # b, 64, p/2, p/2
        amp = self._conv_en2(amp)                                       # b, 3, p/2, p/2

        phs = self._downscaling2(x.imag)
        residual2 = phs
        phs = self._res_en4.forward(phs)                                # b, 64, p/2, p/2
        phs = self._res_en5.forward(phs)                                # b, 64, p/2, p/2
        phs = self._res_en6.forward(phs)                                # b, 64, p/2, p/2
        phs = self._conv_en3(phs)                                       # b, 64, p/2, p/2
        phs = torch.add(residual2, phs)                                  # b, 64, p/2, p/2
        phs = self._conv_en4(phs)                                       # b, 3, p/2, p/2
        x = torch.complex(amp,phs)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_de1(x)                                       # b, 64, p/2, p/2
        residual1 = x
        x = self._res_de1.forward(x)                                # b, 64, p/2, p/2
        x = self._res_de2.forward(x)                                # b, 64, p/2, p/2
        x = self._res_de3.forward(x)                                # b, 64, p/2, p/2
        x = self._conv_de2(x)                                       # b, 64, p/2, p/2
        x = torch.add(residual1, x)                                  # b, 64, p/2, p/2
        x = self._upscaling1(x)                                      # b, 3, p, p

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        holo_sr_complex = torch.complex(x.real * torch.cos((x.imag-0.5) * 2.0 * np.pi), 
                        x.real * torch.sin((x.imag-0.5) * 2.0 * np.pi)
                        )
        sr_recon_complex = propagation_ASM(u_in=holo_sr_complex, z=-0.02, linear_conv=hologram_params["pad"],
                                    feature_size=hologram_params["pitch"],
                                    wavelength=hologram_params["wavelengths"],
                                    precomped_H=self.Hbackward)
        x = torch.abs(sr_recon_complex)
        x = self.decode(x)

        return x
# orginal AETAD
class AETAD(nn.Module):

    def __init__(self):
        super(AETAD, self).__init__()
        # Build encoding part for amp part of hologram.
        self._downscaling1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1),
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            _ReversePixelShuffle_(downscale_factor=2),
        )
        self._res_en1 = _Resblock_(64)
        self._res_en2 = _Resblock_(64)
        self._res_en3 = _Resblock_(64)
        self._conv_en1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self._conv_en2 = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        # Build decoding part for amp part of hologram.
        self._conv_de1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self._res_de1 = _Resblock_(64)
        self._res_de2 = _Resblock_(64)
        self._res_de3 = _Resblock_(64)
        self._conv_de2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self._upscaling1 = nn.Sequential(
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

        self._downscaling2 = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1),
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            _ReversePixelShuffle_(downscale_factor=2),
        )
        self._res_en4 = _Resblock_(64)
        self._res_en5 = _Resblock_(64)
        self._res_en6 = _Resblock_(64)
        self._conv_en3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self._conv_en4 = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        # Build decoding part.
        self._conv_de3 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self._res_de4 = _Resblock_(64)
        self._res_de5 = _Resblock_(64)
        self._res_de6 = _Resblock_(64)
        self._conv_de4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self._upscaling2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:              # b, 3, p, p
        amp = self._downscaling1(x.real)                                    # b, 64, p/2, p/2
        residual1 = amp
        amp = self._res_en1.forward(amp)                                # b, 64, p/2, p/2
        amp = self._res_en2.forward(amp)                                # b, 64, p/2, p/2
        amp = self._res_en3.forward(amp)                                # b, 64, p/2, p/2
        amp = self._conv_en1(amp)                                       # b, 64, p/2, p/2
        amp = torch.add(residual1, amp)                                  # b, 64, p/2, p/2
        amp = self._conv_en2(amp)                                       # b, 3, p/2, p/2

        phs = self._downscaling2(x.imag)
        residual2 = phs
        phs = self._res_en4.forward(phs)                                # b, 64, p/2, p/2
        phs = self._res_en5.forward(phs)                                # b, 64, p/2, p/2
        phs = self._res_en6.forward(phs)                                # b, 64, p/2, p/2
        phs = self._conv_en3(phs)                                       # b, 64, p/2, p/2
        phs = torch.add(residual2, phs)                                  # b, 64, p/2, p/2
        phs = self._conv_en4(phs)                                       # b, 3, p/2, p/2
        x = torch.complex(amp,phs)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        amp = self._conv_de1(x.real)                                       # b, 64, p/2, p/2
        residual1 = amp
        amp = self._res_de1.forward(amp)                                # b, 64, p/2, p/2
        amp = self._res_de2.forward(amp)                                # b, 64, p/2, p/2
        amp = self._res_de3.forward(amp)                                # b, 64, p/2, p/2
        amp = self._conv_de2(amp)                                       # b, 64, p/2, p/2
        amp = torch.add(residual1, amp)                                  # b, 64, p/2, p/2
        amp = self._upscaling1(amp)                                      # b, 3, p, p

        phs = self._conv_de1(x.imag)                                       # b, 64, p/2, p/2
        residual2 = phs
        phs = self._res_de1.forward(phs)                                # b, 64, p/2, p/2
        phs = self._res_de2.forward(phs)                                # b, 64, p/2, p/2
        phs = self._res_de3.forward(phs)                                # b, 64, p/2, p/2
        phs = self._conv_de2(phs)                                       # b, 64, p/2, p/2
        phs = torch.add(residual2, phs)                                  # b, 64, p/2, p/2
        phs = self._upscaling1(phs)                                      # b, 3, p, p
        x = torch.complex(amp,phs)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# =============================================================================
# Customly implemented building blocks (nn.Modules).
# =============================================================================
class _ComplexResblock_(nn.Module):
    """ Residual convolutional block consisting of two convolutional
    layers, a RELU activation in between and a residual connection from
    start to end. The inputs size (=s) is therefore contained. The number
    of channels is contained as well, but can be adapted (=c). """

    __constants__ = ['channels']

    def __init__(self, c):
        super(_ComplexResblock_, self).__init__()
        self.conv1 = ComplexConv2d(c, c, 3, stride=1, padding=1, bias=True)     # b, c, s, s
        self.conv2 = ComplexConv2d(c, c, 3, stride=1, padding=1, bias=True)      # b, c, s, s

        self.channels = c

    def forward(self, x):
        x = self.conv1(x)
        x = complex_relu(x)
        x = self.conv2(x)
        return x

    def extra_repr(self):
        return 'channels={}'.format(self.channels)

class _Resblock_(nn.Module):
    """ Residual convolutional block consisting of two convolutional
    layers, a RELU activation in between and a residual connection from
    start to end. The inputs size (=s) is therefore contained. The number
    of channels is contained as well, but can be adapted (=c). """

    __constants__ = ['channels']

    def __init__(self, c):
        super(_Resblock_, self).__init__()
        self.filter_block = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=1, padding=1, bias=True),     # b, c, s, s
            nn.ReLU(True),                                          # b, c, s, s
            nn.Conv2d(c, c, 3, stride=1, padding=1, bias=True)      # b, c, s, s
        )
        self.channels = c

    def forward(self, x):
        return x + self.filter_block(x)

    def extra_repr(self):
        return 'channels={}'.format(self.channels)

class _ReversePixelShuffle_(nn.Module):
    """ Reverse pixel shuffeling module, i.e. rearranges elements in a tensor
    of shape (*, C, H*r, W*r) to (*, C*r^2, H, W). Inverse implementation according
    to https://pytorch.org/docs/0.3.1/_modules/torch/nn/functional.html#pixel_shuffle. """

    __constants__ = ['downscale_factor']

    def __init__(self, downscale_factor):
        super(_ReversePixelShuffle_, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        _, c, h, w = input.shape
        assert all([x % self.downscale_factor == 0 for x in [h, w]])
        return self.inv_pixel_shuffle(input, self.downscale_factor)

    def extra_repr(self):
        return 'downscale_factor={}'.format(self.downscale_factor)

    @staticmethod
    def inv_pixel_shuffle(input, downscale_factor):
        batch_size, in_channels, height, width = input.size()
        out_channels = in_channels * (downscale_factor ** 2)
        height //= downscale_factor
        width //= downscale_factor
        # Reshape input to new shape.
        input_view = input.contiguous().view(
            batch_size, in_channels, height, downscale_factor,
            width, downscale_factor)
        shuffle_out = input_view.permute(0,1,3,5,2,4).contiguous()
        return shuffle_out.view(batch_size, out_channels, height, width)