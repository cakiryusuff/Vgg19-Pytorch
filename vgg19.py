# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 20:00:24 2023

@author: cakir
"""

from torch import nn, Tensor
from torchvision.ops import Conv2dNormActivation

class Vgg19(nn.Module):
    def __init__(self, num_classes = 1000):
      super(Vgg19, self).__init__()
      self.planes = [3, 64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]

      self.network = self._make_layer(self.planes)
      self.avgPool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
      self.fc = nn.Sequential(
          nn.Linear(512*7*7, 4096),
          nn.ReLU(),
          nn.Dropout(),

          nn.Linear(4096, 4096),
          nn.ReLU(),
          nn.Dropout(),

          nn.Linear(4096, num_classes),
          )

    def _make_layer(self, planes):
      layers = []
      for i in range(len(self.planes) - 1):
          layers.append(Conv2dNormActivation(planes[i], planes[i + 1],
                                             kernel_size = 3, padding = 1,
                                             norm_layer = nn.BatchNorm2d,
                                             activation_layer = nn.ReLU))
          if i in [1, 3, 7, 11, 15]:
            layers.append(nn.MaxPool2d(3, 2, padding = 1))
      return (nn.Sequential(*layers))

    def forward(self, x: Tensor) -> Tensor:
      x = self.avgPool(self.network(x))
      x = x.view(x.size()[0], -1)
      x = self.fc(x)
      x = nn.functional.softmax(x, dim = 1)

      return x