import torch
import numpy as np

from torch import nn
import torch.nn.functional as F 


class BottleneckResidual(nn.Module):
   def __init__(self, c, g, stride=1, conv_1x1=False):
      super().__init__()

      self.bottleneck = nn.Sequential(
         nn.LazyConv2d(c, kernel_size=1, groups=g),
         nn.LazyBatchNorm2d(),
         nn.ReLU(),

         nn.LazyConv2d(c, kernel_size=3, stride=stride, groups=g, padding=1),
         nn.LazyBatchNorm2d(),
         nn.ReLU(),

         nn.LazyConv2d(c, kernel_size=1, groups=g),
         nn.LazyBatchNorm2d(),
         nn.ReLU()
      )

      self.conv_1x1 = conv_1x1

      if self.conv_1x1:
         self.identity_conv = nn.LazyConv2d(c, kernel_size=1)
      

   def forward(self, x: torch.Tensor) -> torch.Tensor:
      y = self.bottleneck(x)

      if self.conv_1x1:
         x = self.identity_conv(x)
         
      print(y.shape, x.shape)
      return F.relu(y + x)

class KeypointDetector(nn.Module):
   def __init__(self):
      super().__init__()
      
      self.model = nn.Sequential(
         # Initial Block
         nn.LazyConv2d(32, kernel_size=7, stride=5, padding=3),
         nn.LazyBatchNorm2d(),
         nn.ReLU(),
         
         nn.LazyConv2d(64, kernel_size=3, stride=2, padding=3),
         nn.LazyBatchNorm2d(),
         nn.ReLU(),
         
         # Bottleneck Blocks
         self.block(64, 2),
         self.block(128,2),
         self.block(256,2),
         self.block(512, 2),
         
         
         # Fully connected
         nn.Flatten(),
         nn.LazyLinear(512),
         nn.ReLU(),
         nn.LazyLinear(256),
         nn.ReLU(),
         nn.LazyLinear(16),
         nn.ReLU(),
      )
   
   def block(self, c: int, num_blocks: int) -> nn.Module:
      layers = []

      for i in range(num_blocks):
         layers.append(BottleneckResidual(c, 4, conv_1x1=(i==0)))

      return nn.Sequential(*layers)
   
   def forward(self, x) -> torch.Tensor:
      return self.model(x)
      
from torchsummary import summary

test = KeypointDetector()

summary(test, ( 3, 80, 80))