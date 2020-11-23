import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class Generator(nn.Module):
    def __init__(self,latent_dim,img_shape):
        super(Generator,self).__init__()
        self.img_shape= img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=True),
            *block(128, 256),
            *block(256, 512),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )
    def forward(self, z):
        # img:[batch_size,img_size,img_size], z:[batch_size,latent_dim]
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return (img,z)


class Encoder(nn.Module):
    def __init__(self,latent_dim,img_shape):
        super(Encoder,self).__init__()
        self.img_shape= img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)),512),
            *block(512, 256, normalize=True),
            *block(256, 128),
            nn.Linear(128, latent_dim),
            nn.Tanh()
        )

    def forward(self, img):
        # img:[batch_size,img_size,img_size], z:[batch_size,latent_dim]
        z = self.model(img)
        img = img.view(img.size(0), *self.img_shape)
        
        return (img,z)
    
class Discriminator(nn.Module):
    def __init__(self, latent_dim,img_shape):
        super(Discriminator, self).__init__()

        joint_shape = latent_dim + np.prod(img_shape)

        self.model = nn.Sequential(
            nn.Linear(joint_shape, 512),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, z):
        # img:[batch_size,img_size,img_size], z:[batch_size,latent_dim]
        joint = torch.cat((img.view(img.size(0),-1),z),dim=1)
        validity = self.model(joint)

        return validity


