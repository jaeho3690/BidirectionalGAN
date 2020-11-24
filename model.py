import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.optim import lr_scheduler

from modules import Generator,Encoder,Discriminator

class BiGAN(nn.Module):
    def __init__(self,config):
        super(BiGAN,self).__init__()

        self._work_type = config.work_type
        self._epochs = config.epochs
        self._batch_size = config.batch_size

        self._encoder_lr = config.encoder_lr
        self._generator_lr = config.generator_lr
        self._discriminator_lr = config.discriminator_lr
        self._latent_dim = config.latent_dim
        self._weight_decay = config.weight_decay

        self._img_shape = (config.input_size,config.input_size)
        self._img_save_path = config.image_save_path
        self._model_save_path = config.model_save_path
        self._device = config.device

        if self._work_type == 'train':
            # Loss function
            self._adversarial_criterion = torch.nn.MSELoss()

            # Initialize generator, encoder and discriminator
            self._G = Generator(self._latent_dim,self._img_shape).to(self._device)
            self._E = Encoder(self._latent_dim,self._img_shape).to(self._device)
            self._D = Discriminator(self._latent_dim,self._img_shape).to(self._device)

            self._G.apply(self.weights_init)
            self._E.apply(self.weights_init)
            self._D.apply(self.discriminator_weights_init)

            self._G_optimizer = torch.optim.Adam([{'params' : self._G.parameters()},{'params' : self._E.parameters()}],
                                                lr=self._generator_lr,betas=(0.5,0.999),weight_decay=self._weight_decay)
            self._D_optimizer = torch.optim.Adam(self._D.parameters(),lr=self._discriminator_lr,betas=(0.5,0.999))
            
            self._G_scheduler = lr_scheduler.ExponentialLR(self._G_optimizer, gamma= 0.99) 
            self._D_scheduler = lr_scheduler.ExponentialLR(self._D_optimizer, gamma= 0.99) 

    def train(self,train_loader):
        Tensor = torch.cuda.FloatTensor if self._device == 'cuda' else torch.FloatTensor
        n_total_steps = len(train_loader)
        for epoch in range(self._epochs):
            self._G_scheduler.step()
            self._D_scheduler.step()

            for i, (images, _) in enumerate(train_loader):
                # Adversarial ground truths
                valid = Variable(Tensor(images.size(0), 1).fill_(1), requires_grad=False)
                fake = Variable(Tensor(images.size(0), 1).fill_(0), requires_grad=False)

                
                # ---------------------
                # Train Encoder
                # ---------------------
                
                # Configure input
                images = images.reshape(-1,np.prod(self._img_shape)).to(self._device)

                # z_ is encoded latent vector
                (original_img,z_)= self._E(images)
                predict_encoder = self._D(original_img,z_)
  

                # ---------------------
                # Train Generator
                # ---------------------
                
                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0],self._latent_dim))))
                (gen_img,z)=self._G(z)
                predict_generator = self._D(gen_img,z)
                                                                                                               
                G_loss = (self._adversarial_criterion(predict_generator,valid)+self._adversarial_criterion(predict_encoder,fake)) *0.5   

                self._G_optimizer.zero_grad()
                G_loss.backward()
                self._G_optimizer.step()         

                # ---------------------
                # Train Discriminator
                # ---------------------

                z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0],self._latent_dim))))
                (gen_img,z)=self._G(z)
                (original_img,z_)= self._E(images)
                predict_encoder = self._D(original_img,z_)
                predict_generator = self._D(gen_img,z)

                D_loss = (self._adversarial_criterion(predict_encoder,valid)+self._adversarial_criterion(predict_generator,fake)) *0.5                
                
                self._D_optimizer.zero_grad()
                D_loss.backward()
                self._D_optimizer.step()

                

                
                if i % 100 == 0:
                    print (f'Epoch [{epoch+1}/{self._epochs}], Step [{i+1}/{n_total_steps}]')
                    print (f'Generator Loss: {G_loss.item():.4f} Discriminator Loss: {D_loss.item():.4f}')
 
                if i % 400 ==0:
                    vutils.save_image(gen_img.unsqueeze(1).cpu().data[:16, ], f'{self._img_save_path}/E{epoch}_Iteration{i}_fake.png')
                    vutils.save_image(original_img.unsqueeze(1).cpu().data[:16, ], f'{self._img_save_path}/E{epoch}_Iteration{i}_real.png')
                    print('image saved')
                    print('')




    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.bias.data.fill_(0)

    def discriminator_weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.5)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.bias.data.fill_(0)



