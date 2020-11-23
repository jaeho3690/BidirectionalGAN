import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms

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

        self._img_shape = (config.input_size,config.input_size)
        self._device = config.device

        if self._work_type == 'train':
            # Loss function
            self._adversarial_criterion = torch.nn.BCELoss()

            # Initialize generator, encoder and discriminator
            self._G = Generator(self._latent_dim,self._img_shape).to(self._device)
            self._E = Encoder(self._latent_dim,self._img_shape).to(self._device)
            self._D = Discriminator(self._latent_dim,self._img_shape).to(self._device)

            self._G_optimizer = torch.optim.Adam(self._G.parameters(),lr=self._generator_lr)
            self._E_optimizer = torch.optim.Adam(self._E.parameters(),lr=self._encoder_lr)
            self._D_optimizer = torch.optim.Adam(self._D.parameters(),lr=self._discriminator_lr)
            

    def train(self,train_loader):
        Tensor = torch.cuda.FloatTensor if self._device == 'cuda' else torch.FloatTensor
        n_total_steps = len(train_loader)
        for epoch in range(self._epochs):
            for i, (images, _) in enumerate(train_loader):
                # Adversarial ground truths
                valid = Variable(Tensor(images.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(images.size(0), 1).fill_(0.0), requires_grad=False)
                
                # ---------------------
                # Train Encoder
                # ---------------------
                
                # Configure input
                images = images.reshape(-1,np.prod(self._img_shape)).to(self._device)
                
                #self._E_optimizer.zero_grad()
                (original_img,enc_img)= self._E(images)
                predict_encoder = self._D(original_img,enc_img)
                E_loss = self._adversarial_criterion(predict_encoder,valid)
                #E_loss.backward()
                #self._E_optimizer.step()

                # ---------------------
                # Train Generator
                # ---------------------

                self._G_optimizer.zero_grad()
                # Sample noise as generator input
                real_noise = Variable(Tensor(np.random.normal(0, 1, (images.shape[0],self._latent_dim))))
                (gen_img,real_noise)=self._G(real_noise)
                predict_generator = self._D(gen_img,real_noise)
                G_loss = self._adversarial_criterion(predict_generator,fake)
                G_loss.backward(retain_graph=True)
                


                # ---------------------
                # Train Discriminator
                # ---------------------
                self._D_optimizer.zero_grad()
                # 이거 이렇게 해도 되나...???
                D_loss = (self._adversarial_criterion(predict_encoder,valid)+self._adversarial_criterion(predict_generator,fake)) *0.5                
                D_loss.backward()

                self._G_optimizer.step() 
                self._D_optimizer.step()
                
                if (i) % 100 == 0:
                    print (f'Epoch [{epoch+1}/{self._epochs}], Step [{i+1}/{n_total_steps}]')
                    print (f'Encoder Loss: {E_loss.item():.4f} Generator Loss: {G_loss.item():.4f} Discriminator Loss: {D_loss.item():.4f}')
 
               



    def test(self,test_loader):
        pass


