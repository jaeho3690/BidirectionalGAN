import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from model import BiGAN


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--work_type', type=str, default='train', help="choose work type 'train' or 'test'")
    parser.add_argument('--epochs', default=400, type=int,help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int,help='mini-batch size (default: 32)')
    parser.add_argument('--early_stopping', default=50, type=int,
                        metavar='N', help='early stopping (default: 50)')
    
    # Model
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder')
    parser.add_argument('--generator_lr', type=float, default=4e-4, help='learning rate for generator')
    parser.add_argument('--discriminator_lr', type=float, default=4e-4, help='learning rate for discriminator')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension of z')

    # Data
    parser.add_argument('--input_size', type=int, default=28, help='image size')


    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    return config


def main():
    config = parse_args()

    # MNIST dataset 
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=config.batch_size,shuffle=False)

    # Model
    model = BiGAN(config)
    model.train(train_loader)

if __name__ == '__main__':
    main()