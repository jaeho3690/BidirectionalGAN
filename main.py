import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from model import BiGAN


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--work_type', type=str, default='train', help="choose work type 'train' or 'test'")
    parser.add_argument('--epochs', default=400, type=int,help='number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int,help='mini-batch size (default: 32)')
    parser.add_argument('--early_stopping', default=50, type=int,
                        metavar='N', help='early stopping (default: 50)')
    
    # Model
    parser.add_argument('--encoder_lr', type=float, default=2e-4, help='learning rate for encoder')
    parser.add_argument('--generator_lr', type=float, default=2e-4, help='learning rate for generator')
    parser.add_argument('--discriminator_lr', type=float, default=2e-4, help='learning rate for discriminator')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension of z')
    parser.add_argument('--weight_decay', type=float, default=2.5*1e-5, help='Weight decay')
    # Data
    parser.add_argument('--input_size', type=int, default=28, help='image size')
    parser.add_argument('--image_save_path', type=str, default='saved/generated_images', help='generated image save path')
    parser.add_argument('--model_save_path', type=str, default='saved/model_weight', help='model save path')

    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    return config


def main():
    config = parse_args()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5,], [0.5,])])
    # MNIST dataset 
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform,download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data',train=False,transform=transform)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=config.batch_size,shuffle=False)

    # Model
    model = BiGAN(config)
    model.train(train_loader)

if __name__ == '__main__':
    main()