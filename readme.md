# Bidirectional GAN (Adversarial Feature Learning)- Pytorch implementation
This is the pytorch implementation of Bidirectional GAN(BiGAN). Unlike ordinary GANs which are focused on generating data, the Bidirectional GAN is focused on creating an embedding of the original data. The model has an additional "Encoder" Structure from the original GAN which helps to encode original data. The descriminator will then discriminate the joint distribution of the latent vector and original data. The [paper]((https://arxiv.org/abs/1605.09782)) theoretically proves that the latent embedding will become an inverse mapping of the original input data when trained properly. 

# Implementation Notes
There are not many, but a few, references for BiGAN. However, I wasn't able to find a pytorch version and without a convolutional structure. My codes are implemented in MLP style where the image is flattened. The model structure were referenced from this great [github](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py)repo. However, I made some minor adjustments as the implemented model struck on mode collapse.

### Requirements
Dependencies (with python=3.6)
pytorch = 1.6.0
torchvision = 0.7.0

### Installation
```
pip install requirements.txt
```

### Run
```
python main.py
```

### Results
**Epoch 1**  
![Image1](/figures/E1_Iteration0_fake.png){: width="100" height="100"}    
**Epoch 69**  
![Image1](/figures/E69_Iteration400_fake.png){: width="100" height="100"}    
**Epoch 250**  
![Image1](/figures/E250_Iteration400_fake.png){: width="100" height="100"}    
**Epoch 399**  
![Image1](/figures/E399_Iteration400_fake.png)  {: width="100" height="100"}  