

### VAE
AE,AutoEncoder
- ./dlkit/models/ae_conv.py:AutoEncoder
- ./dlkit/aeconv1_mnist.py
- ./dlkit/aeconv2_mnist.py

VAE, Variant AutoEncoder
- ./dlkit/models/vae_vanilla.py
- ./dlkit/trainer_vae.py
- ./train_vae_ssl.py:(Semi-Supervised VAE)

CVAE, Conditional VAE, Conditional Variant AutoEncoder and Conv-CVAE, Convolutional Conditional Variant AutoEncoder
- ./dlkit/models/cvae_conv.py:(CNN-based)
- ./dlkit/models/vaeconv_cond.py:(CNN-Based)
- ./dlkit/models/cvae_linear.py:(MLP-based)
- ./dlkit/cvae_mnist.py


### GAN
GAN
- ./dlkit/gan_mnist.py
- ./dlkit/models/gan_conv.py

InfoGAN:
- ./ganinfo_mnist.py
- ./dlkit/models/gan_info.py

### Graph Neural Network

./gnn/*
- GCN, Graph Convolutional Neural Network
- GAT, Graph Attention Neural Network
- explore research about GNN and Multi-Head Attention Layer


### Attention
./attn/*
- MHA: Multi-Head Attention Layer
