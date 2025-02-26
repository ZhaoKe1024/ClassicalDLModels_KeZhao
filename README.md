

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

### Heterogeneous Neural Network
- ./gnn/hgnn_fjsp.py
    
    - Mode reference: *WenSong, Flexible_Job-Shop_Scheduling_via_Graph_Neural_Network_and_Deep_Reinforcement_Learning*
    - The code is simplified on the basis of Reference: [https://github.com/songwenas12/fjsp-drl](https://github.com/songwenas12/fjsp-drl)
    
    Basic Properties:
    - Node Types: Operation, Embedding    
    - Edge Types: batch_idxes, Matrix 1~6
    - batch_idxes: (batch_size, 1)

    Feature(Matrix1, Matrix2, Matrix3):
    - Matrix1: T_{ij}, the process time of the operation i executed on machine j.
    - Matrix2: feature latent of operation(batch_size, operation_latent_dim, operations)
    - Matrix3: feature latent of machine(batch_size, machine_latent_dim, machines)

    Node Type 1: Feature of Operation:
    - matrix4: machine - operation
    - matrix5: operation_0 -> operation_1
    - matrix6: operation_1 -> operation_2
    - batch_idxes
    - feature
    
    Node Type 2: Feature of Machines:
    - matrix4: machine - operation
    - batch_idxes
    - feature
- ...

### Attention
./attn/*
- MHA: Multi-Head Attention Layer
