# Requirements

Install `torchgan` and `torch` using the following commands:

```
pip install torchgan
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# Run TnT Evaluation

Please run the `TnT.ipynb` Jupyter Notebook to evaluate the TnT artifacts

# Pre-train flower GAN

To start tuning TnT, please download the pre-trained flower GAN model from the following [link](https://drive.google.com/file/d/1etVNk5GU2Ux4uclKzSwqujGvAAzY5_pC/view?usp=share_link)

# Train a GAN
We use `torchgan` to train our flower GAN, to use it on any of your naturalistic dataset, follow this tutorial from `torchgan` to train by replacing the dataset and model architecture to your own datasets/config from this [TorchGAN Training](https://github.com/torchgan/torchgan/blob/master/tutorials/Tutorial%201.%20Introduction%20to%20TorchGAN.ipynb)
