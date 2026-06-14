# DDS
Our code was developed on https://github.com/prescient-design/walk-jump/tree/main. Follow the instructions as mentioned on their repo for installations. 

# Dataset creation

Download the dataset as directed in the above repo. Run protein_kde_fast.py inside the directory src. It will extract the key features, estimates local density using KDE later converts into noise level. A new file is created which is used by the training and sampling models.

# Training
The entrypoint walkjump_train is the main driver for training and accepts parameters using Hydra syntax. The available parameters for configuration can be found by  looking in the src/walkjump/hydra_config directory

# Sampling
The entrypoint walkjump_sample is the main driver for sampling and accepts parameters using Hydra syntax. The available parameters for configuration can be found by looking in the src/walkjump/hydra_config directory

