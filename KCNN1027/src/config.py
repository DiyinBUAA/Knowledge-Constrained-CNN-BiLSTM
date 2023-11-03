
import os
# DATASET
num_works = 2
sampling_interval = 10

val_size = 0.2
T = 1
capa_norm = 2.9
# MODEL:
model_name = "DTNN"
features = 3
filters = 256
kernel_size = 3
dropout = 0.2
hidden_units = 64


# TRAIN=
epoch = epochs = 250
optimizer = 'adam'
learning_rate = 1e-2
metrics = 'mae'
patience = 20

