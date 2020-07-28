import csv
import src.plot_RNN as plot_RNN
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


# data from official dataset https://zenodo.org/record/3522205#.Xh7UmOEo85k
# the dataset will be named AOlivaresDataset, and some functions to read the dataset are in src/AOlivares_functions

###############################
# AOlivaresDataset
###############################

import src.AOlivares_functions as AOlivares_functions
import src.AOlivares_dataset as AOlivares_dataset

output_folder = "input_waveform_aolivares_data/"

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# variables for processing the data
type_of_dataset = 'natural'  # natural or mechanical
labels = {0: 'grab', 1: 'move', 2: 'polish'}
dataset_folder = 'force_based_human_intention_inference/data/'

training_portion = 0.75
random_seed_dataset = 4

# parameters for data length
number_of_measurements = 3500  # size of the window -> set longer that all the sequences to get seq completed
final_signal_size_percentage = 1  # 0.02  # percentage of the total signal which is kept after subsampling
step = int(round(1 / final_signal_size_percentage))  # for subsampling

number_time_instants_to_plot = 500

processed_data = AOlivares_functions.read_dataset_(dataset_folder, type_of_dataset, labels)

data_training, data_test = AOlivares_functions.pick_training_dataset_randomly_(
    processed_data, training_portion, number_of_measurements, step, labels, random_seed_dataset, normalize=True)

# lets plot all the data (here we don't care if it train or test:
data_vector = [data_training, data_test]

fx_data_vector = []
fy_data_vector = []
fz_data_vector = []

tx_data_vector = []
ty_data_vector = []
tz_data_vector = []

seq_name_vector = []
for data in data_vector:

    dataset_t = AOlivares_dataset.AOlivaresDataset(data)

    t_loader = DataLoader(dataset=dataset_t,
                              batch_size=1,
                              shuffle=True)

    for iii, (input_samples, seq_name, Y) in enumerate(t_loader):

        fx_data_vector.append(input_samples[0][0].tolist())
        fy_data_vector.append(input_samples[0][1].tolist())
        fz_data_vector.append(input_samples[0][2].tolist())

        tx_data_vector.append(input_samples[0][3].tolist())
        ty_data_vector.append(input_samples[0][4].tolist())
        tz_data_vector.append(input_samples[0][5].tolist())

        seq_name_vector.append(seq_name[0])

# and plot all the sequences:
for ind in range(len(fx_data_vector)):

    plot_RNN.plot_one_sample(fx_data_vector, fy_data_vector, fz_data_vector,
                                  tx_data_vector, ty_data_vector, tz_data_vector,
                                  ind, output_folder, seq_name_vector[ind], number_time_instants_to_plot)


