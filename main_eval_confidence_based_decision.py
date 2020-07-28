import time
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse

import sklearn.metrics as skm

from src.model import RNNClassifier

import src.plot_RNN as plot_RNN
from src.simbiots_dataset import SimbiotsDataset
from src.simbiots_dataset import SimbiotsActions


# Some utility functions
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Testing cycle
def detect_action_confidence_score(dataset_in, dataset_name_in, min_confidence_score, skip_first_n_samples,
                                   do_conf_matrix=False,
                                   normalize_conf_matrix=True):

    torch.set_grad_enabled(False)
    classifier.eval()

    prediction_samples = []
    groundtruth_samples = []
    index_decision = []

    for (input_samples, name_sample, Y) in dataset_in:

        # name sample is a tuple with only one value, lets get the string:
        name_sample = name_sample[0]

        if use_gpu:
            input_samples = input_samples.cuda()
            Y = Y.cuda()

        output = classifier(input_samples)

        # remove the batch dimension of output: output[t,1,3]->output[t,3]
        output = output.view(output.size(0), -1)
        # pred_vector = output.data.max(1, keepdim=True)[1]
        pred_probabilities = nn.functional.softmax(output, dim=1)

        # lets get the probabilities for all samples in cpu:
        probs = pred_probabilities.cpu().numpy()

        probs_skipped_first = probs[skip_first_n_samples:, :]

        probs_index = np.argwhere(probs_skipped_first > min_confidence_score)

        if len(probs_index) > 0:
            first_index = probs_index[0, 0] + skip_first_n_samples
            class_det = probs_index[0, 1]

        # just keep the last element:
        else:
            first_index = probs_skipped_first.shape[0] + skip_first_n_samples
            class_det = np.argmax(probs[-1, :])

        prediction_samples.append(class_det)
        groundtruth_samples.append(Y.item())
        index_decision.append(first_index)

    if do_conf_matrix:

        values = range(3)  # 3 actions!!

        conf_matrix = skm.confusion_matrix(groundtruth_samples, prediction_samples, values)

        if normalize_conf_matrix:
            conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        f1_meas = skm.f1_score(groundtruth_samples, prediction_samples, values, average="micro")

    # accuracy at the last value-> in evaluation mode we check more position of the sequence
    # values = range(3)  # 3 actions!!
    # accuracy_final_seq = skm.accuracy_score(groundtruth_samples[-1], prediction_samples[-1], values)

    return conf_matrix, f1_meas, index_decision


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", help="select number of hidden units for the RNN")
    parser.add_argument("--dataloader_seed", help="select seed to divide train and test datasets")
    parser.add_argument("--use_gru", action='store_true', help="Use gru units in the RNN")
    parser.add_argument("--use_lstm", action='store_true', help="Use lstm units in the RNN")
    parser.add_argument("--num_samples_train", help="Samples to take from each sequence for training")
    parser.add_argument("--train_set_size", type=float, default=75, help="percentage of samples for training")
    parser.add_argument("--output_folder", default="checkpoint_simbiots", help="output folder to generate results")

    parser.add_argument("--execution_mode", help="execution_mode can be whether train or eval")
    parser.add_argument("--use_cpu", action='store_true', default=False, help="Use cpu instead of gpu")

    # deactivate if we want to get inference time:
    parser.add_argument("--compute_conf_matrix", action='store_true', default=True, help="Use lstm units in the RNN")

    # if the confidence is above min_confidence_score the action will be recognised
    # min_confidence_score = 0.5
    parser.add_argument("--min_confidence_score",
                        help="if the confidence is above min_confidence_score the action will be recognised")

    args = parser.parse_args()

    if args.use_gru == args.use_lstm:
        print("ERROR use_gru and use_lstm have the same value")
        exit()

    if args.use_gru:
        use_gru = True
    if args.use_lstm:
        use_gru = False

    if args.execution_mode == "train" or args.execution_mode == "eval":
        execution_mode = args.execution_mode
    else:
        print("ERROR execution_mode should be whether train or eval!!")
        exit()

    hidden_size = int(args.hidden_size)

    # tested with a relu layer after the gru / lstm -> same results in clothes dataset
    use_relu = False

    # seed to divide between train and test:
    dataloader_random_seed = int(args.dataloader_seed)

    input_size = 6  # fx,fy,fz,tx,ty,tz
    output_size = 3  # 3 actions: 0:'open_gripper', 1:'move', 2:'hold'
    # Parameters and DataLoaders
    # hidden_size = 200 -> as parameter

    # TODO: test with more layers?
    n_layers = 1

    # only works for batch_size = 1 since the dataloader needs a fix size for the last size (it can be done
    # with a collate function but I would prefer not to append values
    # https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/2
    BATCH_SIZE = 1
    N_EPOCHS = 200  # 00

    # the first temporal samples are not used to do the decision -> keep half of the first window that we were
    # using until now (first window is 0.1 seconds -> 50 samples
    skip_initials = 25

    # sensor give 500 samples each second
    sensor_samples_per_second = 500

    # positions where we will check the output of the model:
    # todo: make it work with 500 samples also!
    windows_check_accuracy = [50, 100, 250, 350, -1]  # , 500]

    lr_value = 0.001  # TODO: test

    # parameters for data length (aolivares)
    number_of_measurements = int(args.num_samples_train)  # 1000  # 350 size of the window

    min_confidence_score = float(args.min_confidence_score)

    print(min_confidence_score)

    # ######################################
    # creating output path and dataloaders:
    # ######################################

    # lets create the name of the output method:
    name_method = "aolivares_hidden_" + str(hidden_size) + "_" + str(n_layers) + "_" + str(lr_value)

    if use_gru:
        name_method = name_method + "_" + "gru"
    else:
        name_method = name_method + "_" + "lstm"

    if use_relu:
        name_method = name_method + "_" + "relu"

    checkpoint_dir = "checkpoint/" + "train_size_" + str(int(args.train_set_size*100)) + "/random_seed_" + str(dataloader_random_seed) + "/"

    outpath_recap = args.output_folder  # + "train_size_" + str(int(args.train_set_size*100)) + "/random_seed_" + str(dataloader_random_seed) + "/"
    if not os.path.isdir(outpath_recap):
        os.makedirs(outpath_recap)

    classifier = RNNClassifier(input_size, hidden_size, output_size, n_layers,
                               bidirectional=False, use_gru=use_gru, use_relu=use_relu)

    # by default we use gpu
    use_gpu = True
    if args.use_cpu:
        use_gpu = False

    if use_gpu:

        print("using GPU")

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
            classifier = torch.nn.DataParallel(classifier)

        if torch.cuda.is_available():
            classifier.cuda()
    else:
        print("using CPU")

        classifier.cpu()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr_value)

    # CrossEntropyLoss This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    # LogSoftmax Applies the log(Softmax(x))\log(\text{Softmax}(x))log(Softmax(x))
    # function to an n-dimensional input Tensor
    # NLLLoss The negative log likelihood loss. It is useful to train a classification problem with C classes.
    criterion = torch.nn.CrossEntropyLoss()


    import src.AOlivares_functions as AOlivares_functions
    import src.simbiots_dataset as simbiots_dataset

    # variables for processing the data
    type_of_dataset = 'natural'  # natural or mechanical  -> natural is the hard one
    labels = {0: 'grab', 1: 'move', 2: 'polish'}
    dataset_folder = 'force_based_human_intention_inference/data/'

    final_signal_size_percentage = 1  # 0.02  # percentage of the total signal which is kept after subsampling
    step = int(round(1 / final_signal_size_percentage))  # for subsampling

    processed_data = AOlivares_functions.read_dataset_(dataset_folder, type_of_dataset, labels)

    data_training, data_test = AOlivares_functions.pick_training_dataset_randomly_(processed_data,
                                                                                   args.train_set_size,
                                                                                   number_of_measurements, step,
                                                                                   labels,
                                                                                   random_seed_dataset=dataloader_random_seed,
                                                                                   normalize=True)

    test_dataset = simbiots_dataset.SimbiotsDatasetAOlivares(data_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    train_dataset = simbiots_dataset.SimbiotsDatasetAOlivares(data_training)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, )

    datasets_to_check = [train_loader, test_loader]
    datasets_names = ["train", "test"]

    print(train_dataset.len)
    print(test_dataset.len)

    start = time.time()

    # eval mode:
    # if execution_mode == "eval":

    # lets load the best model and d the inference doing the plot:
    print("doing inference with the final model")

    # epoch it is used in test to print the current epoch:
    f = open(checkpoint_dir + name_method + "/best_epoch.txt", "r")
    best_epoch = int(f.read())

    classifier.load_state_dict(torch.load(checkpoint_dir + name_method + "/best.pt"))

    for elem in range(len(datasets_to_check)):
        dataset_name = datasets_names[elem]
        dataset = datasets_to_check[elem]

        print("Dataset: " + dataset_name)
        epoch_start = time.time()

        # loss_final, accuracy_final, conf_matrix, accuracy_aligned_final, conf_matrix_aligned = \
        conf_mat, f1_measure, index_decisions = \
            detect_action_confidence_score(dataset, dataset_name, min_confidence_score, skip_initials,
                                           do_conf_matrix=args.compute_conf_matrix)

        print("Dataset: " + dataset_name + " time = " + time_since(epoch_start))

        mean_index_decision = np.mean(index_decisions)

        # plot results:
        actions_name = list(SimbiotsActions.keys())

        # as we have equal samples of each class accuracy = f1_measure
        # https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/

        print("Writing results to " + outpath_recap)

        window_size_name = "confidence_" + str(min_confidence_score)

        # write f1 obtained for the selected confidence:
        with open(outpath_recap + dataset_name + "_" + window_size_name + "_f1.txt", "w") as fp:
            fp.write(str(f1_measure) + "\n")

        # write number of samples needed to do the decision:
        with open(outpath_recap + dataset_name + "_" + window_size_name + "_mean_index.txt", "w") as fp:
            fp.write(str(mean_index_decision / sensor_samples_per_second) + "\n")

        output_confusion_matrix = outpath_recap + \
                                  dataset_name + "_" + window_size_name + "_conf_matrix"

        plot_RNN.write_matrix_to_disk(output_confusion_matrix + ".txt", conf_mat, best_epoch)

        plot_RNN.plot_confusion_matrix(conf_mat, classes=actions_name,
                                           title='Confusion Matrix',
                                           file=output_confusion_matrix + ".png")

        print(window_size_name + "\t\t Acc: " + '{0:.2f}'.format(f1_measure) +
              "\t\t Mean index decision: " + '{0:.2f}'.format(mean_index_decision))

