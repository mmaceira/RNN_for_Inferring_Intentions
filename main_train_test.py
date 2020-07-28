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
from src.AOlivares_dataset import AOlivaresActions

import src.AOlivares_functions as AOlivares_functions
from src.AOlivares_dataset import AOlivaresDataset


# Some utility functions
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60

    return '%dm %ds' % (m, s)


def time_since_in_seconds(since):
    s = time.time() - since

    # return also string:
    mins = math.floor(s / 60)
    secs = s - mins * 60

    return s, '%dm %ds' % (mins, secs)


def train():

    torch.set_grad_enabled(True)
    classifier.train()

    total_loss = 0

    for (input_samples, seq_name, Y) in train_loader:

        if use_gpu:
            input_samples = input_samples.cuda()
            Y = Y.cuda()

        output = classifier(input_samples)

        # remove the batch dimension of output: output[t,1,3]->output[t,3]
        output = output.view(output.size(0), -1)

        # lets skip the first samples to give some time to the rnn
        output_skipping = output[skip_initials:, :]
        y_skipping = Y.expand(output.size(0) - skip_initials)

        if use_gpu:
            y_skipping = y_skipping.cuda()

        loss = criterion(output_skipping, y_skipping)

        total_loss += loss.item()

        classifier.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)


# Testing cycle
def test(dataset_in, dataset_name_in, do_plot=False, out_folder_dataset=[], do_conf_matrix=False,
         normalize_conf_matrix=True):

    torch.set_grad_enabled(False)
    classifier.eval()

    total_loss = 0

    if check_output_fixed_positions:

        prediction_samples = [[] for i in range(len(windows_check_accuracy))]
        groundtruth_samples = [[] for i in range(len(windows_check_accuracy))]

    for (input_samples, name_sample, Y) in dataset_in:

        # name sample is a tuple with only one value, lets get the string:
        name_sample = name_sample[0]

        if use_gpu:
            input_samples = input_samples.cuda()
            Y = Y.cuda()

        output = classifier(input_samples)

        # remove the batch dimension of output: output[t,1,3]->output[t,3]
        output = output.view(output.size(0), -1)
        pred_vector = output.data.max(1, keepdim=True)[1]
        pred_probabilities = nn.functional.softmax(output, dim=1)

        loss = criterion(output, Y.expand(output.size(0)))
        if use_gpu:
            loss = loss.cuda()
        total_loss += loss.item()

        if check_output_fixed_positions:
            # lets keep track of prediction and groundtruth to compute accuracy, f1 and conf_matrix:
            # this should be done in check_output_fixed_positions: keep pred and gt at fixed pos
            for i in range(len(windows_check_accuracy)):
                prediction_samples[i].append(pred_vector[windows_check_accuracy[i]].item())
                groundtruth_samples[i].append(Y.item())

        # plt the output and the probability of each output for each sequence:
        if do_plot:
            plot_RNN.plot_sequence_pred_and_gt(pred_vector, Y.expand(pred_vector.size(0)),
                                                   out_folder_dataset,
                                                   name_sample, output_size, AolivaresActions)

            window_based = True
            plot_RNN.plot_sequence_probabilities_and_gt(pred_probabilities,
                                                                     out_folder_dataset,
                                                                     name_sample + "_window",
                                                                     AolivaresActions, window_based)

            window_based = False  # confidence based
            plot_RNN.plot_sequence_probabilities_and_gt(pred_probabilities,
                                                                     out_folder_dataset,
                                                                     name_sample + "_conf",
                                                                     AolivaresActions, window_based)

            plot_RNN.plot_sequence_probabilities_and_gt(pred_probabilities,
                                                                     out_folder_dataset,
                                                                     name_sample + "_crop350", AolivaresActions,
                                                                     number_of_samples_to_plot=350)

    # same as training: divide the total_loss for the number of sequence analyzed:
    total_loss = total_loss / len(dataset_in)
    print("total_loss " + dataset_name_in + " = {0:.2f}".format(total_loss))

    if do_conf_matrix:  # todo: how should I return the conf_matrix for the different sequences?

        values = range(3)  # 3 actions!!
        conf_mat_v = [[] for ll in range(len(windows_check_accuracy))]
        f1_measure_v = [[] for ll in range(len(windows_check_accuracy))]

        for i in range(len(windows_check_accuracy)):

            conf_matrix = skm.confusion_matrix(groundtruth_samples[i], prediction_samples[i], values)
            if normalize_conf_matrix:
                conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

            f1_meas = skm.f1_score(groundtruth_samples[i], prediction_samples[i], values, average="micro")

            # accuracy = skm.accuracy_score(groundtruth_samples[i], prediction_samples[i], values)

            # print("accuracy at windows_check_accuracy[i] = " + str(windows_check_accuracy[i]))
            # print(accuracy)

            # print("f1_measure at windows_check_accuracy[i] = " + str(windows_check_accuracy[i]))
            # print(f1_measure)

            # print("Normalized confusion matrix")
            # print(conf_mat)

            conf_mat_v[i] = conf_matrix
            f1_measure_v[i] = f1_meas

    # accuracy at the last value-> in evaluation mode we check more position of the sequence
    values = range(3)  # 3 actions!!
    accuracy_final_seq = skm.accuracy_score(groundtruth_samples[-1], prediction_samples[-1], values)

    if not do_conf_matrix:
        return total_loss, accuracy_final_seq, [], []
    else:
        return total_loss, accuracy_final_seq, conf_mat_v, f1_measure_v


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", help="select number of hidden units for the RNN")
    parser.add_argument("--dataloader_seed", help="select seed to divide train and test datasets")
    parser.add_argument("--use_gru", action='store_true', help="Use gru units in the RNN")
    parser.add_argument("--use_lstm", action='store_true', help="Use lstm units in the RNN")
    parser.add_argument("--num_samples_train", help="Samples to take from each sequence for training")
    parser.add_argument("--train_set_size", type=float, default=0.75, help="percentage of samples for training")
    parser.add_argument("--output_folder_eval", default="", help="output folder to store results")

    parser.add_argument("--execution_mode", help="execution_mode can be whether train or eval")
    parser.add_argument("--use_cpu", action='store_true', default=False, help="Use cpu instead of gpu")

    # deactivate if we want to get inference time:
    parser.add_argument("--do_plot", action='store_true', default=False, help="Use lstm units in the RNN")
    parser.add_argument("--compute_conf_matrix", action='store_true', default=False, help="Generate conf matrix")

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

    # by default we use gpu
    use_gpu = True
    if args.use_cpu:
        use_gpu = False

    hidden_size = int(args.hidden_size)

    train_size = args.train_set_size

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

    # the first temporal samples are not used to compute the loss since the
    skip_initials = 25
    # todo skip also the finals N samples?
    # skip_finals=25

    # positions where we will check the output of the model:
    # todo: make it work with 500 samples also!
    windows_check_accuracy = [50, 100, 250, 350, -1]  # , 500]

    lr_value = 0.001  # TODO: test

    # parameters for data length (aolivares)
    number_of_measurements = int(args.num_samples_train)  # 1000  # 350 size of the window

    # check the output at some fixed positions
    check_output_fixed_positions = True

    # lets generate the model:
    classifier = RNNClassifier(input_size, hidden_size, output_size, n_layers,
                               bidirectional=False, use_gru=use_gru, use_relu=use_relu)

    # lets create the name of the output mehtod:
    name_method = "aolivares_hidden_" + str(hidden_size) + "_" + str(n_layers) + "_" + str(lr_value)

    if use_gru:
        name_method = name_method + "_" + "gru"
    else:
        name_method = name_method + "_" + "lstm"

    if use_relu:
        name_method = name_method + "_" + "relu"

    if use_gpu:

        print("using GPU")

        execution_device = "gpu"

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
            classifier = torch.nn.DataParallel(classifier)

        if torch.cuda.is_available():
            classifier.cuda()
    else:
        print("using CPU")
        execution_device = "cpu"
        classifier.cpu()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr_value)

    # CrossEntropyLoss This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    # LogSoftmax Applies the log(Softmax(x))\log(\text{Softmax}(x))log(Softmax(x))
    # function to an n-dimensional input Tensor
    # NLLLoss The negative log likelihood loss. It is useful to train a classification problem with C classes.
    criterion = torch.nn.CrossEntropyLoss()

    # variables for processing the data
    type_of_dataset = 'natural'  # natural or mechanical  -> natural is the hard one
    labels = {0: 'grab', 1: 'move', 2: 'polish'}
    dataset_folder = 'force_based_human_intention_inference/data/'

    final_signal_size_percentage = 1  # 0.02  # percentage of the total signal which is kept after subsampling
    step = int(round(1 / final_signal_size_percentage))  # for subsampling

    processed_data = AOlivares_functions.read_dataset_(dataset_folder, type_of_dataset, labels)

    data_training, data_test = AOlivares_functions.pick_training_dataset_randomly_(processed_data,
                                                                                   train_size,
                                                                                   number_of_measurements, step,
                                                                                   labels,
                                                                                   random_seed_dataset=dataloader_random_seed,
                                                                                   normalize=True)

    test_dataset = AOlivaresDataset(data_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    train_dataset = AOlivaresDataset(data_training)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, )

    datasets_to_check = [train_loader, test_loader]
    datasets_names = ["train", "test"]

    # ######################################
    # creating output paths:
    # ######################################

    # in checkpoint_dir we will save the weights of the model:
    checkpoint_dir = "checkpoint" + "/train_size_" + str(int(train_size*100)) + \
                     "/random_seed_" + str(dataloader_random_seed) + "/" + name_method + "/"

    print("checkpoint folder: " + checkpoint_dir)

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # in base_outpath we will generate the plots for each individual sample:
    if args.do_plot:

        outpath_individual_samples = checkpoint_dir + "output_waveforms/"

        print("output individual_samples folder: " + outpath_individual_samples)
        if not os.path.isdir(outpath_individual_samples):
            os.makedirs(outpath_individual_samples)

        for dataset_name in datasets_names:
            if not os.path.isdir(outpath_individual_samples + dataset_name):
                os.mkdir(outpath_individual_samples + dataset_name)

        print(train_dataset.len)
        print(test_dataset.len)
    else:
        outpath_individual_samples = ""

    if args.execution_mode == "eval":

        if not os.path.isdir(args.output_folder_eval):
            os.makedirs(args.output_folder_eval)
    start = time.time()

    if execution_mode == "train":

        # ######################################
        # training:
        # ######################################

        min_loss = float("inf")
        best_epoch = float("inf")
        print("Training for %d epochs..." % N_EPOCHS)
        for epoch in range(0, N_EPOCHS):

            epoch_start = time.time()

            # Train cycle
            print("\n\ntraining model ...")
            loss_epoch = train()

            print('[{}] Train Epoch: {} \tLoss: {:.4f}'.format(time_since(start), epoch, loss_epoch))

            # Testing -> do plot of each sequence for the last one
            print("\n\nEvaluating trained model ...")

            for elem in range(len(datasets_to_check)):

                dataset_name = datasets_names[elem]
                dataset = datasets_to_check[elem]

                loss_epoch, accuracy_epoch, _, _ = test(dataset, dataset_name, do_conf_matrix=False, do_plot=False)

                plot_RNN.write_losses(checkpoint_dir + "/" + dataset_name + "_loss.txt",
                                          epoch, dataset_name, loss_epoch, accuracy_epoch)

                if dataset_name == "test":

                    if loss_epoch < min_loss:
                        min_loss = loss_epoch
                        best_epoch = epoch
                        torch.save(classifier.state_dict(), checkpoint_dir + "/best.pt")

                        f = open(checkpoint_dir + "/best_epoch.txt", "w")
                        f.write(str(epoch))
                        f.close()

            print("epoch time = " + time_since(epoch_start))

        print("total training time = " + time_since(start))

    # eval mode:
    elif execution_mode == "eval":

        # lets load the best model and d the inference doing the plot:
        print("doing inference with the final model")

        # epoch it is used in test to print the current epoch:
        f = open(checkpoint_dir + "/best_epoch.txt", "r")
        best_epoch = int(f.read())

        classifier.load_state_dict(torch.load(checkpoint_dir + "/best.pt"))

        for elem in range(len(datasets_to_check)):
            dataset_name = datasets_names[elem]
            dataset = datasets_to_check[elem]

            print("Dataset: " + dataset_name)
            epoch_start = time.time()

            # loss_final, accuracy_final, conf_matrix, accuracy_aligned_final, conf_matrix_aligned = \
            loss_final, acc_final, conf_mat_vector, f1_measure_vector = \
                test(dataset, dataset_name, do_plot=args.do_plot, do_conf_matrix=args.compute_conf_matrix,
                     out_folder_dataset=outpath_individual_samples + dataset_name + "/")

            time_test, time_string = time_since_in_seconds(epoch_start)

            print("Dataset: " + dataset_name + " time = " + time_string)

            with open(args.output_folder_eval + execution_device + "_" + dataset_name + "_" + "time_inference" + ".txt", "w") as fp:
                fp.write(str(time_test) + "\n")

            with open(args.output_folder_eval + execution_device + "_" + dataset_name + "_" + "num_seq" + ".txt", "w") as fp:
                fp.write(str(len(dataset)) + "\n")

            # plot results:
            actions_name = list(AolivaresActions.keys())

            if args.compute_conf_matrix:
                print("Writing confusion matrix to " + args.output_folder_eval)
                for ind, conf_mat in enumerate(conf_mat_vector):

                    eval_start = time.time()

                    window_size = windows_check_accuracy[ind]

                    if window_size != -1:
                        window_size_time = window_size / 500  # sensor give 500 samples each second

                        window_size_name = str(window_size_time)
                    else:
                        window_size_name = "final_sample"

                    # as we have equal samples of each class accuracy = f1_measure
                    # https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/
                    f1_measure = f1_measure_vector[ind]

                    # print("Storing results for window_size " + str(window_size))
                    plot_RNN.write_losses(
                        args.output_folder_eval + dataset_name + "_" + window_size_name + "_loss.txt",
                        best_epoch, dataset_name, loss_final, f1_measure)

                    output_confusion_matrix = args.output_folder_eval + \
                                              dataset_name + "_" + window_size_name + "_conf_matrix"

                    plot_RNN.write_matrix_to_disk(output_confusion_matrix + ".txt", conf_mat, best_epoch)

                    plot_RNN.plot_confusion_matrix(conf_mat, classes=actions_name,
                                                       title='Confusion Matrix',
                                                       file=output_confusion_matrix + ".png")

                    print("window_size = " + str(window_size) + " samples\t->\twindow_size_time = " + window_size_name +
                          "\t\t Acc: " + '{0:.2f}'.format(f1_measure))
