import os
import time
import csv
import numpy as np


def read_float_from_file(file_name, position_read=0):
    with open(file_name, 'r') as f:
        csv_f = csv.reader(f, delimiter=' ')
        first_line = next(csv_f)
        value = float(first_line[position_read])
    return value


def get_execution_name_and_file_to_check(experiment_to_run):

    # in train we check that the best.pt file exists:
    if experiment_to_run == "train":
        execution_folder = ""
        output_file_check = "best.pt"

    else:
        # all evaluation modes will have the folder name as their execution:
        execution_folder = experiment_to_run + "/"

        # first we check if we have done this experiments already:
        if experiment_to_run == "eval_conf_matrix":
            # lets see if the results have already been generated (power outage)
            output_file_check = "test_final_sample_conf_matrix.txt"

            # time measure execution:
        elif "eval_time_measure" in experiment_to_run:

            if "cpu" in experiment_to_run:
                output_file_check = "cpu" + "_test_time_inference.txt"
            elif "gpu" in experiment_to_run:
                output_file_check = "gpu" + "_test_time_inference.txt"
            else:
                print("unknown experiment: " + experiment_to_run)
                exit(0)

        elif experiment_to_run == "eval_conf_based_experiments":
            # lets see if the results have already been generated (power outage)
            output_file_check = "test_confidence_" + conf_th_decisions[-1] + "_f1.txt"

        else:
            print("unknown experiment: " + experiment_to_run)
            exit(0)

    return execution_folder, output_file_check


def get_experiment_execution_string(experiment_to_run, checkpoint_dir):

    experiment_bin_vector = []
    log_files_vector = []

    # run_experiments only has one execution for each configuration
    if not experiment_to_run == "eval_conf_based_experiments":

        to_run = "python main_train_test.py "
        log_file = "log/log_" + experiment_to_run + "_" + str(training_size) + "_seed_" + str(dataloader_seed) + \
                   "_hidden_" + str(hidden_size) + "_" + rnn_to_use[3:] + ".txt"

        # we only need to plot the experiments once:
        if experiment_to_run == "train":
            to_run = to_run
        elif experiment_to_run == "eval_conf_matrix":
            to_run = to_run + " --compute_conf_matrix " + " --do_plot "

        elif "eval_time_measure" in experiment_to_run:

            if "cpu" in experiment_to_run:
                to_run = to_run + " --use_cpu "

        # for all the eval executions we generate a different folder:
        if not "train" == experiment_to_run:
            to_run = to_run + " --output_folder_eval " + checkpoint_dir + experiment_to_run + "/"

        experiment_bin_vector.append(to_run)
        log_files_vector.append(log_file)

    # run_conf_based_experiments has multiple executions (one for each th)
    elif experiment_to_run == "eval_conf_based_experiments":
        experiment_bin = "python main_eval_confidence_based_decision.py " + \
                         "--output_folder " + checkpoint_dir + experiment_to_run + "/ "

        for th in conf_th_decisions:
            experiment_bin_vector.append(experiment_bin + "--min_confidence_score " + th)

            log_file = "log/log_" + \
                       "seed_" + str(dataloader_seed) + "_hidden_" + str(hidden_size) + "_" \
                       + rnn_to_use[3:] + "_" + execution_mode + "_conf_" + th + ".txt"

            log_files_vector.append(log_file)

    return experiment_bin_vector, log_files_vector


hidden_size_vector = [2, 5, 10, 20, 50, 100, 200, 500]
dataloader_seed_vector = [4, 10, 14, 22, 27, 42, 56, 70, 93, 97]
use_gru_or_lstm_vector = [True, False]
# execution_mode = "eval"  # "eval"  # "train"

# names for the samples that we are taking (comparison with alberto)
window_names = ["0.1", "0.2", "0.5", "0.7", "final_sample"]
# confidence thresholds taken in the conf based experiments
conf_th_decisions = ["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.95"]

hidden_size_vector = [200]  # , 5, 10, 20, 50, 100, 200, 500]
dataloader_seed_vector = [4, 10, 14, 22, 27, 42, 56, 70, 93, 97]
use_gru_or_lstm_vector = [True]  # , False]

training_size = 0.75  # 0.05

# normal experiments: selecting the most probable class at the end of the window as aolivares
# this execution calls main_train_test.py. It can be with execution_mode = "train" or "eval"

# experiments to run:                       # NAME
#  1 train                                  # train
#  2 eval -> confidence matrix              # eval_conf_matrix
#  3 eval -> time measures (gpu or cpu)     # eval_time_measure_gpu / eval_time_measure_cpu
#  4 eval -> conf based experiments         # eval_conf_based_experiments
#  5 recap -> prerequisite: 1, 2, 3, 4      # recap

experiment_to_run = "train"

print("\n\n\nExperiment to run = " + experiment_to_run + "\n\n\n")

# number of process running in parallel:
max_n_process = 2
if "eval_time_measure" in experiment_to_run:
    max_n_process = 3

if experiment_to_run != "recap":

    # create log folder if not exists:
    if not os.path.isdir("log/"):
        os.mkdir("log/")

    print("This script has to be called from the command line, (with pytorch the conda environment is not loaded!):")
    print("conda activate IRI-DL")
    print("python script_main_train_test.py")

    for hidden_size in hidden_size_vector:
        for dataloader_seed in dataloader_seed_vector:
            for use_gru_or_lstm in use_gru_or_lstm_vector:

                if use_gru_or_lstm:
                    rnn_to_use = " --use_gru"
                    name_rnn = "gru"
                else:
                    rnn_to_use = " --use_lstm"
                    name_rnn = "lstm"

                if "train" == experiment_to_run:
                    execution_mode = "train"
                else:
                    execution_mode = "eval"

                # base folder for this method (created in main_train_test):
                checkpoint_folder = "checkpoint/train_size_" + str(int(training_size * 100)) + "/" + \
                                    "random_seed_"+str(dataloader_seed) + "/" \
                                    "aolivares_hidden_" + str(hidden_size) + "_1_0.001_" + name_rnn + "/" \

                while True:
                    n_processes = os.popen('ps -U mmaceira | grep python | wc -l').read()

                    if int(n_processes) < max_n_process:
                        print("\nlets execute a new script ( int(n_processes) < max_n_process)")

                        experiment_folder_name, file_to_check = get_execution_name_and_file_to_check(experiment_to_run)

                        file_check_path = checkpoint_folder + experiment_folder_name + file_to_check
                        print(file_check_path)

                        # check if we already have executed this configuration:
                        if os.path.isfile(file_check_path):
                            print("skipping execution (already computed): seed " + str(dataloader_seed) +
                                  " hidden "+str(hidden_size) + " " + name_rnn)
                            execute = False
                        else:
                            execute = True

                        # if we need to do the experiments:
                        if execute:

                            # experiment_vector will have the specific parameters needed for the execution, the common
                            # ones are below (python_cmd):
                            experiment_vector, log_files = get_experiment_execution_string(experiment_to_run, checkpoint_folder)

                            for experiment_bin, l_file in zip(experiment_vector, log_files):

                                python_cmd = experiment_bin + " --dataloader_seed " + str(dataloader_seed) + \
                                             " --hidden_size " + str(hidden_size) + rnn_to_use + \
                                             " --train_set_size " + str(training_size) + \
                                             " --execution_mode " + execution_mode + " --num_samples_train 500" \
                                             " 2>&1 > " + l_file + " &"

                                print(python_cmd)
                                os.system(python_cmd)
                                time.sleep(1)

                                n_processes = 1000
                                while int(n_processes) > max_n_process:
                                    n_processes = os.popen('ps -U mmaceira | grep python | wc -l').read()
                                    print("lets wait for some process to finish ( int(n_processes) = max_n_process)")
                                    time.sleep(1)

                        break
                    else:
                        print("lets wait for some process to finish ( int(n_processes) = max_n_process)")
                        time.sleep(10)

# recap results
elif experiment_to_run == "recap":

    recap_folder = "checkpoint/" + "train_size_" + str(int(training_size * 100)) + "/recap/"

    if not os.path.isdir(recap_folder):
        os.mkdir(recap_folder)

    for use_gru_or_lstm in use_gru_or_lstm_vector:

        if use_gru_or_lstm:
            name_rnn = "gru"
        else:
            name_rnn = "lstm"

        for hidden_size in hidden_size_vector:

            window_results = []
            th_conf_f1 = []
            th_conf_position = []
            inference_times_cpu = []
            inference_times_gpu = []

            # windows based detections:
            for window in window_names:
                window_results.append([])

            # confidence based detections:
            for th in conf_th_decisions:
                th_conf_f1.append([])
                th_conf_position.append([])

            for dataloader_seed in dataloader_seed_vector:

                # folder to start all
                seed_folder = "checkpoint/" + "train_size_" + str(int(training_size*100)) + "/" +\
                              "random_seed_" + str(dataloader_seed) + "/" + \
                              "aolivares_hidden_" + str(hidden_size) + "_1_0.001_" + name_rnn + "/"

                method_folder_windows_based = seed_folder + "/" + "eval_conf_matrix" + "/"

                for ind, window in enumerate(window_names):

                    file_name = method_folder_windows_based + "test_" + window + "_loss.txt"

                    # print(file_name)
                    # check that the file only has one line:
                    num_lines_files = sum(1 for line in open(file_name))

                    if num_lines_files != 1:
                        print("File " + file_name + " has " + str(num_lines_files) + " lines, exiting")

                    accuracy_value = read_float_from_file(file_name, 4)
                    window_results[ind].append(accuracy_value)

                # windows based detections compute time:

                # gpu time:
                file_name = seed_folder + "/" + "eval_time_measure_gpu" + "/" + "gpu_test_time_inference.txt"
                time_gpu = read_float_from_file(file_name)
                inference_times_gpu.append(time_gpu)

                # cpu time:
                file_name = seed_folder + "/" + "eval_time_measure_cpu" + "/" + "cpu_test_time_inference.txt"
                print(file_name)
                time_cpu = read_float_from_file(file_name)
                inference_times_cpu.append(time_cpu)

                # confidence based detections:
                method_folder_confidence_based = seed_folder + "eval_conf_based_experiments/"

                for ind, th in enumerate(conf_th_decisions):

                    file_f1_name = method_folder_confidence_based + "test_" + "confidence_" + th + "_f1.txt"
                    file_pos_det_name = method_folder_confidence_based + "test_" + "confidence_" + th + "_mean_index.txt"

                    # print(file_name)
                    # check that the file only has one line:
                    num_lines_files = sum(1 for line in open(file_f1_name))

                    if num_lines_files != 1:
                        print("File " + file_name + " has " + str(num_lines_files) + ", exiting")

                    f1_value = read_float_from_file(file_f1_name)
                    th_conf_f1[ind].append(f1_value)

                    pos_value = read_float_from_file(file_pos_det_name)
                    th_conf_position[ind].append(pos_value)

            # out_file for the window based execution:
            out_file = recap_folder + name_rnn + "%03d" % (hidden_size) + ".txt"

            fp = open(out_file, "w")
            print("storing results at " + out_file)
            for ind, window in enumerate(window_results):

                if len(window) != len(dataloader_seed_vector):
                    print("Window size different than dataloader_seed_vector size:")
                    print(str(len(window)) + " " + str(len(dataloader_seed_vector)))
                    print("exiting")
                    exit(0)

                # print(window)
                # print(len(window))
                mean_f1_score = sum(window) / len(window)
                print(mean_f1_score)

                fp.write(str(mean_f1_score) + "\n")

            fp.close()

            # windows based detections compute time:
            out_file = recap_folder + name_rnn + "%03d" % (hidden_size) + "_time_cpu.txt"
            with open(out_file, 'w') as f:
                mean_inference_cpu = sum(inference_times_cpu) / len(inference_times_cpu)
                print(mean_inference_cpu)

                f.write(str(mean_inference_cpu) + "\n")

            out_file = recap_folder + name_rnn + "%03d" % (hidden_size) + "_time_gpu.txt"
            with open(out_file, 'w') as f:
                mean_inference_gpu = sum(inference_times_gpu) / len(inference_times_gpu)
                print(mean_inference_gpu)

                f.write(str(mean_inference_gpu) + "\n")

            print("\n")

            # and we have to collect also results for the confidence based execution:
            out_file_conf = recap_folder + name_rnn + "%03d" % (hidden_size) + "_conf_based.txt"

            fp = open(out_file_conf, "w")
            print("storing results at " + out_file_conf)
            for f1_v, pos_v in zip(th_conf_f1, th_conf_position):

                if len(conf_th_decisions) != len(th_conf_f1):
                    print("Confidence size conf_th_decisions than th_conf_f1 size:")
                    print(str(len(conf_th_decisions)) + " " + str(len(th_conf_f1)))
                    print("exiting")
                    exit(0)

                # print(window)
                # print(len(window))
                mean_f1_score = sum(f1_v) / len(f1_v)
                mean_position = sum(pos_v) / len(pos_v)
                print(mean_f1_score, mean_position)

                fp.write(str(mean_f1_score) + ", " + str(mean_position) + "\n")

            fp.close()

    # lets also get the confusion matrix for the confidence based method (Table 1 in the article):
    for use_gru_or_lstm in use_gru_or_lstm_vector:

        if use_gru_or_lstm:
            name_rnn = "gru"
        else:
            name_rnn = "lstm"

        for hidden_size in hidden_size_vector:

            # I need a matrix for each conf_th:
            for ind, th in enumerate(conf_th_decisions):

                conf_matrix_vector = []
                # confidence + confusion matrix :D
                out_file_conf_conf_matrix = recap_folder + name_rnn + "%03d" % (hidden_size) + \
                                            "_conf_based_conf_matrix_"+str(th)+".txt"

                # confidence based detections:
                # for th in conf_th_decisions:
                #     conf_matrix_vector.append([])

                for dataloader_seed in dataloader_seed_vector:

                    # folder to start all
                    seed_folder = "checkpoint/" + "train_size_" + str(int(training_size * 100)) + "/" + \
                                  "random_seed_" + str(dataloader_seed) + "/" + \
                                  "aolivares_hidden_" + str(hidden_size) + "_1_0.001_" + name_rnn + "/"

                    method_folder_confidence_based = seed_folder + "eval_conf_based_experiments/"

                    # confidence based detectionconf_matrix
                    method_folder_confidence_based = seed_folder + "eval_conf_based_experiments/"

                    file_conf_matrix = method_folder_confidence_based + "test_" + "confidence_" + th + "_conf_matrix.txt"
                    print(file_conf_matrix)

                    # check that the file only has six line:
                    num_lines_files = sum(1 for line in open(file_conf_matrix))
                    if num_lines_files != 6: # header + 3 lines confusion matrix + 2 empty
                        print("Confusion matrix file " + file_conf_matrix + " has " + str(num_lines_files) + ", exiting")

                    # read matrix from disk
                    data = []
                    matrix = []
                    with open(file_conf_matrix, 'r') as f:
                        data = f.readlines()  # read
                        # keep only the 1 -> 4 rows, where is the confusion matrix:
                        for i in range(1, 4):
                            matrix.append(data[i].strip().split(" "))
                    #print(matrix)
                    # append all the matrices in conf_matrix_vector:
                    conf_matrix_vector.append(matrix)
                #print(conf_matrix_vector)

                # lets average the values:
                matrix_by_index = [[[], [], []], [[], [], []], [[], [], []]]
                # first reorganize the data instead of [matrix][row][column] do [row][column][matrix]
                for matrix in conf_matrix_vector:
                    for ind_r, row in enumerate(matrix):
                        for ind_c, column in enumerate(row):
                            print(float(column))
                            matrix_by_index[ind_r][ind_c].append(float(column))
                print(matrix_by_index)

                # now lets average  [row][column][matrix] to [row][column]:
                avg_matrix = []
                for ind_r, row in enumerate(matrix):
                    for ind_c, column in enumerate(row):
                        avg_matrix.append(np.mean(matrix_by_index[ind_r][ind_c]))
                print(avg_matrix)

                # store result to disk:
                with open(out_file_conf_conf_matrix, 'w') as f:
                    for item in avg_matrix:
                        # write results over 100: 0.95 -> 95
                        # f.write("%d\n" % round(item*100))
                        f.write("%.2f\n" % (item * 100))

