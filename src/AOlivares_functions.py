
import copy
import numpy as np
import random
import sklearn.preprocessing as preprocessing


# from force_based_human_intention_inference.raw_data_based_classification.scripts.utils_data_process:
def read_dataset_(dataset_folder, type_of_dataset, labels):
    """
    Description --
    · Reads the dataset and returns a dictionary containing all the axis of the dataset and the names
    and ids of the samples.

    Arguments --
        · dataset_folder - string with the path where the dataset is
        · type_of_dataset - string containing the sort of dataset (e.g. 'easy' or 'hard' to discriminate)
        · labels - dictionary containing the labels from the classification task
        (e.g. labels = {0:'open_gripper', 1:'move', 2:'hold'})

        · return the dictionary 'data_dict'
    """
    data_dict = dict()

    data_dict['fx_data'] = []  # list of lists containing the fx axis data for all the samples
    data_dict['fy_data'] = []  # list of lists containing the fy axis data for all the samples
    data_dict['fz_data'] = []  # list of lists containing the fz axis data for all the samples
    data_dict['tx_data'] = []  # list of lists containing the tx axis data for all the samples
    data_dict['ty_data'] = []  # list of lists containing the ty axis data for all the samples
    data_dict['tz_data'] = []  # list of lists containing the tz axis data for all the samples
    data_dict['y_data'] = []  # list of lists containing the label id (int) for all the samples

    data_dict['sample_ids'] = []  # number assigned to a sample denoting the order in which it was read
    data_dict['sample_names'] = []  # unique identifier based on the name of the rosbag where the sample comes from
    sample_cont = 0
    for id_number, name in labels.items():
        fx_file = open(dataset_folder + name + '_' + type_of_dataset + '_fx.txt', 'r')
        fy_file = open(dataset_folder + name + '_' + type_of_dataset + '_fy.txt', 'r')
        fz_file = open(dataset_folder + name + '_' + type_of_dataset + '_fz.txt', 'r')
        tx_file = open(dataset_folder + name + '_' + type_of_dataset + '_tx.txt', 'r')
        ty_file = open(dataset_folder + name + '_' + type_of_dataset + '_ty.txt', 'r')
        tz_file = open(dataset_folder + name + '_' + type_of_dataset + '_tz.txt', 'r')

        names_file = open(dataset_folder + name + '_' + type_of_dataset + '_sample_id.txt', 'r')

        # Loop through datasets
        for x in fx_file:
            data_dict['fx_data'].append([float(ts) for ts in x.split()])
            data_dict['y_data'].append(1 * id_number)
            data_dict['sample_ids'].append(sample_cont)
            sample_cont += 1
        for x in fy_file:
            data_dict['fy_data'].append([float(ts) for ts in x.split()])
        for x in fz_file:
            data_dict['fz_data'].append([float(ts) for ts in x.split()])
        for x in tx_file:
            data_dict['tx_data'].append([float(ts) for ts in x.split()])
        for x in ty_file:
            data_dict['ty_data'].append([float(ts) for ts in x.split()])
        for x in tz_file:
            data_dict['tz_data'].append([float(ts) for ts in x.split()])

        for x in names_file:
            data_dict['sample_names'].append(x.split())

    return data_dict


# def prepare_data_for_GPy_(x, y)  (modified here):
def prepare_data_rnn(x, y):

    """
    Description --
    ·This method should be used in order to prepare the data for a rnn

    Input Arguments --
        · x is an array of arrays containing data from EACH axis of the sensor (e.g.
                      x = [[samples of axis_1], [samples of axis_2], [samples of axis_3],])
        · y is an array with the label (numeric value) for each sample

    Output Arguments --
        · data is a dictionary which contains two keys (X and Y). The value for data['X'] is a list of lists in which
        we can find each sample. The value for data['Y'] is a list of labels, each element of those sub-lists will be
        a 'int' which represents the label of a sample

    """

    data = dict()

    data['X'] = list()
    n_samp = np.shape(x)[1]
    for i in range(0, n_samp):

        data["X"].append(x[:, i])

        # aolivares:
        # data['X'].append(
        #    np.concatenate((x[:, i]), axis=0))  # each feature vector contains the features of the tree axes

    # for the rnn we just need the y information (no need to create a sublist as prepare_data_for_GPy_:
    data['Y'] = list()
    data['Y'] = y

    data['X'] = np.array(data['X'])

    return data


# from force_based_human_intention_inference.raw_data_based_classification.scripts.utils_evaluation (modified here):
def pick_training_dataset_randomly_(data_dict, training_portion, number_of_measurements,
                                    step, labels, random_seed_dataset, normalize=False):
    """
    Description --
    · It returns the dataset divided into training and test. Note that not only the argument sample_ids is used here,
    several variables (e.g. training_portion) are used but the function picks them from the jupyter saved data

    Arguments --
        · data_dict - is a dictionary which contains several lists with all the axis of the sensor data and all
        relevant information for the samples (ids, names, labels, etc.)
        · training_portion - float between zero and 1 to indicate the portion of data used for training
        · number_of_measurements - integer denoting the number of timestamps to use as window's size
        · step - integer indicating the subsampling step if any

        · return two dictionaries containing all data for training and test (see below the keys of the dictionary)
            - 'X' contains a list of lists which represent the samples in the format needed for GPy library
            - 'Y' contains a list of lists which represent the labels of the samples in the format needed for GPy
            - '*_labels' contains a list of the labels ids for the training/test samples
            - 'sample_ids_*' list of the numerical ids assigned to the training/test samples

    """

    sample_ids_sorted = copy.deepcopy(data_dict['sample_ids'])
    random.Random(random_seed_dataset).shuffle(sample_ids_sorted)

    sample_ids_training = sample_ids_sorted[0:int(len(data_dict['sample_ids']) * training_portion)]
    sample_ids_test = sample_ids_sorted[int(len(data_dict['sample_ids']) * training_portion):len(sample_ids_sorted)]

    fx_training = list()
    fy_training = list()
    fz_training = list()
    tx_training = list()
    ty_training = list()
    tz_training = list()
    y_training = list()
    # mmaceira adding sample names to keep track of each sample:
    sample_names_training = list()

    fx_test = list()
    fy_test = list()
    fz_test = list()
    tx_test = list()
    ty_test = list()
    tz_test = list()
    y_test = list()
    # mmaceira adding sample names to keep track of each sample:
    sample_names_test = list()

    for i in range(0, len(sample_ids_training)):
        example_id = sample_ids_training[i]

        # forces
        window = np.array(data_dict['fx_data'][example_id][0:number_of_measurements:step])
        fx_training.extend([window])  # [] used due to the need of having 2d array for normalization method (below)

        window = np.array(data_dict['fy_data'][example_id][0:number_of_measurements:step])
        fy_training.extend([window])

        window = np.array(data_dict['fz_data'][example_id][0:number_of_measurements:step])
        fz_training.extend([window])

        # torques
        window = np.array(data_dict['tx_data'][example_id][0:number_of_measurements:step])
        tx_training.extend([window])

        window = np.array(data_dict['ty_data'][example_id][0:number_of_measurements:step])
        ty_training.extend([window])

        window = np.array(data_dict['tz_data'][example_id][0:number_of_measurements:step])
        tz_training.extend([window])

        y_ = [data_dict['y_data'][example_id]]
        y_training.extend(y_)

        # sample identifier to do plots, add label of the y sample (labels = {0: 'grab', 1: 'move', 2: 'polish'})
        # labels are created with the format: 'polish_2019-02-05-16-25-47'
        sample_names_training.append(labels[y_[0]] + "_" + data_dict['sample_names'][example_id][0])

    for i in range(0, len(sample_ids_test)):
        example_id = sample_ids_test[i]

        # forces
        window = np.array(data_dict['fx_data'][example_id][0:number_of_measurements:step])
        fx_test.extend([window])  # [] used due to the need of having 2d array for normalization method (below)

        window = np.array(data_dict['fy_data'][example_id][0:number_of_measurements:step])
        fy_test.extend([window])

        window = np.array(data_dict['fz_data'][example_id][0:number_of_measurements:step])
        fz_test.extend([window])

        # torques
        window = np.array(data_dict['tx_data'][example_id][0:number_of_measurements:step])
        tx_test.extend([window])

        window = np.array(data_dict['ty_data'][example_id][0:number_of_measurements:step])
        ty_test.extend([window])

        window = np.array(data_dict['tz_data'][example_id][0:number_of_measurements:step])
        tz_test.extend([window])

        y_ = [data_dict['y_data'][example_id]]
        y_test.extend(y_)

        # sample identifier to do plots, add label of the y sample (labels = {0: 'grab', 1: 'move', 2: 'polish'})
        # labels are created with the format: 'polish_2019-02-05-16-25-47'
        sample_names_test.append(labels[y_[0]] + "_" + data_dict['sample_names'][example_id][0])

    fx_training = np.array(fx_training)
    fy_training = np.array(fy_training)
    fz_training = np.array(fz_training)
    tx_training = np.array(tx_training)
    ty_training = np.array(ty_training)
    tz_training = np.array(tz_training)
    y_training = np.array(y_training)

    fx_test = np.array(fx_test)
    fy_test = np.array(fy_test)
    fz_test = np.array(fz_test)
    tx_test = np.array(tx_test)
    ty_test = np.array(ty_test)
    tz_test = np.array(tz_test)
    y_test = np.array(y_test)

    if normalize:

        fx_training_normalized = preprocessing.normalize(fx_training)
        fy_training_normalized = preprocessing.normalize(fy_training)
        fz_training_normalized = preprocessing.normalize(fz_training)
        tx_training_normalized = preprocessing.normalize(tx_training)
        ty_training_normalized = preprocessing.normalize(ty_training)
        tz_training_normalized = preprocessing.normalize(tz_training)

        fx_test_normalized = preprocessing.normalize(fx_test)
        fy_test_normalized = preprocessing.normalize(fy_test)
        fz_test_normalized = preprocessing.normalize(fz_test)
        tx_test_normalized = preprocessing.normalize(tx_test)
        ty_test_normalized = preprocessing.normalize(ty_test)
        tz_test_normalized = preprocessing.normalize(tz_test)
        """

        fx_training_normalized = np.subtract(fx_training, np.mean(fx_training))
        fy_training_normalized = np.subtract(fy_training, np.mean(fy_training))
        fz_training_normalized = np.subtract(fz_training, np.mean(fz_training))
        tx_training_normalized = np.subtract(tx_training, np.mean(tx_training))
        ty_training_normalized = np.subtract(ty_training, np.mean(ty_training))
        tz_training_normalized = np.subtract(tz_training, np.mean(tz_training))

        fx_test_normalized = np.subtract(fx_test, np.mean(fx_test))
        fy_test_normalized = np.subtract(fy_test, np.mean(fy_test))
        fz_test_normalized = np.subtract(fz_test, np.mean(fz_test))
        tx_test_normalized = np.subtract(tx_test, np.mean(tx_test))
        ty_test_normalized = np.subtract(ty_test, np.mean(ty_test))
        tz_test_normalized = np.subtract(tz_test, np.mean(tz_test))
        """
        array_of_signals_for_training = np.concatenate(([fx_training_normalized], [fy_training_normalized],
                                                        [fz_training_normalized], [tx_training_normalized],
                                                        [ty_training_normalized], [tz_training_normalized]), axis=0)

        array_of_signals_for_test = np.concatenate(([fx_test_normalized], [fy_test_normalized], [fz_test_normalized],
                                                    [tx_test_normalized], [ty_test_normalized], [tz_test_normalized]),
                                                   axis=0)
    else:
        array_of_signals_for_training = np.concatenate(([fx_training], [fy_training],
                                                        [fz_training], [tx_training],
                                                        [ty_training], [tz_training]), axis=0)

        array_of_signals_for_test = np.concatenate(([fx_test], [fy_test], [fz_test],
                                                    [tx_test], [ty_test], [tz_test]), axis=0)

    # aolivares was converting class 'STANDING' (5, in numeric value), the sub-list would be: [-1, -1, -1, -1, 1, -1]
    # here we don't need that, just build the dictionary as
    # https://github.com/albertoOA/force-based-human-intention-inference/blob/2b22dc03ba5bc261ce716f2132ddf68c54453acf/feature_based_classification/scripts/utils_data_process.py#L99
    data_training = prepare_data_rnn(array_of_signals_for_training, y_training)
    data_test = prepare_data_rnn(array_of_signals_for_test, y_test)

    data_training["labels"] = sample_names_training
    data_test["labels"] = sample_names_test

    return data_training, data_test
