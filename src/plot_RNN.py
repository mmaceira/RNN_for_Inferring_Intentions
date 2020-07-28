import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# plot one repetition:
def plot_one_sample(fx_data, fy_data, fz_data, tx_data, ty_data, tz_data, num_sample, output_folder, output_name="",
                    number_time_instants_to_plot=-1):

    # by default print all the sequence:
    if number_time_instants_to_plot == -1:
        number_time_instants_to_plot = len(fx_data[num_sample])

    t = np.arange(0, number_time_instants_to_plot/500, 1/500)  # 500 samples each second!

    matplotlib.use('Agg')
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    #fig.subplots_adjust(top=0.8)

    ax1.plot(t, fx_data[num_sample][0:number_time_instants_to_plot], 'y-', label='force x')
    ax1.plot(t, fy_data[num_sample][0:number_time_instants_to_plot], 'm-', label='force y')
    ax1.plot(t, fz_data[num_sample][0:number_time_instants_to_plot], 'k-', label='force z')
    ax1.legend(loc='lower right')
    #ax1.xa(loc='upper right')
    ax1.set_ylabel("Normalized forces")

    # lets center the plots: find max value of forces:
    list_forces = fx_data[num_sample]+fy_data[num_sample]+fz_data[num_sample]
    list_forces_abs = [abs(ele) for ele in list_forces]
    # find the max and multiplicate it for a factor (1.2) to set x_lim:
    max_v = max(list_forces_abs) * 1.2
    ax1.set_ylim(-max_v, max_v)

    ax2 = fig.add_subplot(212)
    #fig.subplots_adjust(top=0.8)

    ax2.plot(t, tx_data[num_sample][0:number_time_instants_to_plot], 'y-', label='torque x')
    ax2.plot(t, ty_data[num_sample][0:number_time_instants_to_plot], 'm-', label='torque y')
    ax2.plot(t, tz_data[num_sample][0:number_time_instants_to_plot], 'k-', label='torque z')
    ax2.set_ylabel("Normalized torques")
    ax2.set_xlabel("time(s)")

    # lets center the plots: find max value of forces:
    list_torques = tx_data[num_sample]+ty_data[num_sample]+tz_data[num_sample]
    list_forces_abs = [abs(ele) for ele in list_torques]
    max_v = max(list_forces_abs) * 1.2
    ax2.set_ylim(-max_v, max_v)

    ax2.legend(loc='lower right')
    # plt.show()
    if output_name == "":
        plt.savefig(output_folder+"frame_"+str(num_sample).zfill(4)+".png")
    else:
        plt.savefig(output_folder+output_name+".png")

    plt.close()


# plot all the samples in a vector:
def plot_all_samples(fx_data, fy_data, fz_data, tx_data, ty_data, tz_data, output_folder):

    plt.figure(1)

    for num_sample in range(len(fx_data)):

        t = np.arange(0., len(fx_data[num_sample]), 1)

        plt.subplot(321)
        plt.plot(t, fx_data[num_sample], 'r--', label='fx_data')
        if num_sample==0:
            plt.legend(loc='upper left')

        plt.subplot(323)
        plt.plot(t, fy_data[num_sample], 'g--', label='fy_data')
        if num_sample==0:
            plt.legend(loc='upper left')

        plt.subplot(325)
        plt.plot(t, fz_data[num_sample], 'b--', label='fz_data')

        if num_sample==0:
            plt.legend(loc='upper left')

        plt.subplot(322)
        plt.plot(t, tx_data[num_sample], 'r-', label='tx_data')
        if num_sample == 0:
            plt.legend(loc='upper left')

        plt.subplot(324)
        plt.plot(t, ty_data[num_sample], 'g-', label='ty_data')
        if num_sample == 0:
            plt.legend(loc='upper left')

        plt.subplot(326)
        plt.plot(t, tz_data[num_sample], 'b-', label='tz_data')

        if num_sample == 0:
            plt.legend(loc='upper left')

    plt.savefig(output_folder+"plot_all_samples"+".png")
    plt.close()
    # plt.show()


# plot all the samples in a vector:
# action has for each value on the vector if belongs to 0,1 or 2 action
def plot_all_samples_from_1_action(fx_data,fy_data,fz_data,tx_data,ty_data,tz_data,
                                    action_vector,action_to_plot,output_folder):

    plt.figure(1)

    for num_sample in range(len(fx_data)):

        if action_vector[num_sample]!=action_to_plot:
            continue

        t = np.arange(0., len(fx_data[num_sample]), 1)

        plt.subplot(321)
        plt.plot(t, fx_data[num_sample], 'r--', label='fx_data')
        if num_sample==0:
            plt.legend(loc='upper left')

        plt.subplot(323)
        plt.plot(t, fy_data[num_sample], 'g--', label='fy_data')
        if num_sample==0:
            plt.legend(loc='upper left')

        plt.subplot(325)
        plt.plot(t, fz_data[num_sample], 'b--', label='fz_data')

        if (num_sample==0):
            plt.legend(loc='upper left')

        plt.subplot(322)
        plt.plot(t, tx_data[num_sample], 'r-', label='tx_data')
        if num_sample == 0:
            plt.legend(loc='upper left')

        plt.subplot(324)
        plt.plot(t, ty_data[num_sample], 'g-', label='ty_data')
        if num_sample == 0:
            plt.legend(loc='upper left')

        plt.subplot(326)
        plt.plot(t, tz_data[num_sample], 'b-', label='tz_data')

        if num_sample == 0:
            plt.legend(loc='upper left')

#    plt.show()
    plt.savefig(output_folder+"plot_all_samples_from_action_"+str(action_to_plot)+".png")
    plt.close()


import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from collections import OrderedDict


# from SLD/pells/train_test_model -> to display results in visdom:
def write_losses(path, epoch, phase, epoch_loss, epoch_acc):

    fileW = open(path, "a")
    fileW.write('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc) + "\n")
    fileW.close()


def write_matrix_to_disk(path, matrix, epoch):

    fileW = open(path, "a")
    fileW.write("epoch " + str(epoch) + "\n")

    # if the matrix is of ints we will store as it, if not as decimal with 4 floating points:
    if matrix.dtype.name == "int64":
        np.savetxt(fileW, matrix, fmt='%i')
    elif matrix.dtype.name == "float64":
        np.savetxt(fileW, matrix, fmt='%1.4f')

    fileW.write("\n\n")
    fileW.close()


def plot_sequence_pred_and_gt(pred_vector, y, out_path, name_sequence, output_size, actions):

    # pred_vector may be a list
    if isinstance(pred_vector, list):
        pred_vector_numpy = array(pred_vector)
    # or a tensor
    else:
        pred_vector_numpy = pred_vector.cpu().numpy()

    y_numpy = y.squeeze().cpu().numpy()

    # plot ground truth:
    plt.plot(range(len(pred_vector_numpy)), y_numpy, 'g', linewidth=4, label="Ground truth")

    # plot prediction:
    plt.plot(range(len(pred_vector_numpy)), pred_vector_numpy, 'r', linewidth=1, label='Prediction')
    plt.axis([0, len(pred_vector_numpy), -1, output_size])

    action_names = list(actions.keys())
    action_names[action_names.index('open_gripper')] = 'open'
    plt.yticks(list(actions.values()), action_names)
    plt.legend(loc="upper left")

    # plt.show()
    plt.savefig(out_path+name_sequence + "_categories")
    plt.close()


def plot_sequence_probabilities_and_gt(pred_probabilities, out_path, name_sequence, actions):

    pred_probabilities_numpy = pred_probabilities.cpu().detach().numpy()
    number_classes = pred_probabilities_numpy.shape[1]

    # lets generate one color for each class:
    cmap = plt.get_cmap(name='hsv', lut=number_classes)
    linestyles = ['-', '--', ':']  #  '-.',
    for i in range(number_classes):
        plt.plot(range(len(pred_probabilities_numpy)), pred_probabilities[:, i].cpu().detach().numpy(),
                 c=cmap(i), linestyle=linestyles[i % len(linestyles)])

    plt.legend(actions)
    plt.axis([0, len(pred_probabilities_numpy), 0, 1])
    # plt.show()

    plt.savefig(out_path + name_sequence + "_probs")
    plt.close()


def plot_sequence_probabilities_and_gt_simbiots(pred_probabilities, out_path, name_sequence, actions, window_based=True,
                                                number_of_samples_to_plot=-1):
    fig, ax = plt.subplots(figsize=(10, 5))

    # lets generate one color for each class:
    cmap = ["red", "green", "blue"]  # plt.get_cmap(name='hsv', lut=number_classes)
    # linestyles = ['-', '--', ':']  # '-.',

    if number_of_samples_to_plot != -1:
        pred_probabilities = pred_probabilities[:number_of_samples_to_plot]

    # 500 samples is a second:
    x_max = len(pred_probabilities)/500

    x_samples = np.arange(0, x_max, 1 / 500)

    for i in range(len(cmap)):
        ax.plot(x_samples, pred_probabilities[:, i].cpu().detach().numpy(),
                c=cmap[i])  # , linestyle=linestyles[i % len(linestyles)])

    # change legend: instead of open_gripper put open:
    actions_mod = OrderedDict([('open', v) if k == 'open_gripper' else (k, v) for k, v in actions.items()])

    plt.legend(actions_mod, loc="upper right")
    plt.axis([0, x_max*1.02, 0, 1])
    plt.xlabel("time(s)")
    plt.ylabel("Confidence scores")

    # removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    #ax.set_yticks([0.2, 0.6, 0.8], minor=False)
    #ax.set_yticks([0.3, 0.55, 0.7], minor=True)
    #ax.yaxis.grid(True, which='major')
    #ax.yaxis.grid(True, which='minor')

    # option window based:
    if window_based:
        ax.set_xticks([0.1, 0.2, 0.5, 0.7, 1.0], minor=False)  # "0.1", "0.2", "0.5", "0.7", "final_sample"
        ax.set_xticks([0.1, 0.2, 0.5, 0.7, 1.0], minor=True)  # "0.1", "0.2", "0.5", "0.7", "final_sample"
        ax.xaxis.grid(True, which='minor')
        ax.xaxis.grid(True, which='major')
    else:  # confidence based:
        ax.set_yticks([0.9], minor=False)  # "0.1", "0.2", "0.5", "0.7", "final_sample"
        ax.set_yticks([0.9], minor=True)  # "0.1", "0.2", "0.5", "0.7", "final_sample"
        ax.yaxis.grid(True, which='minor')
        ax.yaxis.grid(True, which='major')

    # plt.title(name_sequence)
    # plt.show()

    plt.savefig(out_path + name_sequence + "_probs")
    plt.close()


# from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title=None,
                          file=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix -> already computed
    if 0:
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    # lets write the matrix to an output file
    plt.savefig(file)
    plt.close()
