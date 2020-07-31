# Recurrent Neural Networks for Inferring Intentions in Shared Tasks for Industrial Collaborative Robots
Repository with the implementation of RO-MAN article 

**Recurrent Neural Networks for Inferring Intentions in Shared Tasks for Industrial Collaborative Robots**

### Abstract 

Industrial robots are evolving to work closely with humans in shared spaces. Hence, robotic tasks are increasingly 
shared between humans and robots in collaborative settings. To enable a fluent human robot collaboration, 
robots need to predict and respond in real-time to worker's intentions. %In this paradigm, workers behave naturally 
carrying their tasks while robots adapt to their intentions. We present a method for early decision using force 
information.
  
Forces are provided naturally by the user through the manipulation of a shared object in a collaborative task. The 
proposed algorithm uses a recurrent neural network to recognize operator's intentions. The algorithm is evaluated in 
terms of action recognition on a force dataset. It excels at detecting intentions when partial data is provided, 
enabling early detection and facilitating a quick robot reaction.

## Installation

It is recommended to run the code with a conda environment:

conda create -n IRI-DL  
conda activate IRI-DL  
conda install pytorch=1.0 torchvision cudatoolkit=10.0 -c pytorch  
conda install matplotlib opencv pillow scikit-learn scikit-image cython tqdm  

conda activate IRI-DL

For extra installation options such as setting the Pycharm IDE: 
https://github.com/albertpumarola/IRI-DL

## Setting up the dataset

The dataset used in this work is Force-based human intention inference.

It can be downloaded from:
https://zenodo.org/record/3522205#.XqG1F_mxU5k  

After downloading and unzipping it, it has to be stored in a force_based_human_intention_inference inside this repo as follows:

cp -r ~/Downloads/force-based-human-intention-inference-v1.0/albertoOA-force-based-human-intention-inference-d4e48c5/ RNN_for_Inferring_Intentions/force_based_human_intention_inference


## Experiments: 


### Check dataset

You can check that the dataset is stored correctly with: <br>

python main_data_load.py <br>

It creates a folder input_waveform_aolivares_data/ with all the waveforms of the dataset <br>


### Train the model

To run the experiments in the paper, the main_train_test.py program is provided.

The following parameters are accessible via command line:

main_train_test.py 
optional arguments:
-h, --help            show this help message and exit
-  --hidden_size HIDDEN_SIZE
                        select number of hidden units for the RNN
-  --dataloader_seed DATALOADER_SEED
                        select seed to divide train and test datasets
-  --use_gru             Use gru units in the RNN
-  --use_lstm            Use lstm units in the RNN
-  --num_samples_train NUM_SAMPLES_TRAIN
                        Samples to take from each sequence for training
-  --train_set_size TRAIN_SET_SIZE
                        percentage of samples for training
-  --output_folder_eval OUTPUT_FOLDER_EVAL
                        output folder to store results
-  --execution_mode EXECUTION_MODE
                        execution_mode can be whether train or eval
-  --use_cpu             Use cpu instead of gpu
-  --do_plot             Use lstm units in the RNN
-  --compute_conf_matrix
                        Generate conf matrix
  
### Test the model

Once the network has been trained, it can be used to detect with the Confidence-Based Evaluation defined in the paper: <br>
python main_eval_confidence_based_decision.py

<br>


### Reproduce RO-MAN article results

To reproduce all the results from the RO-MAN article, the script script_main_train_test.py calls main_train_test.py 
with all the configuration options used in the paper:  
python script_main_train_test.py

To select with experiment has to be run, just modify the variable experiment_to_run = "train" in script_main_train_test.py:


| Operation        | Description           | Option (variable experiment_to_run)  |
| ------------- |:--------------------:| ------------:|
| train     | train the network with the parameters defined in script_main_train_test | train |
| eval confidence matrix     | Evaluate the performance of the network in terms of confusion matrix  (Window based evaluation)  |   eval_conf_matrix |
| eval time measures     | Evaluate the performance of the network in terms of time measures ( gpu or cpu)    |   eval_time_measure_gpu / eval_time_measure_cpu |
| eval confidence based experiments          |        Evaluate the performance of the network with the Confidence based evaluation definedin the article |  eval_conf_based_experiments     |
| recap | It summarizes all the experiments into a folder (all the options before are prerequisites of this option    |    recap |
