# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class

import numpy as np
from collections import OrderedDict

# SimbiotsActions = {0: 'open_gripper', 1: 'move', 2: 'hold'}
AOlivaresActions = OrderedDict()
AOlivaresActions['open_gripper'] = 0
AOlivaresActions['move'] = 1
AOlivaresActions['hold'] = 2


class AOlivaresDataset:

    # Initialize your data, download, etc.
    def __init__(self, data):

        self.data = data
        self.len = len(data["X"])

    def __getitem__(self, index):

        # first 6 elements of X are the sensor data
        sensor_data = np.asarray(self.data["X"][index], dtype=np.float32) # np.asarray(self.X[index][0:6], dtype=np.float32)
        # 7th element is the name of the sample:
        sequence_name = self.data["labels"][index]  #self.X[index][6]
        class_output = self.data["Y"][index]  # np.asarray(self.Y[index], dtype=np.long)

        return sensor_data, sequence_name, class_output

    def __len__(self):
        return self.len

