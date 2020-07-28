import torch
import torch.nn as nn


class RNNClassifier(nn.Module):

    # Our model: can be whether gru (use_gru=True) or LSTM (use_gru=False)

    def __init__(self, input_size, hidden_size, output_size, n_layers=1,
                 bidirectional=False, use_gru=True, use_relu=False, dropout=0):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.use_gru = use_gru
        self.use_relu = use_relu

        if self.use_gru:
            self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout,
                                            bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout,
                                            bidirectional=bidirectional)

        if use_relu:
            self.relu = nn.ReLU()

        self.fc = nn.Linear(hidden_size, output_size)

        print("\n\n\nInitialized model with use_gru = {} and use_relu = {}\n\n\n".format(use_gru, use_relu))

    def forward(self, input):
        # Note: we run this all at once (over the whole input sequence)
        # input shape: B x S (input size)
        # transpose to make S(sequence) x B (batch)
        # input = input.t()

        batch_size = input.size(0)

        # Make a hidden
        # hidden = self._init_hidden(batch_size)

        # lets change the structure of the input data to match the dimensions expected by the net:
        input_permuted = input.permute(2, 0, 1)  # .cuda()

        if self.use_gru:
            output, hidden = self.gru(input_permuted)#, hidden)
        else:
            output, hidden = self.lstm(input_permuted)#, hidden)

        if self.use_relu:
            fc_output = self.fc(self.relu(output[:, -1]))
        else:
            fc_output = self.fc(output)

        # output is directly the fully connected, no need to softmax:
        # https://discuss.pytorch.org/t/vgg-output-layer-no-softmax/9273/7
        # this is done directly in the loss function : #CrossEntropyLoss This criterion combines
        # nn.LogSoftmax() and nn.NLLLoss() in one single class.
        # output_softmax = nn.functional.softmax(fc_output, dim=2)
        return fc_output

    def _init_hidden(self, batch_size):

        if self.use_gru:

            hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)

            # if torch.cuda.is_available():
            #     hidden = hidden.cuda()

        else:

            weight = next(self.parameters()).data
            hidden = (weight.new(self.n_layers*self.n_directions, batch_size, self.hidden_size).zero_(),
                      weight.new(self.n_layers*self.n_directions, batch_size, self.hidden_size).zero_())

        return hidden
