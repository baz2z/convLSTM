import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim


np.random.seed(2)

T = 20
L = 1000
N = 200

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')
torch.save(data, open('traindata.pt', 'wb'))



class LSTM_cell(torch.nn.Module):

    def __init__(self, input_length=10, hidden_length=20):
        super(LSTM_cell, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # Combined Gates

        self.W = nn.Linear(self.input_length, 4 * self.hidden_length, bias=True)
        self.U = nn.Linear(self.hidden_length, 4 * self.hidden_length)

        self.activation_final = nn.Tanh()

    def forward(self, x, tuple_in):
        (h, c_prev) = tuple_in
        # gates
        gates = self.W(x) + self.U(h)
        i_t, f_t, g_t, o_t = gates.chunk(chunks = 4, dim=-1)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * self.activation_final(c_t)

        return h_t, c_t




class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.rnn1 = LSTM_cell(1, 51)
        self.linear = nn.Linear(51, 1) # cause from self.input to 4 * self.hidden_sz


    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.rnn1(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]


        # if we should predict the future
        for i in range(future):
            h_t, c_t = self.rnn1(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1)
        outputs = outputs.squeeze(2)
        return outputs


if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])

    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()


    optimizer = optim.Adam(seq.parameters(), lr=0.005)
    # begin to train
    for i in range(10):
        print('STEP: ', i)

        output = seq(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()


        # draw the result
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)


        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)


        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('./images/predict%d.png' % i)
        plt.close()
