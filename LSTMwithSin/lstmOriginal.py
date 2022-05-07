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


class LSTM_cell_AI_SUMMER(torch.nn.Module):
    """
    A simple LSTM cell network for educational AI-summer purposes
    """
    def __init__(self, input_length=10, hidden_length=20):
        super(LSTM_cell_AI_SUMMER, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # forget gate components
        self.linear_forget_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_forget_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_forget = nn.Sigmoid()

        # input gate components
        self.linear_gate_w2 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_gate = nn.Sigmoid()

        # cell memory components
        self.linear_gate_w3 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.activation_gate = nn.Tanh()

        # out gate components
        self.linear_gate_w4 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r4 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_hidden_out = nn.Sigmoid()

        self.activation_final = nn.Tanh()

    def forget(self, x, h):
        x = self.linear_forget_w1(x)
        h = self.linear_forget_r1(h)
        return self.sigmoid_forget(x + h)

    def input_gate(self, x, h):
        # Equation 1. input gate
        x_temp = self.linear_gate_w2(x)
        h_temp = self.linear_gate_r2(h)
        return self.sigmoid_gate(x_temp + h_temp)

    def cell_memory_gate(self, i, f, x, h, c_prev):
        x = self.linear_gate_w3(x)
        h = self.linear_gate_r3(h)

        # new information part that will be injected in the new context
        k = self.activation_gate(x + h)
        g = k * i

        # forget old context/cell info
        c = f * c_prev
        # learn new context/cell info
        c_next = g + c
        return c_next

    def out_gate(self, x, h):
        x = self.linear_gate_w4(x)
        h = self.linear_gate_r4(h)
        return self.sigmoid_hidden_out(x + h)

    def forward(self, x, tuple_in ):
        (h, c_prev) = tuple_in
        # Equation 1. input gate
        i = self.input_gate(x, h)

        # Equation 2. forget gate
        f = self.forget(x, h)

        # Equation 3. updating the cell memory
        c_next = self.cell_memory_gate(i, f, x, h,c_prev)

        # Equation 4. calculate the main output gate
        o = self.out_gate(x, h)

        # Equation 5. produce next hidden output
        h_next = o * self.activation_final(c_next)

        return h_next, c_next


class GRU_cell_AI_SUMMER(torch.nn.Module):
    """
    A simple GRU cell network for educational purposes
    """

    def __init__(self, input_length=10, hidden_length=20):
        super(GRU_cell_AI_SUMMER, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # reset gate components
        self.linear_reset_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_reset_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)

        self.linear_reset_w2 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_reset_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.activation_1 = nn.Sigmoid()

        # update gate components
        self.linear_gate_w3 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r3 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.activation_2 = nn.Sigmoid()

        self.activation_3 = nn.Tanh()

    def reset_gate(self, x, h):
        x_1 = self.linear_reset_w1(x)
        h_1 = self.linear_reset_r1(h)
        # gate update
        reset = self.activation_1(x_1 + h_1)
        return reset

    def update_gate(self, x, h):
        x_2 = self.linear_reset_w2(x)
        h_2 = self.linear_reset_r2(h)
        z = self.activation_2(h_2 + x_2)
        return z

    def update_component(self, x, h, r):
        x_3 = self.linear_gate_w3(x)
        h_3 = r * self.linear_gate_r3(h)
        gate_update = self.activation_3(x_3 + h_3)
        return gate_update

    def forward(self, x, h):
        # Equation 1. reset gate vector
        r = self.reset_gate(x, h)

        # Equation 2: the update gate - the shared update gate vector z
        z = self.update_gate(x, h)

        # Equation 3: The almost output component
        n = self.update_component(x, h, r)

        # Equation 4: the new hidden state
        h_new = (1 - z) * n + z * h

        return h_new


class Sequence(nn.Module):
    def __init__(self, LSTM=True, custom=True):
        super(Sequence, self).__init__()
        self.LSTM = LSTM

        if LSTM:
            if custom:
                print("AI summer LSTM cell implementation...")
                self.rnn1 = LSTM_cell_AI_SUMMER(1, 51)
                self.rnn2 = LSTM_cell_AI_SUMMER(51, 51)
            else:
                print("Official PyTorch LSTM cell implementation...")
                self.rnn1 = nn.LSTMCell(1, 51)
                self.rnn2 = nn.LSTMCell(51, 51)
        # GRU
        else:
            if custom:
                print("AI summer GRU cell implementation...")
                self.rnn1 = GRU_cell_AI_SUMMER(1, 51)
                self.rnn2 = GRU_cell_AI_SUMMER(51, 51)
            else:
                print("Official PyTorch GRU cell implementation...")
                self.rnn1 = nn.GRUCell(1, 51)
                self.rnn2 = nn.GRUCell(51, 51)

        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):

            if self.LSTM:
                h_t, c_t = self.rnn1(input_t, (h_t, c_t))
                h_t2, c_t2 = self.rnn2(h_t, (h_t2, c_t2))
            else:
                h_t = self.rnn1(input_t, h_t)
                h_t2 = self.rnn2(h_t, h_t2)

            output = self.linear(h_t2)
            outputs += [output]

        # if we should predict the future
        for i in range(future):
            if self.LSTM:
                h_t, c_t = self.rnn1(input_t, (h_t, c_t))
                h_t2, c_t2 = self.rnn2(h_t, (h_t2, c_t2))
            else:
                h_t = self.rnn1(input_t, h_t)
                h_t2 = self.rnn2(h_t, h_t2)

            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    print(input.shape)
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])

    # build the model. LSTM=False means GRU cell
    seq = Sequence(LSTM=True, custom=True)

    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    # begin to train
    for i in range(3):
        print('STEP: ', i)


        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss


        optimizer.step(closure)
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
        plt.savefig('predict%d.png' % i)
        plt.close()


















