import numpy as np
import matplotlib.pyplot as plt
import sys

X = np.array([
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0]
])

y = np.array([
    [1],
    [1],
    [0],
    [0]
])

num_i_units = 2
num_h_units = 2
num_o_units = 1

learning_rate = 0.01
reg_param = 0
max_iter = 5000
m = 4

np.random.seed(1)
W1 = np.random.normal(0, 1, (num_h_units, num_i_units))
W2 = np.random.normal(0, 1, (num_o_units, num_h_units))
B1 = np.random.random((num_h_units, 1))
B2 = np.random.random((num_o_units, 1))


def sigmoid(z, derv=False):
    if derv: return z * (1 - z)
    return 1 / (1 + np.exp(-z))


def forward(x, predict=False):
    a1 = x.reshape(x.shape[0], 1)

    z2 = W1.dot(a1) + B1
    a2 = sigmoid(z2)

    z3 = W2.dot(a2) + B2
    a3 = sigmoid(z3)

    if predict: return a3
    return (a1, a2, a3)


dW1 = 0
dW2 = 0

dB1 = 0
dB2 = 0

cost = np.zeros((max_iter, 1))


def train(_W1, _W2, _B1, _B2):  # The arguments are to bypass UnboundLocalError error
    for i in range(max_iter):
        c = 0

        dW1 = 0
        dW2 = 0

        dB1 = 0
        dB2 = 0

        for j in range(m):
            sys.stdout.write("\rIteration: {} and {}".format(i + 1, j + 1))

            # Forward Prop.
            a0 = X[j].reshape(X[j].shape[0], 1)  # 2x1

            z1 = _W1.dot(a0) + _B1  # 2x2 * 2x1 + 2x1 = 2x1
            a1 = sigmoid(z1)  # 2x1

            z2 = _W2.dot(a1) + _B2  # 1x2 * 2x1 + 1x1 = 1x1
            a2 = sigmoid(z2)  # 1x1

            # Back prop.
            dz2 = a2 - y[j]  # 1x1
            dW2 += dz2 * a1.T  # 1x1 .* 1x2 = 1x2

            dz1 = np.multiply((_W2.T * dz2), sigmoid(a1, derv=True))  # (2x1 * 1x1) .* 2x1 = 2x1
            dW1 += dz1.dot(a0.T)  # 2x1 * 1x2 = 2x2

            dB1 += dz1  # 2x1
            dB2 += dz2  # 1x1

            c = c + (-(y[j] * np.log(a2)) - ((1 - y[j]) * np.log(1 - a2)))
            sys.stdout.flush()  # Updating the text.

        _W1 = _W1 - learning_rate * (dW1 / m) + ((reg_param / m) * _W1)
        _W2 = _W2 - learning_rate * (dW2 / m) + ((reg_param / m) * _W2)

        _B1 = _B1 - learning_rate * (dB1 / m)
        _B2 = _B2 - learning_rate * (dB2 / m)
        cost[i] = (c / m) + (
                (reg_param / (2 * m)) *
                (
                        np.sum(np.power(_W1, 2)) +
                        np.sum(np.power(_W2, 2))
                )
        )
    return (_W1, _W2, _B1, _B2)


W1, W2, B1, B2 = train(W1, W2, B1, B2)

print()
print(f"W1={W1}")
print(f"W2={W2}")
print(f"B1={B1}")
print(f"B2={B2}")

plt.plot(range(max_iter), cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()
