# We already have the Network trained and weights stored. All we have to do is get them from the file and use them to predict for an input.

import numpy as np

# Rounding function
def rnd (n) :
    if n >= 0.5 :
        return 1
    else :
        return 0


# Activation Fucntions
def sigmoid(x, deriv=False):
    if (deriv == True):
        return np.multiply(x, 1 - x)
    return 1 / (1 + np.exp(-x))

def tanh(x,deriv=False):
    if deriv == True :
        return 1-tanh(x)**2
    else :
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


# Defining Initial dimensions of the Model
seq_len = 6
input_dim = 2
hidden_dim = 16
output_dim = 1


# Loading Weights from the file 'save_weights.npz'
data = np.load('save_weights.npz')

w2 = data['name1']
w1 = data['name2']
w1rec = data['name3']
b2 = data['name4']
b1 = data['name5']


# Predicting
while True :
    ch = raw_input("Do you want to conyinue ? [y,n]: ")
    if ch == 'n' :
        break
    a, b = [int(x) for x in raw_input("Enter two numbers: ").split()]
    X = np.zeros((seq_len, input_dim, 1), dtype=np.int)  # 3D input matrix
    for j in range(seq_len):
        X[j][0][0] = (a & (1 << j)) / (1 << j)  # Filling X with bits of first number one eg at a time
        X[j][1][0] = (b & (1 << j)) / (1 << j)  # Filling X with bits of second number
    c = a + b
    ans = []

    # FORWARD PROPAGATION
    a1_last = np.zeros((hidden_dim,1))
    for tm_step in range(seq_len):
        x = X[tm_step]

        z1 = np.dot(w1, x) + np.dot(w1rec, a1_last) + b1
        a1 = tanh(z1)  # Output of first layer

        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)  # Output of second Layer

        ans.append(rnd(a2[0][0]))

    na = 0
    for j in range(seq_len):
        na = (na << 1) + ans[seq_len-j-1]

    print 'Expected Output : ' + str(c), 'Output : ' + str(na)