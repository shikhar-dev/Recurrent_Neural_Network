# Importing dependencies
import numpy as np
from copy import deepcopy

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


# Seeding random with constant for reproducibility
np.random.seed(7)


# Length of binary seq ie number of timesteps
seq_len = 6
maxx = 2 ** (seq_len - 1)  # We will generate number till this number


# Setting Layer sizes
input_dim = 2
hidden_dim = 16
output_dim = 1


# Initializing derivative matrices by zero
dw1 = np.zeros((hidden_dim,input_dim))
dw1rec = np.zeros((hidden_dim,hidden_dim))
dw2 = np.zeros((output_dim,hidden_dim))
db2 = np.zeros((output_dim,1))
db1 = np.zeros((hidden_dim,1))


# Randomizing weights and bias matrices
w2 = 2 * np.random.random((output_dim, hidden_dim)) - 1
w1 = 2 * np.random.random((hidden_dim, input_dim)) - 1
w1rec = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

b2 = 2 * np.random.random((output_dim, 1)) - 1
b1 = 2 * np.random.random((hidden_dim,1)) - 1


# Number of Training examples
m = 4000


# Generating Training Data set
X = np.zeros((seq_len, input_dim, m), dtype=np.int)  # 3D input matrix
Y = np.zeros((seq_len, output_dim, m), dtype=np.int)  # 3D output matrix
Yint = []

for i in range(m):
    a = np.random.randint(maxx - 1) + np.random.randint(10)  # Generating one test case
    b = np.random.randint(maxx - 1) + np.random.randint(4)
    c = a + b  # Generating output
    Yint.append(c)

    for j in range(seq_len):
        X[j][0][i] = (a & (1 << j)) / (1 << j)  # Filling X with bits of first number one eg at a time
        X[j][1][i] = (b & (1 << j)) / (1 << j)  # Filling X with bits of second number

        Y[j][0][i] = (c & (1 << j)) / (1 << j)  # Filling Y with bits of corresponding output


# Setting number of iterations
n_iter = 20000


# Setting Learning rate
alpha = 0.3


# Training Model
for iter in range(n_iter) :
    error = np.zeros((output_dim,m))
    A1 = []  # For storing output of first layer at all Timesteps
    A2 = []  # FOr storing output of output layer at all Timesteps
    Z1 = []
    A1.append(np.zeros((hidden_dim,m)))  # Dummy output for timestep before timestep '0'

    # FORWARD PROPAGATION
    for tm_step in range(seq_len) :
        a1_last = A1[-1]  # Getting last output of layer 1 in a1_last
        x = X[tm_step]

        z1 = np.dot(w1,x) + np.dot(w1rec,a1_last) + b1
        a1 = tanh(z1)  # Output of first layer

        z2 = np.dot(w2,a1) + b2
        a2 = sigmoid(z2)  # Output of second Layer

        A1.append(deepcopy(a1))  # Storing Output of hidden Layer/first Layer
        A2.append(deepcopy(a2))  # Storing Output of Output Layer/second Layer
        Z1.append(deepcopy(z1))  # Storing Summation of hidden Layer/first Layer

    # BACKWARD PROPAGATION
    dz1_future = np.zeros((hidden_dim,m))

    for tm_step in range(seq_len) :
        a1 = A1[-tm_step-1]             # Prerequisites required for back prop
        a2 = A2[-tm_step-1]             # We are iterating from last bit (MSB) thats why '-tm_step-1' (from end)
        a1_last = A1[-tm_step-2]
        y = Y[-tm_step-1]
        x = X[-tm_step-1]
        z1 = Z1[-tm_step-1]

        dz2 = a2 - y                # error for output layer
        dw2 += (1.0 / m) * np.dot(dz2,a1.T)
        db2 += (1.0 / m) * np.sum(dz2,axis=1,keepdims=True)

        dz1 = np.multiply(np.dot(w2.T,dz2) + np.dot(w1rec.T,dz1_future), tanh(z1,True))             #error for hidden layer
        dw1 += (1.0 / m) * np.dot(dz1, x.T)
        db1 += (1.0 / m) * np.sum(dz1, axis=1, keepdims=True)
        dw1rec += (1.0 / m) * np.dot(dz1,a1_last.T)

        error += np.abs(dz2)        # adding total error over all timesteps

        dz1_future = deepcopy(dz1)      # storing for the timestep 'BEFORE' this

    # Gradient Descent
    w2 = w2 - alpha * dw2
    w1 = w1 - alpha * dw1
    w1rec = w1rec - alpha * dw1rec
    b2 = b2 - alpha * db2
    b1 = b1 - alpha * db1


    dw1 *= 0        # Clearing accumulators for another run
    dw2 *= 0
    dw1rec *= 0
    db1 *= 0
    db2 *= 0


    # Progress
    if (iter+1)%2000 == 0 :
        print str((iter+1)/2000) + '0% completed, Error : ' + str(np.sum(error,axis=1)/m)
        alpha += 0.02       # Incrementing learning rate (OPTIONAL) the further we go while applying Gradient Descent the slower 'weights' values
                            # change, thus we increase alpha to keep the 'weights' value changing


for i in range(20) :        # Printing 20 Training Dataset to Verify
    na = 0
    for j in range(seq_len) :
        na = (na<<1) + rnd(A2[seq_len-j-1][0][i])
    print 'Expected Output : ' + str(Yint[i]), 'Output : ' + str(na)


# Saving Parameters
np.savez('save_weights',name1=w2,name2=w1,name3=w1rec,name4=b2,name5=b1)        # Saving weights to a file st Predictor can directly load them