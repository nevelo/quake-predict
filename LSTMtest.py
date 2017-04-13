"""Short and sweet LSTM implementation in Tensorflow.
Motivation:
When Tensorflow was released, adding RNNs was a bit of a hack - it required
building separate graphs for every number of timesteps and was a bit obscure
to use. Since then TF devs added things like `dynamic_rnn`, `scan` and `map_fn`.
Currently the APIs are decent, but all the tutorials that I am aware of are not
making the best use of the new APIs.
Advantages of this implementation:
- No need to specify number of timesteps ahead of time. Number of timesteps is
  infered from shape of input tensor. Can use the same graph for multiple
  different numbers of timesteps.
- No need to specify batch size ahead of time. Batch size is infered from shape
  of input tensor. Can use the same graph for multiple different batch sizes.
- Easy to swap out different recurrent gadgets (RNN, LSTM, GRU, your new
  creative idea)
"""


import numpy as np
import csv
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from my_lstm_cell import MyLSTMCell as myCell
import os
import urllib2
#from tensorflow.examples.mnist import input_data

map_fn = tf.map_fn

################################################################################
##                           DATASET GENERATION                               ##
##                                                                            ##
##  The problem we are trying to solve is adding two binary numbers. The      ##
##  numbers are reversed, so that the state of RNN can add the numbers        ##
##  perfectly provided it can learn to store carry in the state. Timestep t   ##
##  corresponds to bit len(number) - t.                                       ##
################################################################################

def as_bytes(num, final_size):
    res = []
    for _ in range(final_size):
        res.append(num % 2)
        num //= 2
    return res

def generate_example(num_bits):
    a = random.randint(0, 2**(num_bits - 1) - 1)
    return (as_bytes(a,  num_bits))

def generate_batch(num_bits, batch_size):
    """Generates instance of a problem.
    Returns
    -------
    x: np.array
        two numbers to be added represented by bits.
        shape: b, i, n
        where:
            b is bit index from the end
            i is example idx in batch
            n is one of [0,1] depending for first and
                second summand respectively
    y: np.array
        the result of the addition
        shape: b, i, n
        where:
            b is bit index from the end
            i is example idx in batch
            n is always 0
    """

    print(batch_size)

    x = np.empty((num_bits, batch_size, 2))
    y = np.empty((num_bits, batch_size, 1))

    for i in range(batch_size):
        a = generate_example(num_bits)
        x[:, i, 0] = a
        x[:, i, 1] = 0
        y[:, i, 0] = 1
    return x, y

def acquireData():
    if os.path.exists('m_combined.csv'):
        pass
    else:
        response = urllib2.urlopen('https://raw.githubusercontent.com/nevelo/quake-predict/master/m_combined.csv')
        data = response.read()
        filename="m_combined.csv"
        file_ = open(filename, "wb")
        file_.write(data)
        file_.close()

    with open('m_combined.csv', 'rb') as csvfile:
        data = csv.reader(csvfile)
        i = 0
        tempYear2 = 0
        for _ in data:
            i += 1
            t = np.arange(i)
        csvfile.close()
    with open('m_combined.csv', 'rb') as csvfile:
        pwr = np.zeros(i)
        highMag = np.zeros(i)
        meanPwr = np.zeros(i)
        i = -1
        data = csv.reader(csvfile)
        for row in data:
            if i >= 0:
                pwr[i] = row[5]
                meanPwr[i] = row[6]
                highMag[i] = row[7]
            i += 1

        return t, pwr, meanPwr, highMag

################################################################################
##                           GRAPH DEFINITION                                 ##
################################################################################

INPUT_SIZE    = 1       # 2 bits per timestep
RNN_HIDDEN    = 100
OUTPUT_SIZE   = 1       # 1 bit per timestep
TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 0.01

acquireData()

inputs  = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))  # (time, batch, in)
outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE)) # (time, batch, out)



cell = myCell(RNN_HIDDEN)

batch_size    = tf.shape(inputs)[1]
print(batch_size)
initial_state = cell.zero_state(batch_size, tf.float32)

rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)


final_projection = lambda x: layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)

WOutput = tf.Variable(0, name="output_Weighting", dtype="float32")

# apply projection to every timestep.
predicted_outputs = map_fn(final_projection, rnn_outputs*WOutput)

# compute elementwise cross entropy.
error = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
error = tf.reduce_mean(error)

# optimize
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)

# assuming that absolute difference between output and correct answer is 0.5
# or less we can round it to the correct output.
accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_outputs) < 0.5, tf.float32))


################################################################################
##                           TRAINING LOOP                                    ##
################################################################################

NUM_BITS = 10
ITERATIONS_PER_EPOCH = 100
BATCH_SIZE = 5

t, pwr, meanPWR, highMag = acquireData()
size = np.size(t)
trainPwr = pwr[:size-2000]
trainMPwr = meanPWR[:size-2000]
trainHMag = highMag[:size-2000]
print(np.size(trainHMag))
x = np.empty((size, BATCH_SIZE, 1))
y = np.empty((size, BATCH_SIZE, 1))
X, Y = generate_batch(num_bits=2, batch_size=5)
print(X[0])

for j in range(240):
    for i in range(BATCH_SIZE):
        x[j,i, 0] = trainPwr[i + j*BATCH_SIZE]

for j in range(240):
    for i in range(BATCH_SIZE):
        y[j,i, 0] = trainHMag[i + j*BATCH_SIZE]

session = tf.Session()

session.run(tf.global_variables_initializer())

for epoch in range(150):
    epoch_error = 0
    i = 0
    j = 0
    for _ in range(ITERATIONS_PER_EPOCH):
        # here train_fn is what triggers backprop. error and accuracy on their
        # own do not trigger the backprop.
        epoch_error += session.run([error, train_fn], {
            inputs: (x),
            outputs: y,
        })[0]
        j +=1
        i += BATCH_SIZE
        print(".", end='')
        if (_%50==0):
            print("")

    epoch_error /= ITERATIONS_PER_EPOCH
    valid_accuracy = session.run(accuracy, {
        inputs:  valid_x,
        outputs: valid_y,
    })
    print "Epoch %d, train error: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_error, valid_accuracy * 100.0)
