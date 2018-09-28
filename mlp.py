#Multilayer perceptron (MLP) with variable number of hidden layers
#Variable number of linear threshold units (LTUs) in each layer
#N.B. zero hidden layers is equivalent to simple multioutput perceptron

import numpy as np
import random

class Mlp(object):

    def __init__(self,layers,nepoch,eta=0.01,softmaxfunc=True,
                 actfunc='relu',randseed=123,regularisation='L2',rcoeff=0.0,
                 nminibatches=3):
        self.eta = eta # learning rate
        self.layers = layers # list of number of neurons in each layer (input, hidden1, hidden2, ..., output)
            # discludes bias neuron for passthrough and hidden layers
        self.ninput = layers[0] # number of input attributes
        self.noutput = layers[-1] # number of output classes
        self.nhidden = len(layers) - 2 # number of hidden layers
        self.nepoch = nepoch # max. no. of epochs
        self.softmaxfunc = softmaxfunc # use a shared softmax function in the output layer Y/N
        self.actfunc = actfunc # (continuous) activation function for use in backpropagation algorithm
                               # options: 'tanh', 'logit' or 'relu' (default)
        self.randseed = randseed # random seed
        self.regularisation = regularisation # regularisation method
                                             # options: 'L2' (default) or 'L1'
        self.rcoeff = rcoeff # regularisation coeff l1 or l2 (as applicable)
                             # default no regularisation (rcoeff=0.0)
        self.nminibatches = nminibatches # no. of minibatches for batch update
        self.logfile = open("mlp_logfile.log","w")
        self.logfile.write("Training\nCycle\tMini-batch avg. cost\n")
        random.seed(self.randseed)
        self.initialise_weights()
        if actfunc == 'logit':
            self.logitfunc = True

    # delta for logit and softmax activation functions
    def delta(self, a, y):
        return (a-y)

    def softmax(self, z):
        e = np.exp(z)
        return (e/np.sum(e))

    def logit(self, z):
        return (1.0 / (1.0 + np.exp(-z)))

    def tanh(self, z):
        x = self.logit(2.0*z)
        return (2.0*x - 1.0)

    def relu(self, z):
        if z >= 0.0:
            return z
        else:
            return 0.0

    # Derivatives of cost functions
    def derivative(self, z):
        if (self.logitfunc):
            f = self.logit
        return f(z)*(1-f(z))

    # cross-entropy function is the cost function associated with sigmoid neurons
    def cross_entropy(self, a, y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1.0-y)*np.log(1-a)))

    # log-likelihood function is the cost function associated with softmax neurons
    def log_likelihood(self, a, y):
        return -np.dot(y, self.softmax(a).transpose())

    # Randomly initialised weights have a Gaussian distribution with mean 0
    # and std dev 1/sqrt(x) where x is the number of weights connecting
    # to the same neuron
    def initialise_weights(self):
        # the jth elem of the bias list is a list of m weights for the neurons
        # connecting the bias neuron of the jth layer to the m neurons of the
        # (j+1)th layer
        self.biases = [np.random.randn(y,1) for y in self.layers[1:]]
        # the jth elem of the weights list is a (m x n)-dimensional array
        # containing the connection weights of the m neurons (discluding the
        # bias) of the jth layer to the n neurons of the (j+1)th layer
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in \
                        zip(self.layers[:-1], self.layers[1:])]
        return

    # Gradient for cost function associated with softmax neurons is calculated
    # from softmax output and back-propagated to input layer
    def back_propagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x # passthrough neurons
        activations = [np.array([activation])]
        zs = [] # list of neuron inputs
        i = 0
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b.transpose()[0]
            zs.append(z)
            if i == self.nhidden and self.softmaxfunc: # softmax output layer
                activation = self.softmax(z)
            elif self.logitfunc:
                activation = self.logit(z)
            activations.append(np.array([activation]))
            i += 1
        print "activations", activations, "\nzs", zs
        if self.softmaxfunc: # error in softmax output layer
            cost = abs(self.log_likelihood(activations[-1], y)[0])
        dell = np.array(self.delta(activations[-1], y))
        nabla_b[-1] = dell.transpose()
        nabla_w[-1] = np.dot(dell.transpose(), activations[-2])
        for l in range(2, self.nhidden + 2):
            dell = np.dot(dell, self.weights[-l+1])*self.derivative(zs[-l])
            nabla_b[-l] = dell.transpose()
            nabla_w[-l] = np.dot(dell.transpose(), activations[-l-1])
        print "delta_nabla_b", nabla_b, "\ndelta_nabla_w", nabla_w
        return (nabla_b, nabla_w, cost)

    # Update the network's weights and biases by applying gradient descent using
    # backpropagation to a single mini batch
    def batch_update(self, mini_batch, n, cycle):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        cumcost = 0.0
        print "mini_batch", mini_batch
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w, cost = self.back_propagation(x, y)
            cumcost += cost
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.logfile.writelines((str(cycle),"\t",str(cumcost/float(n/self.nminibatches)),"\n"))
        self.biases = [b-(self.eta/len(mini_batch))*nb for b, nb in \
                       zip(self.biases, nabla_b)]
        print "biases", self.biases
        if self.regularisation == 'L2':
            self.weights = [(1.0-self.eta*(self.rcoeff/n))*w - \
                            (self.eta/len(mini_batch))*nw for w, nw in \
                            zip(self.weights, nabla_w)]
        elif self.regularisation == 'L1':
            self.weights = [w - (self.eta*self.rcoeff*np.sign(w)/n) - \
                            (self.eta/len(mini_batch))*nw for w, nw in \
                            zip(self.weights, nabla_w)]
        print "weights", self.weights
        return

    def train(self, traindata, targets):
        n = np.shape(traindata)[0]
        minibatchsize = int(n/self.nminibatches)
        print "weights", self.weights
        print "biases", self.biases
        for i in range(self.nepoch):
            traindata = zip(traindata, targets)
            np.random.shuffle(traindata)
            mini_batches = [traindata[k:k+minibatchsize] for k in range(0,n,
                            n/self.nminibatches)]
            for mini_batch in mini_batches:
                self.batch_update(mini_batch, n, i)
            traindata, targets = zip(*traindata)
        return

    # Calculate outputs for test data
    def forwardpass(self, dataset, targets):
        n = np.shape(dataset)[0]
        cumerr = 0.0
        for i in range(n):
            activation = dataset[i,:]
            j = 0
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation) + b.transpose()[0]
                if j == self.nhidden and self.softmaxfunc:
                    activation = self.softmax(z)
                elif self.logitfunc:
                    activation = self.logit(z)
                j += 1
            err = np.sum(abs(self.delta(activation, targets[i])))/float(len(activation))
            cumerr += err
            print "output", activation
        self.logfile.writelines(("% error in test set:\t",str(cumerr/float(n)*100.0)))
        return

    def readdata(self, fname):
        with open(fname) as fh:
            dataset = [line.split()[0:self.ninput] for line in fh]
        fh.close()
        if self.softmaxfunc:
            targets = []
            i = 0
            with open(fname) as fh:
                for line in fh.readlines():
                    targets.append(np.zeros(self.noutput))
                    targets[i][int(line.split()[-1])] = 1.0
                    i += 1
            fh.close()
        dataset = np.array(dataset,dtype=float)
        targets = np.array(targets,dtype=float)
        return dataset, targets

#Driver code
fname = 'mlp_dataset.dat' # training data
fname2 = 'mlp_testdata.dat' #test data
layers1 = [2, 4, 3]
mlp1 = Mlp(layers1,1000,softmaxfunc=True,actfunc='logit',regularisation='L1',rcoeff=0.01)
#Train neural network
traindata, targets = mlp1.readdata(fname)
for i in range(np.shape(traindata)[0]):
    print traindata[i,:], targets[i]
mlp1.train(traindata, targets)
#Test neural network
#testdata = [[3.0, 2.9 ],[2.0, -10.5], [1.0, 10.5]]
testdata, targets = mlp1.readdata(fname2)
mlp1.forwardpass(testdata, targets)
mlp1.logfile.close()
