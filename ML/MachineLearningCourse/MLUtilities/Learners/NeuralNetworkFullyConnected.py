import time
import numpy as np
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryProbabilityEstimate as EvaluateBinaryProbabilityEstimate

class NeuralNetworkFullyConnected(object):
    """Framework for fully connected neural network"""
    def __init__(self, numInputFeatures, hiddenLayersNodeCounts=[2], seed=1000):
        np.random.seed(seed)

        self.totalEpochs = 0
        self.lastLoss    = None
        self.converged   = False

        # set up the input layer
        self.layerWidths = [ numInputFeatures ]

        # set up the hidden layers
        for i in range(len(hiddenLayersNodeCounts)):
            self.layerWidths.append(hiddenLayersNodeCounts[i])

        # output layer
        self.layerWidths.append(1)

        ###
        ## now set up all the parameters and any arrays you want for forward/backpropagation
        ###
        numWeightLayers = len(self.layerWidths)-1
        self.layers = []
        self.hiddens = []
        self.errors = []
        self.weight0 = []
        self.prev_w_delta = []
        self.prev_w0_delta = []
        for i in range(numWeightLayers):
            layerDimCol = self.layerWidths[i]
            layerDimRow = self.layerWidths[i+1]

            std = 1.0 / np.sqrt(layerDimCol)
            self.layers.append(np.random.uniform(-std, std, (layerDimRow, layerDimCol)))
            self.weight0.append(np.random.uniform(-std, std, (layerDimRow,)))
            self.prev_w_delta.append(np.zeros((layerDimRow, layerDimCol)))
            self.prev_w0_delta.append(np.zeros((layerDimRow,)))

            self.hiddens.append(np.zeros((layerDimRow,), dtype=float))
            self.errors.append(np.zeros((layerDimRow,), dtype=float))
            
    def feedForward(self, x):
        v_i = x
        for k in range(len(self.layerWidths)-1):
            self.hiddens[k] = np.matmul( self.layers[k], v_i ) + self.weight0[k]
            self.hiddens[k] = 1.0/(1.0 + np.exp(-self.hiddens[k]))
            v_i = self.hiddens[k]

    def backpropagate(self, y):
        loss_g = self.hiddens[-1]*(1-self.hiddens[-1])*(y - self.hiddens[-1])
        self.errors[-1] = loss_g
        for k in range(len(self.layerWidths)-2, 0, -1):
            self.errors[k-1] = self.hiddens[k-1] * (1 - self.hiddens[k-1]) * np.matmul( self.layers[k].transpose(), self.errors[k])

    def updateweights(self, x, step, momentum):
        w0_delta = step * self.errors[0]
        self.weight0[0] += w0_delta + momentum * self.prev_w0_delta[0]
        self.prev_w0_delta[0] = w0_delta

        w_delta = np.matmul( np.expand_dims( step * self.errors[0], axis=1 ), np.expand_dims( np.array(x), axis=0))
        self.layers[0] += w_delta + momentum * self.prev_w_delta[0]
        self.prev_w_delta[0] = w_delta

        for k in range(1, len(self.layerWidths)-1):
            w0_delta = step * self.errors[k]
            self.weight0[k] += w0_delta + momentum * self.prev_w0_delta[k]
            self.prev_w0_delta[k] = w0_delta

            w_delta = np.matmul( np.expand_dims( step * self.errors[k], axis=1 ), np.expand_dims( self.hiddens[k-1], axis=0) ) 
            self.layers[k] += w_delta + momentum * self.prev_w_delta[k]
            self.prev_w_delta[k] = w_delta

    def loss(self, x, y):        
        return EvaluateBinaryProbabilityEstimate.MeanSquaredErrorLoss(y, self.predictProbabilities(x))

    def predictOneProbability(self, x):
        self.feedForward(x)
        
        # return the activation of the neuron in the output layer
        return self.hiddens[-1][0]

    def predictProbabilities(self, x):    
        return [ self.predictOneProbability(sample) for sample in x ]

    def predict(self, x, threshold = 0.5):
        return [ 1 if probability > threshold else 0 for probability in self.predictProbabilities(x) ]
    
    def __CheckForConvergence(self, x, y, convergence):
        loss = self.loss(x,y)
        print("model loss - {}".format(loss))
        if self.lastLoss != None:
            deltaLoss = abs(self.lastLoss - loss)
            self.converged = deltaLoss < convergence

        self.lastLoss = loss
    
    # Allows you to partially fit, then pause to gather statistics / output intermediate information, then continue fitting
    def incrementalFit(self, x, y, epochs=1, step=0.01, momentum=0.1, convergence = None):
        for _ in range(epochs):
            if self.converged:
                return
        
            # do a full epoch of stocastic gradient descent
            for x_, y_ in zip(x,y):
                self.feedForward(x_)
                self.backpropagate(y_)
                self.updateweights(x_, step, momentum)

            self.totalEpochs += 1
            if convergence is not None:
                self.__CheckForConvergence(x, y, convergence)
             
                
    def fit(self, x, y, maxEpochs=50000, stepSize=0.01, convergence=0.00001, verbose = True):        
        startTime = time.time()
        
        self.incrementalFit(x, y, epochs=maxEpochs, step=stepSize, convergence=convergence)
        
        endTime = time.time()
        runtime = endTime - startTime
      
        if not self.converged:
            print("Warning: NeuralNetwork did not converge after the maximum allowed number of epochs.")
        elif verbose:
            print("NeuralNetwork converged in %d epochs (%.2f seconds) -- %d features. Hyperparameters: stepSize=%f and convergence=%f." % (self.totalEpochs, runtime, len(x[0]), stepSize, convergence))

