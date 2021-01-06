import math
import numpy as np

def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value < 0 or value > 1:
            valueError = True
    for value in yPredicted:
        if value < 0 or value > 1:
            valueError = True

    if valueError:
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected values. Must be between 0 and 1.")

def MeanSquaredErrorLoss(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)
    return 0.5 * np.dot( np.array(y) - np.array(yPredicted), np.array(y) - np.array(yPredicted) )

def approxLog(v):
    vLogApprox = []
    for i in v:
        if i < 0.000000000001:
            vLog = 0.0
        else:
            vLog = np.log(i)
        vLogApprox.append(vLog)
    return vLogApprox

def LogLoss(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)
    yPredicted = np.array(yPredicted)
    y = np.array(y)
    
    totalLoss = np.multiply( y, approxLog(yPredicted)) + np.multiply( 1-y, approxLog(1-yPredicted))
    return -np.sum(totalLoss)/len(totalLoss)

