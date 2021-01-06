# This file contains stubs for evaluating binary classifications. You must complete these functions as part of your assignment.
#     Each function takes in: 
#           'y':           the arrary of 0/1 true class labels; 
#           'yPredicted':  the prediction your model made for the cooresponding example.


def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value not in [0, 1]:
            valueError = True
    for value in yPredicted:
        if value not in [0, 1]:
            valueError = True

    if valueError:
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected values. Must be 0 or 1.")

def Accuracy(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    correct = []
    for i in range(len(y)):
        if(y[i] == yPredicted[i]):
            correct.append(1)
        else:
            correct.append(0)

    return sum(correct)/len(correct)

def Precision(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    positives = []
    for i in range(len(yPredicted)):
        if yPredicted[i] == 1 and y[i] == 1:
            positives.append(1)
        elif yPredicted[i] == 1 and y[i] == 0:
            positives.append(0)

    if len(positives) == 0:
        return 0

    return sum(positives)/len(positives)

def Recall(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    recalled = []
    for i in range(len(y)):
        if y[i] == 1 and yPredicted[i] == 1:
            recalled.append(1)
        elif y[i] == 1 and yPredicted[i] == 0:
            recalled.append(0)

    if len(recalled) == 0:
        return 0

    return sum(recalled)/len(recalled)

def FalseNegativeRate(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    falseNegatives = []
    for i in range(len(y)):
        if y[i] == 1 and yPredicted[i] == 1:
            falseNegatives.append(0)
        elif y[i] == 1 and yPredicted[i] == 0:
            falseNegatives.append(1)
    
    if len(falseNegatives) == 0:
        return 0

    return sum(falseNegatives)/len(falseNegatives)

def FalsePositiveRate(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    falsePositives = []
    for i in range(len(y)):
        if y[i] == 0 and yPredicted[i] == 0:
            falsePositives.append(0)
        elif y[i] == 0 and yPredicted[i] == 1:
            falsePositives.append(1)

    if len(falsePositives) == 0:
        return 0

    return sum(falsePositives)/len(falsePositives)

def ConfusionMatrix(y, yPredicted):
    # This function should return: [[<# True Negatives>, <# False Positives>], [<# False Negatives>, <# True Positives>]]
    #  Hint: writing this function first might make the others easier...
    
    falsePositive = FalsePositiveRate(y, yPredicted)
    falseNegative = FalseNegativeRate(y, yPredicted)

    return [ [1-falseNegative, falsePositive], [falseNegative, 1-falsePositive]]

def ExecuteAll(y, yPredicted):
    print(ConfusionMatrix(y, yPredicted))
    print("Accuracy:", Accuracy(y, yPredicted))
    print("Precision:", Precision(y, yPredicted))
    print("Recall:", Recall(y, yPredicted))
    print("FPR:", FalsePositiveRate(y, yPredicted))
    print("FNR:", FalseNegativeRate(y, yPredicted))
    
