#import numpy as np
import time
import torch
import torchvision.transforms as transforms
#from torch.utils import data
from PIL import Image

import MachineLearningCourse.MLProjectSupport.Blink.BlinkDataset as BlinkDataset
import MachineLearningCourse.Assignments.Module03.SupportCode.BlinkNeuralNetwork as BlinkNeuralNetwork
import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds

(xRaw, yRaw) = BlinkDataset.LoadRawData()

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw)

print("Train is %d samples, %.4f percent opened." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
print("Validate is %d samples, %.4f percent opened." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
print("Test is %d samples %.4f percent opened" % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

##
# Load the images, normalize and convert them into tensors
##

transform = transforms.Compose([
            transforms.ToTensor()
            ,transforms.Normalize(mean=[0.], std=[0.5])
            ])

xTrainImages = [ Image.open(path) for path in xTrainRaw ]
xTrain = torch.stack([ transform(image) for image in xTrainImages ])

yTrain = torch.Tensor([ [ yValue ] for yValue in yTrain ])

xValidateImages = [ Image.open(path) for path in xValidateRaw ]
xValidate = torch.stack([ transform(image) for image in xValidateImages ])

yValidate = torch.Tensor([ [ yValue ] for yValue in yValidate ])

xTestImages = [ Image.open(path) for path in xTestRaw ]
xTest = torch.stack([ transform(image) for image in xTestImages ])

yTest = torch.Tensor([ [ yValue ] for yValue in yTest ])


#trainingDataSet = BlinkNeuralNetwork.BlinkDataSet(xTrainRaw, yTrain)
#validationDataSet = BlinkNeuralNetwork.BlinkDataSet(xValidateRaw, yValidate)
#testDataSet = BlinkNeuralNetwork.BlinkDataSet(xTestRaw, yTest)

#batchSize = 1024
#trainDataSetGenerator = data.DataLoader(trainingDataSet, batch_size=batchSize, shuffle=True, num_workers=0)
#validationDataSetGenerator = data.DataLoader(validationDataSet, batch_size=batchSize, shuffle=True, num_workers=0)
#testDataSetGenerator = data.DataLoader(testDataSet, batch_size=batchSize, shuffle=True, num_workers=0)

##
# Move the model and data to the GPU if you're using your GPU
##
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is:", device)

kOutputDirectory = "MachineLearningCourse/Assignments/Module03/Graphs/tuning/"

# A helper function for calculating FN rate and FP rate across a range of thresholds
def TabulateModelPerformanceForROC(model, xData, yData):
    pointsToEvaluate = 100
    thresholds = [ x / float(pointsToEvaluate) for x in range(pointsToEvaluate + 1)]
    FPRs = []
    FNRs = []

    try:
        for threshold in thresholds:
            yPredicted = model(xData) 
            FPRs.append(EvaluateBinaryClassification.FalsePositiveRate(yData, [ 1 if pred > threshold else 0 for pred in yPredicted ]))
            FNRs.append(EvaluateBinaryClassification.FalseNegativeRate(yData, [ 1 if pred > threshold else 0 for pred in yPredicted ]))
    except NotImplementedError:
        raise UserWarning("The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

    return (FPRs, FNRs, thresholds)

def trainModel( m, op, maxE, patience=10, saveChartName=""):
    startTime = time.time()
    
    lossFunction = torch.nn.BCELoss(reduction='mean')

    trainLosses = []
    validationLosses = []

    converged = False
    epoch = 1
    lastValidationLoss = None

    currPatience = 0
    while not converged and epoch < maxE:
        # Reset the gradients in the network to zero
        op.zero_grad()
        #for batchXTensor, batchYTensor in trainDataSetGenerator:
        #    x = batchXTensor.to(device)
        #    y = batchYTensor.to(device)
        #
            # Do the forward pass
        #    yPredicted = m(x)

            # Compute the total loss summed across training samples in the epoch
            #  note this is different from our implementation, which took one step
            #  of gradient descent per sample.
        #    trainLoss = lossFunction(yPredicted, y)

            # Backprop the errors from the loss on this iteration
        #    trainLoss.backward()
        yTrainPredicted = m(xTrain)
        trainLoss = lossFunction(yTrainPredicted, yTrain)
        trainLoss.backward()
        # Do a weight update step
        op.step()
    
        # now check the validation loss
        m.train(mode=False)
        #validationLossTotal = 0
        #for batchXTensor, batchYTensor in validationDataSetGenerator:
        #    x = batchXTensor.to(device)
        #    y = batchYTensor.to(device)
        #    yPredicted = m(x)
        #    validationLoss = lossFunction(yPredicted, y)
        #    validationLossTotal += validationLoss.item()
        #validationLoss = validationLossTotal / len(validationDataSet)
        #validationLosses.append(validationLoss)
        yValidationPredicted = m(xValidate)
        validationLoss = lossFunction(yValidationPredicted, yValidate)


        #trainingLossTotal = 0
        #for batchXTensor, batchYTensor in trainDataSetGenerator:
        #    x = batchXTensor.to(device)
        #    y = batchYTensor.to(device)
        #    yPredicted = m(x)
        #    trainLoss = lossFunction(yPredicted, y) 
        #    trainingLossTotal += trainLoss.item()
        #trainLosses.append(trainingLossTotal / len(trainingDataSet))
    
        #print("epoch %d: training loss {}, validation loss {}".format(epoch, trainLosses[-1], validationLoss))
        #if lastValidationLoss is not None and validationLoss > lastValidationLoss and saveChartName == "":
        #    converged = True
        #else:
        #    lastValidationLoss = validationLoss
        yTrainingPredicted = m(xTrain)
        trainLoss = lossFunction(yTrainingPredicted, yTrain)
        trainLosses.append(trainLoss.item()/len(yTrain))
        validationLosses.append(validationLoss.item()/len(yValidate))
        print("epoch {}: training loss {}, validation loss {}".format(epoch, trainLosses[-1], validationLosses[-1]))
        if lastValidationLoss is not None and validationLoss > lastValidationLoss:
            if currPatience < patience:
                currPatience += 1
            else:
                converged = True
        else:
            lastValidationLoss = validationLoss
            currPatience = 0
        epoch = epoch + 1
        m.train(mode=True)

    endTime = time.time()
    print("Runtime: %s" % (endTime - startTime))

    ##
    # Visualize Training run
    ##
    if saveChartName != "":
        xValues = [ i + 1 for i in range(len(trainLosses))]
        Charting.PlotSeries([trainLosses, validationLosses], ["Train Loss", "Validate Loss"], xValues, useMarkers=False, chartTitle="Blink LeNet Model Loss/Epoch", xAxisTitle="Epoch", yAxisTitle="Loss", yBotLimit=0.0, outputDirectory=kOutputDirectory, fileName="4-"+saveChartName)

    ##
    # Get the model accuracy on validation set
    ##
    model.train(mode=False)
    #yValidatePredicted = []
    #for batchXTensor, batchYTensor in validationDataSetGenerator:
    #        x = batchXTensor.to(device)
    #        y = batchYTensor.to(device)
    #        yPredicted = m(x)
    #        yValidatePredicted += yPredicted.tolist()
    yValidatePredicted = m(xValidate)
    return EvaluateBinaryClassification.Accuracy(yValidate, [ 1 if pred > 0.5 else 0 for pred in yValidatePredicted ]) 

# Hyperparameter Space
conv1FilterNum = [6, 12, 18]
conv1WindowSize = [2, 4, 6]
conv2FilterNum = [12, 18, 24]
conv2WindowSize = [4, 5, 6]
fc1HiddenSize = [5, 10, 20]
fc2HiddenSize = [5, 10, 20] 
maxEpoch = 5000
step = 0.001
momentum = 0.9
# Number of times to average training results for hyperparameter search
numTrials = 3

initHiddenSweep = {}
# Initial parameter sweeps for hidden sizes
#for h1 in fc1HiddenSize:
#    for h2 in fc2HiddenSize:
#        state = "{} {}".format(h1, h2)
#        accTotal = 0
#        for n in range(numTrials):
#            model = BlinkNeuralNetwork.LeNet(imageSize=24, convFilters=[], fcLayers=[h1, h2])
#            model.to(device)
#            #optimizer = torch.optim.SGD(model.parameters(), lr=step, momentum=momentum)
#            optimizer = torch.optim.Adam(model.parameters(), lr=step)
#            acc = trainModel( model, optimizer, maxEpoch, 25, "LossEpochHidden{}-{}-{}".format(h1, h2, n))
#            #lower, upper = trainModel( model, optimizer, maxEpoch)
#            accTotal += acc
#        accTotal = accTotal / numTrials
#        lower, _ = ErrorBounds.GetAccuracyBounds(accTotal, len(yValidate), 0.95)
#        initHiddenSweep[state] = ( accTotal, accTotal - lower )

#for item in initHiddenSweep.items():
#    print(item[0], " {:0.4f} +/- {:0.4f}".format(item[1][0], item[1][1]))

# Parameter search the first layer convolution
#accuracyYResults = []
#accuracyYLabels = []
#initConv1Sweep = {}
#for f in conv1FilterNum:
#    accuracyYLabels.append("{} filters".format(f))
#    accuracyYResults.append([])
#    for w in conv1WindowSize:
#        accTotal = 0
#        state = "{} {}".format(f, w)
#        for n in range(numTrials):
#            # Create the model
#            model = BlinkNeuralNetwork.LeNet(imageSize=xTrain[0].shape[1], convFilters=[ (f, w) ], fcLayers=[ 20, 10 ])
#            model.to(device)
#            # Create the optimization method (Stochastic Gradient Descent) and the step size (lr -> learning rate)
#            optimizer = torch.optim.Adam(model.parameters(), lr=step)
#            acc = trainModel( model, optimizer, maxEpoch, 25, "LossEpochConv1{}-{}-{}".format(f, w, n))
#            accTotal += acc
#        accTotal = accTotal / numTrials
#        lower, _ = ErrorBounds.GetAccuracyBounds(accTotal, len(yValidate), 0.95)
#        accuracyYResults[-1].append(accTotal)
#        initConv1Sweep[state] = ( accTotal, accTotal - lower )
#Charting.PlotSeries(accuracyYResults, accuracyYLabels, conv1WindowSize, useMarkers=False, chartTitle="Blink LeNet Model Conv Layer Accuracy", xAxisTitle="Window Size", yAxisTitle="Accuracy", yBotLimit=0.8, outputDirectory=kOutputDirectory, fileName="4-Conv1LayerTuning")   

#for item in initConv1Sweep.items():
#    print(item[0], " {:0.4f} +/- {:0.4f}".format(item[1][0], item[1][1]))

#accuracyYResults = []
#accuracyYLabels = []
#initConv2Sweep = {}
#for f in conv2FilterNum:
#    accuracyYLabels.append("{} filters".format(f))
#    accuracyYResults.append([])
#    for w in conv2WindowSize:
#        accTotal = 0
#        state = "{} {}".format(f, w)
#        for n in range(numTrials):
#            # Create the model
#            model = BlinkNeuralNetwork.LeNet(imageSize=xTrain[0].shape[1], convFilters=[ (12, 6), (f, w) ], fcLayers=[ 20, 10 ])
#            model.to(device)
#            # Create the optimization method (Stochastic Gradient Descent) and the step size (lr -> learning rate)
#            optimizer = torch.optim.Adam(model.parameters(), lr=step)
#            acc = trainModel( model, optimizer, maxEpoch, 25, "LossEpochConv2{}-{}-{}".format(f, w, n))
#            accTotal += acc
#        accTotal = accTotal / numTrials
#        lower, _ = ErrorBounds.GetAccuracyBounds(accTotal, len(yValidate), 0.95)
#        accuracyYResults[-1].append(accTotal)
#        initConv2Sweep[state] = ( accTotal, accTotal - lower )
#Charting.PlotSeries(accuracyYResults, accuracyYLabels, conv2WindowSize, useMarkers=False, chartTitle="Blink LeNet Model Conv Layer 2 Accuracy", xAxisTitle="Window Size", yAxisTitle="Accuracy", yBotLimit=0.8, outputDirectory=kOutputDirectory, fileName="4-Conv2LayerTuning")   

#for item in initConv2Sweep.items():
#    print(item[0], " {:0.4f} +/- {:0.4f}".format(item[1][0], item[1][1]))

 ##
 # Evaluate the Model
 ##
num_trials = 3
validationAccuracyResults = []
testAccuracyResults = []

# Set up to hold information for creating ROC curves
seriesFPRs = []
seriesFNRs = []
seriesLabels = []

errorImages = {}

for i in range(num_trials):
    errorImages[i] = []
    model = BlinkNeuralNetwork.LeNet(imageSize=xTrain[0].shape[1], convFilters=[ (12, 6), (18, 5) ], fcLayers=[20, 10])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=step)
    acc = trainModel( model, optimizer, maxEpoch, 25, "LossEpochFinal12-6-x-x-20-10-{}".format(i))
    lower, _ = ErrorBounds.GetAccuracyBounds(acc, len(yValidate), 0.95)
    validationAccuracyResults.append( (acc, acc-lower) )

    yTestPredicted = model(xTest)
    testAccuracy = EvaluateBinaryClassification.Accuracy(yTest, [ 1 if pred > 0.5 else 0 for pred in yTestPredicted ])
    lowerTest, _ = ErrorBounds.GetAccuracyBounds(testAccuracy, len(yTest), 0.95)
    testAccuracyResults.append( (testAccuracy, testAccuracy - lowerTest) )

    yTestPredict = [ 1 if pred > 0.5 else 0 for pred in yTestPredicted ]
    for j in range(len(yTest)):
        if int(yTest[j]) != yTestPredict[j]:
            errorImages[i].append(xTestRaw[j])

    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xTest, yTest)
    seriesFPRs.append(modelFPRs)
    seriesFNRs.append(modelFNRs)
    seriesLabels.append('model {}'.format(i))

Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC Comparison", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="Plot-BlinkROCFinal")

for i in range(num_trials):
    for image in errorImages[i]:
        print("trial {}: incorrect image {}".format(i, image))
print(validationAccuracyResults)
print(testAccuracyResults)