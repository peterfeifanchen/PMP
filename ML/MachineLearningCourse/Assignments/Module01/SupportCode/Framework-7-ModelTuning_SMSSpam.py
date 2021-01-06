kOutputDirectory = "MachineLearningCourse/Assignments/Module01/Graphs/visualize\\"

import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

kDataPath = "MachineLearningCourse/MLProjectSupport/SMSSpam/dataset/SMSSpamCollection"

(xRaw, yRaw) = SMSSpamDataset.LoadRawData(kDataPath)

import MachineLearningCourse.MLUtilities.Data.Sample as Sample
(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation

import time
import numpy as np

# A helper function for calculating FN rate and FP rate across a range of thresholds
def TabulateModelPerformanceForROC(model, xValidate, yValidate):
   pointsToEvaluate = 100
   thresholds = [ x / float(pointsToEvaluate) for x in range(pointsToEvaluate + 1)]
   FPRs = []
   FNRs = []

   try:
      for threshold in thresholds:
         FPRs.append(EvaluateBinaryClassification.FalsePositiveRate(yValidate, model.predict(xValidate, classificationThreshold=threshold)))
         FNRs.append(EvaluateBinaryClassification.FalseNegativeRate(yValidate, model.predict(xValidate, classificationThreshold=threshold)))
   except NotImplementedError:
      raise UserWarning("The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

   return (FPRs, FNRs, thresholds)

import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
## This function will help you plot with error bars. Use it just like PlotSeries, but with parallel arrays of error bar sizes in the second variable
#     note that the error bar size is drawn above and below the series value. So if the series value is .8 and the confidence interval is .78 - .82, then the value to use for the error bar is .02

# Charting.PlotSeriesWithErrorBars([series1, series2], [errorBarsForSeries1, errorBarsForSeries2], ["Series1", "Series2"], xValues, chartTitle="<>", xAxisTitle="<>", yAxisTitle="<>", yBotLimit=0.8, outputDirectory=kOutputDirectory, fileName="<name>")

## This helper function should execute a single run and save the results on 'runSpecification' (which could be a dictionary for convienience)
#    for later tabulation and charting...
def ExecuteEvaluationRun(runSpecification, xTrainRaw, yTrain, numberOfFolds = 2):
    print( "runSpecification: ", runSpecification)
    startTime = time.time()
    
    # HERE upgrade this to use crossvalidation    
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords = runSpecification['numFrequentWords'], numMutualInformationWords = runSpecification['numMutualInformationWords'])

    xTrain      = featurizer.Featurize(xTrainRaw)
    xValidate   = featurizer.Featurize(xValidateRaw)

    if numberOfFolds > 1:
        crossValidationAccuracy = []
        for i in range(numberOfFolds):
            xTrainI, yTrainI, xEvaluateI, yEvaluateI = CrossValidation.CrossValidation(xTrain, yTrain, numberOfFolds, i)

            model = LogisticRegression.LogisticRegression()
            model.fit(xTrainI,yTrainI,convergence=runSpecification['convergence'], stepSize=runSpecification['stepSize'], verbose=False)

            crossValidationAccuracy.append(EvaluateBinaryClassification.Accuracy(yEvaluateI, model.predict(xEvaluateI)))
    
        mean = np.mean(crossValidationAccuracy)
        runSpecification['crossValidationMean'] = mean
        lower, _ = ErrorBounds.GetAccuracyBounds(np.mean(crossValidationAccuracy), len(yEvaluateI), .5)
        runSpecification['crossValidationErrorBound'] = mean - lower
        
    
    if numberOfFolds == 1:
        model = LogisticRegression.LogisticRegression()
        model.fit(xTrain, yTrain, convergence=runSpecification['convergence'], stepSize=runSpecification['stepSize'], verbose=False)
        validationSetAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, model.predict(xValidate))
    
        runSpecification['accuracy'] = validationSetAccuracy
        lower, _ = ErrorBounds.GetAccuracyBounds(validationSetAccuracy, len(yValidate), .5)
        runSpecification['accuracyErrorBound'] = validationSetAccuracy - lower
    
    endTime = time.time()
    if numberOfFolds > 1:
        runSpecification['runtime'] = endTime - startTime
    
    return runSpecification

def ExecuteSweep( sweepSpec, optSpec, plotString, plot=False ):
    evaluationRunSpecifications = []
    for sweepValue in sweepSpec[optSpec]:
        runSpecification = {}
        runSpecification['optimizing'] = optSpec
        runSpecification['numMutualInformationWords'] = sweepSpec['numMutualInformationWords']
        runSpecification['stepSize'] = sweepSpec['stepSize']
        runSpecification['convergence'] = sweepSpec['convergence']
        runSpecification['numFrequentWords'] = sweepSpec['numFrequentWords']
        runSpecification[optSpec] = sweepValue 
        evaluationRunSpecifications.append(runSpecification)

    ## if you want to run in parallel you need to install joblib as described in the lecture notes and adjust the comments on the next three lines...
    from joblib import Parallel, delayed
    evaluations = Parallel(n_jobs=4)(delayed(ExecuteEvaluationRun)(runSpec, xTrainRaw, yTrain) for runSpec in evaluationRunSpecifications)

    #evaluations = [ ExecuteEvaluationRun(runSpec, xTrainRaw, yTrain) for runSpec in evaluationRunSpecifications ]
    if plot:
        xValues = sweepSpec[optSpec]
        series1 = [ runSpec['crossValidationMean'] for runSpec in evaluations ]
        errorBarsForSeries1 = [ runSpec['crossValidationErrorBound'] for runSpec in evaluations]
        series2 = [ runSpec['runtime'] for runSpec in evaluations ]
        Charting.PlotSeriesWithErrorBars([series1], [errorBarsForSeries1], ["Cross Validation Accuracy"], xValues, chartTitle=plotString + " Cross Validation Accuracy", xAxisTitle=plotString + " Values", yAxisTitle="Cross Valid. Accuracy", yBotLimit=0.8, outputDirectory=kOutputDirectory, fileName="7-"+plotString+"CrossValidationAccuracy")
        Charting.PlotSeries([series2], ['runtime'], xValues, chartTitle="Cross Validation Runtime", xAxisTitle=plotString + " Values", yAxisTitle="Run Time", outputDirectory=kOutputDirectory, fileName="7-"+plotString+"CrossValidationRunTime")
    return evaluations

def findBestParameters(specs):
    specs = sorted(specs, key=lambda x: x['crossValidationMean'])
    best = specs[0]
    for i in range(1, len(specs)):
        bestUpper = best['crossValidationMean'] + specs[i]['crossValidationErrorBound']
        if specs[i]['crossValidationMean'] > bestUpper:
            best = specs[i]
        elif specs[i]['crossValidationMean'] > best['crossValidationMean'] and specs[i]['runtime'] < best['runtime']:
            best = specs[i]

    return {
        'stepSize': best['stepSize'], 
        'convergence': best['convergence'], 
        'numFrequentWords': best['numFrequentWords'], 
        'numMutualInformationWords': best['numMutualInformationWords']
     }

sweep = list(range(4))
bestParameters = {}
bestParameters['stepSize'] = 1.0
bestParameters['convergence'] = 0.005
bestParameters['numFrequentWords'] = 0
bestParameters['numMutualInformationWords'] = 20

init = ExecuteEvaluationRun(bestParameters, xTrainRaw, yTrain, 1)
validationSetAccuracy = [ init['accuracy'] ]
validationSetAccuracyError = [ init['accuracyErrorBound'] ]

interval = {}
interval['stepSize'] = 10 #percent
interval['convergence'] = 30 # percent
interval['numFrequentWords'] = 20
interval['numMutualInformationWords'] = 20

for i in sweep:
    print("sweep ", i)
    currResult = ExecuteEvaluationRun(bestParameters, xTrainRaw, yTrain)
    prevModelAccuracyUpper = min(currResult['crossValidationMean'] + currResult['crossValidationErrorBound'], 1)
    print("curr parameters: ", currResult['crossValidationMean'], " +/- ", currResult['crossValidationErrorBound'])

    if i % 4 == 0:
        if i < 4:
            bestParameters['numFrequentWords'] = [0, 20, 50, 100, 150, 300]
        else:
            currentBest = bestParameters['numFrequentWords']
            newInterval = interval['numFrequentWords']/3
            newRange = list(range(currentBest), currentBest-interval['numFrequentWords'], -newInterval)
            newRange.reverse()
            newRange += list(range(currentBest+newInterval, currentBest+interval['numFrequentWords'], newInterval))
            interval['numFrequentWords'] = newInterval
            bestParameters['numFrequentWords'] = newRange
        specs = ExecuteSweep( bestParameters, "numFrequentWords", "Num Frequent Words", i < 4)
    elif i % 4 == 1:
        if i < 4:
            bestParameters['numMutualInformationWords'] = [5, 20, 50, 100, 150, 300]
        else:
            currentBest = bestParameters['numMutualInformationWords']
            newInterval = interval['numMutualInformationWords']/3
            newRange = list(range(currentBest), currentBest-interval['numMutualInformationWords'], -newInterval)
            newRange.reverse()
            newRange += list(range(currentBest+newInterval, currentBest+interval['numMutualInformations'], newInterval))
            interval['numMutualInformationWords'] = newInterval
            bestParameters['numMutualInformationWords'] = newRange
        specs = ExecuteSweep( bestParameters, "numMutualInformationWords", "Num MI Words", i < 4 )
    elif i % 4 == 2:
        if i < 4:
            bestParameters['stepSize'] = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        else:
            currentBest = bestParameters['stepSize']
            newInterval = interval['stepSize']/100 * currentBest / pow(10, int(i/4) - 1)
            newRange = list(range(currentBest), currentBest-newInterval*3, -newInterval)
            newRange.reverse()
            newRange += list(range(currentBest+newInterval, currentBest+newInterval*2, newInterval))
            bestParameters['stepSize'] = newRange
        specs = ExecuteSweep( bestParameters, "stepSize", "Step Size", i < 4)
    else:
        if i < 4:
            bestParameters['convergence'] = [0.1, 0.01, 0.005, 0.001, 0.0001]
        else:
            currentBest = bestParameters['convergence']
            newInterval = interval['convergence']/100 * currentBest / pow(10, int(i/4)-1)
            newRange = list(range(currentBest), currentBest-newInterval*3, -newInterval)
            newRange.reverse()
            newRange += list(range(currentBest+newInterval, currentBest+newInterval*2, newInterval))
            bestParameters['convergence'] = newRange
        specs = ExecuteSweep( bestParameters, "convergence", "convergence", i < 4 )
       
    newBestParameters = findBestParameters(specs)

    newBestResult = ExecuteEvaluationRun(newBestParameters, xTrainRaw, yTrain)
    print("new parameters: ", newBestResult['crossValidationMean'], " +/- ", newBestResult['crossValidationErrorBound'])

    if i % 4 == 0 and i > 3 and newBestResult['crossValidationMean'] < prevModelAccuracyUpper:
        break

    bestParameters = {
        'stepSize': newBestParameters['stepSize'], 
        'convergence': newBestParameters['convergence'], 
        'numFrequentWords': newBestParameters['numFrequentWords'], 
        'numMutualInformationWords': newBestParameters['numMutualInformationWords']
     }
    result = ExecuteEvaluationRun(bestParameters, xTrainRaw, yTrain, 1)
    validationSetAccuracy.append( result['accuracy'] )
    validationSetAccuracyError.append( result['accuracyErrorBound'] )

# Plot Validation Accuracy
lastSweep = sweep[len(sweep)-1]
Charting.PlotSeriesWithErrorBars([validationSetAccuracy], [validationSetAccuracyError], ["Accuracy"], sweep+[lastSweep+1], chartTitle="Validation Set Accuracy", xAxisTitle="Sweep #", yAxisTitle="Validation Set Accuracy", yBotLimit=0.8, outputDirectory=kOutputDirectory, fileName="7-ValidationSetAccuracy")
print("BestParameters: ", bestParameters)

# ROC of initial vs. best
seriesFPRs = []
seriesFNRs = []
seriesLabels = []
init = {}
init['stepSize'] = 1.0
init['convergence'] = 0.005
init['numFrequentWords'] = 0
init['numMutualInformationWords'] = 20

featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords = init['numFrequentWords'], numMutualInformationWords = init['numMutualInformationWords'])

xTrain      = featurizer.Featurize(xTrainRaw)
xTest       = featurizer.Featurize(xTestRaw)

model = LogisticRegression.LogisticRegression()
model.fit(xTrain, yTrain, convergence=init['convergence'], stepSize=init['stepSize'], verbose=False)

(modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xTest, yTest)
seriesFPRs.append(modelFPRs)
seriesFNRs.append(modelFNRs)
seriesLabels.append('initial parameters')

featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords = bestParameters['numFrequentWords'], numMutualInformationWords = bestParameters['numMutualInformationWords'])

xTrain      = featurizer.Featurize(xTrainRaw)
xTest       = featurizer.Featurize(xTestRaw)

model = LogisticRegression.LogisticRegression()
model.fit(xTrain, yTrain, convergence=bestParameters['convergence'], stepSize=bestParameters['stepSize'], verbose=False)

(modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xTest, yTest)
seriesFPRs.append(modelFPRs)
seriesFNRs.append(modelFNRs)
seriesLabels.append('best parameters')

Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC Comparison", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="7-BestInitModelROC")