kOutputDirectory = "MachineLearningCourse/Assignments/Module02/Graphs/visualize\\"

import MachineLearningCourse.MLProjectSupport.Adult.AdultDataset as AdultDataset

### UPDATE this path for your environment
kDataPath = "MachineLearningCourse/MLProjectSupport/Adult/dataset/adult.data"

(xRaw, yRaw) = AdultDataset.LoadRawData(kDataPath)

import MachineLearningCourse.MLUtilities.Data.Sample as Sample

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw)

print("Train is %d samples, %.4f percent >50K." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
print("Validate is %d samples, %.4f percent >50K." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
print("Test is %d samples %.4f percent >50K." % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

import MachineLearningCourse.Assignments.Module02.SupportCode.AdultFeaturize as AdultFeaturize

featurizer = AdultFeaturize.AdultFeaturize()
featurizer.CreateFeatureSet(xTrainRaw, yTrain, useCategoricalFeatures = True, useNumericFeatures = False)

for i in range(featurizer.GetFeatureCount()):
    print("%d - %s" % (i, featurizer.GetFeatureInfo(i)))

xTrain    = featurizer.Featurize(xTrainRaw)
xValidate = featurizer.Featurize(xValidateRaw)
xTest     = featurizer.Featurize(xTestRaw)

for i in range(10):
    print("%d - " % (yTrain[i]), xTrain[i])

############################
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.MLUtilities.Learners.MostCommonClassModel as MostCommonClassModel

model = MostCommonClassModel.MostCommonClassModel()
model.fit(xTrain, yTrain)
yValidatePredicted = model.predict(xValidate)
validateAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, yValidatePredicted)
errorBounds = ErrorBounds.GetAccuracyBounds(validateAccuracy, len(yValidate), 0.95)

print()
print("### 'Most Common Class' model validate set accuracy: %.4f (95%% %.4f - %.4f)" % (validateAccuracy, errorBounds[0], errorBounds[1]))

import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation
import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
import time
import numpy as np
## This helper function should execute a single run and save the results on 'runSpecification' (which could be a dictionary for convienience)
#    for later tabulation and charting...
def ExecuteEvaluationRun(runSpecification, xTrain, yTrain, numberOfFolds = 2):
    print( "runSpecification: ", runSpecification)
    startTime = time.time()

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
        runSpecification['stepSize'] = sweepSpec['stepSize']
        runSpecification['convergence'] = sweepSpec['convergence']
        runSpecification[optSpec] = sweepValue 
        evaluationRunSpecifications.append(runSpecification)

    ## if you want to run in parallel you need to install joblib as described in the lecture notes and adjust the comments on the next three lines...
    from joblib import Parallel, delayed
    evaluations = Parallel(n_jobs=4)(delayed(ExecuteEvaluationRun)(runSpec, xTrain, yTrain) for runSpec in evaluationRunSpecifications)

    if plot:
        xValues = sweepSpec[optSpec]
        series1 = [ runSpec['crossValidationMean'] for runSpec in evaluations ]
        errorBarsForSeries1 = [ runSpec['crossValidationErrorBound'] for runSpec in evaluations]
        # series2 = [ runSpec['runtime'] for runSpec in evaluations ]
        Charting.PlotSeriesWithErrorBars([series1], [errorBarsForSeries1], ["Cross Validation Accuracy"], xValues, chartTitle=plotString + " Cross Validation Accuracy", xAxisTitle=plotString + " Values", yAxisTitle="Cross Valid. Accuracy", yBotLimit=0.7, outputDirectory=kOutputDirectory, fileName="1-"+plotString+"CrossValidationAccuracy")
        #Charting.PlotSeries([series2], ['runtime'], xValues, chartTitle="Cross Validation Runtime", xAxisTitle=plotString + " Values", yAxisTitle="Run Time", outputDirectory=kOutputDirectory, fileName="1-"+plotString+"CrossValidationRunTime")
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
     }

sweep = list(range(2))
bestParameters = {}
bestParameters['stepSize'] = 1.0
bestParameters['convergence'] = 0.005

init = ExecuteEvaluationRun(bestParameters, xTrain, yTrain, 1)
validationSetAccuracy = [ init['accuracy'] ]
validationSetAccuracyError = [ init['accuracyErrorBound'] ]

interval = {}
interval['stepSize'] = 10 #percent
interval['convergence'] = 30 # percent

for i in sweep:
    print("sweep ", i)
    currResult = ExecuteEvaluationRun(bestParameters, xTrain, yTrain)
    prevModelAccuracyUpper = min(currResult['crossValidationMean'] + currResult['crossValidationErrorBound'], 1)
    print("curr parameters: ", currResult['crossValidationMean'], " +/- ", currResult['crossValidationErrorBound'])

    if i % 2 == 0:
        if i < 2:
            bestParameters['stepSize'] = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        else:
            currentBest = bestParameters['stepSize']
            newInterval = interval['stepSize']/100 * currentBest / pow(10, int(i/2)-1)
            newRange = list(np.arange(currentBest, currentBest-newInterval*3, -newInterval))
            newRange.reverse()
            newRange += list(np.arange(currentBest+newInterval, currentBest+newInterval*2, newInterval))
            bestParameters['stepSize'] = newRange
        specs = ExecuteSweep( bestParameters, "stepSize", "Step Size", i < 2 )
    else:
        if i < 2:
            bestParameters['convergence'] = [0.1, 0.05, 0.01, 0.005, 0.001]
        else:
            currentBest = bestParameters['convergence']
            newInterval = interval['convergence']/100 * currentBest / pow(10, int(i/2)-1)
            newRange = list(np.arange(currentBest, currentBest-newInterval*3, -newInterval))
            newRange.reverse()
            newRange += list(np.arange(currentBest+newInterval, currentBest+newInterval*2, newInterval))
            bestParameters['convergence'] = newRange
        specs = ExecuteSweep( bestParameters, "convergence", "convergence", i < 2 )
       
    newBestParameters = findBestParameters(specs)

    newBestResult = ExecuteEvaluationRun(newBestParameters, xTrain, yTrain)
    print("new parameters: ", newBestResult['crossValidationMean'], " +/- ", newBestResult['crossValidationErrorBound'])

    if i % 2 == 0 and i > 2 and newBestResult['crossValidationMean'] < prevModelAccuracyUpper:
        break

    bestParameters = {
        'stepSize': newBestParameters['stepSize'], 
        'convergence': newBestParameters['convergence'], 
     }
    result = ExecuteEvaluationRun(bestParameters, xTrain, yTrain, 1)
    validationSetAccuracy.append( result['accuracy'] )
    validationSetAccuracyError.append( result['accuracyErrorBound'] )

# Plot Validation Accuracy
lastSweep = sweep[len(sweep)-1]
#Charting.PlotSeriesWithErrorBars([validationSetAccuracy], [validationSetAccuracyError], ["Accuracy"], sweep+[lastSweep+1], chartTitle="Validation Set Accuracy", xAxisTitle="Sweep #", yAxisTitle="Validation Set Accuracy", yBotLimit=0.8, outputDirectory=kOutputDirectory, fileName="7-ValidationSetAccuracy")
print("BestParameters: ", bestParameters)
print("Validation Accuracies: ", validationSetAccuracy)
print("Validation Accuracies (error): ", validationSetAccuracyError)