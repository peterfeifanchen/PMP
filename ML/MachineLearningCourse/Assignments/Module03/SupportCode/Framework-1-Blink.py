kOutputDirectory = "MachineLearningCourse/Assignments/Module03/Graphs/visualize\\"

import MachineLearningCourse.MLProjectSupport.Blink.BlinkDataset as BlinkDataset

(xRaw, yRaw) = BlinkDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw)

print("Train is %d samples, %.4f percent opened." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
print("Validate is %d samples, %.4f percent opened." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
print("Test is %d samples %.4f percent opened" % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

import MachineLearningCourse.Assignments.Module03.SupportCode.BlinkFeaturize as BlinkFeaturize

import MachineLearningCourse.MLUtilities.Learners.BoostedTree as BoostedTree
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

tuneRounds = False
if tuneRounds:

    featurizer = BlinkFeaturize.BlinkFeaturize()

    #featurizer.CreateFeatureSet(xTrainRaw, yTrain, includeEdgeFeatures=True)
    #filename = "1-BlinkBoostedTreeEdgeFilterFeaturesOnly-Rounds" 
    featurizer.CreateFeatureSet(xTrainRaw, yTrain, includeEdgeFeatures=True, includeEdgeFeaturesMax=True)
    filename = "1-BlinkBoostedTreeEdgeFilterMaxAndAvg-Rounds"

    xTrain    = featurizer.Featurize(xTrainRaw)
    xValidate = featurizer.Featurize(xValidateRaw)
    xTest     = featurizer.Featurize(xTestRaw)

    bestModelBT = None
    kValues = [1, 10, 50, 100, 150]
    maxDepth = 1
    validationAccuracies = []
    validationAccuracyErrorBounds = []
    trainingAccuracies = []
    trainingAccuracyErrorBounds = []
    for kv in kValues:
        model = BoostedTree.BoostedTree()
        model.fit(xTrain, yTrain, maxDepth=maxDepth, k=kv)
        validationAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, model.predict(xValidate))
        lower, upper = ErrorBounds.GetAccuracyBounds(validationAccuracy, len(yValidate), .5)
        trainingAccuracy = EvaluateBinaryClassification.Accuracy(yTrain, model.predict(xTrain))
        lowerTrain, upperTrain = ErrorBounds.GetAccuracyBounds(trainingAccuracy, len(yTrain), .5)

        validationAccuracies.append(validationAccuracy)
        validationAccuracyErrorBounds.append(validationAccuracy-lower)
        trainingAccuracies.append(trainingAccuracy)
        trainingAccuracyErrorBounds.append(trainingAccuracy-lowerTrain)

        print("k: ", kv, " accuracy: ", lower, "-", upper)
        if bestModelBT is None:
            bestModelBT = (model, lower, upper, kv)
        elif lower > bestModelBT[2]:
            bestModelBT = (model, lower, upper, kv)

    print("boosted tree - k-rounds: ", bestModelBT[3], " accuracy: ", bestModelBT[1], "-", bestModelBT[2])
    Charting.PlotSeriesWithErrorBars([validationAccuracies, trainingAccuracies], [validationAccuracyErrorBounds, trainingAccuracyErrorBounds], ["BT-validation", "BT-training"], kValues, chartTitle="Boosted Decision Tree k-Round Search", xAxisTitle="Boosting Rounds", yAxisTitle="Accuracy", yBotLimit=0.5, outputDirectory=kOutputDirectory, fileName=filename)

tuneDepth = False
if tuneDepth:
    
    featurizer = BlinkFeaturize.BlinkFeaturize()

    #featurizer.CreateFeatureSet(xTrainRaw, yTrain, includeEdgeFeatures=True)
    #filename = "1-BlinkBoostedTreeEdgeFilterFeaturesOnly-MaxDepth" 
    featurizer.CreateFeatureSet(xTrainRaw, yTrain, includeEdgeFeatures=True, includeEdgeFeaturesMax=True)
    filename = "1-BlinkBoostedTreeEdgeFilterMaxAndAvg-MaxDepth"

    xTrain    = featurizer.Featurize(xTrainRaw)
    xValidate = featurizer.Featurize(xValidateRaw)
    xTest     = featurizer.Featurize(xTestRaw)

    bestModelBT = None
    kValues = 50
    maxDepth = [1,2,4]
    validationAccuracies = []
    validationAccuracyErrorBounds = []
    trainingAccuracies = []
    trainingAccuracyErrorBounds = []
    for depth in maxDepth:
        model = BoostedTree.BoostedTree()
        model.fit(xTrain, yTrain, maxDepth=depth, k=kValues)
        validationAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, model.predict(xValidate))
        lower, upper = ErrorBounds.GetAccuracyBounds(validationAccuracy, len(yValidate), .5)
        trainingAccuracy = EvaluateBinaryClassification.Accuracy(yTrain, model.predict(xTrain))
        lowerTrain, upperTrain = ErrorBounds.GetAccuracyBounds(trainingAccuracy, len(yTrain), .5)

        validationAccuracies.append(validationAccuracy)
        validationAccuracyErrorBounds.append(validationAccuracy-lower)
        trainingAccuracies.append(trainingAccuracy)
        trainingAccuracyErrorBounds.append(trainingAccuracy-lowerTrain)

        print("depth: ", depth, " accuracy: ", lower, "-", upper)
        if bestModelBT is None:
            bestModelBT = (model, lower, upper, depth)
        elif lower > bestModelBT[2]:
            bestModelBT = (model, lower, upper, depth)

    print("boosted tree - max-depth: ", bestModelBT[3], " accuracy: ", bestModelBT[1], "-", bestModelBT[2])
    Charting.PlotSeriesWithErrorBars([validationAccuracies, trainingAccuracies], [validationAccuracyErrorBounds, trainingAccuracyErrorBounds], ["BT-validation", "BT-training"], maxDepth, chartTitle="Boosted Decision Tree maxDepth Search", xAxisTitle="Max Depth", yAxisTitle="Accuracy", yBotLimit=0.5, outputDirectory=kOutputDirectory, fileName=filename)

# A helper function for calculating FN rate and FP rate across a range of thresholds
def TabulateModelPerformanceForROC(m, x, y):
    pointsToEvaluate = 100
    ts = [ x / float(pointsToEvaluate) for x in range(pointsToEvaluate + 1)]
    FPRs = []
    FNRs = []

    try:
        for threshold in ts:
            FPRs.append(EvaluateBinaryClassification.FalsePositiveRate(y, m.predict(x, classificationThreshold=threshold)))
            FNRs.append(EvaluateBinaryClassification.FalseNegativeRate(y, m.predict(x, classificationThreshold=threshold)))
    except NotImplementedError:
        raise UserWarning("The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

    return (FPRs, FNRs, ts)

ROC = True
if ROC:
    seriesFPRs = []
    seriesFNRs = []
    seriesLabels = []

    featurizer = BlinkFeaturize.BlinkFeaturize()

    featurizer.CreateFeatureSet(xTrainRaw, yTrain, includeEdgeFeatures=True)

    xTrain    = featurizer.Featurize(xTrainRaw)
    xValidate = featurizer.Featurize(xValidateRaw)
    xTest     = featurizer.Featurize(xTestRaw)

    model = BoostedTree.BoostedTree()
    model.fit(xTrain, yTrain, maxDepth=1, k=50) 
    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xTest, yTest)
    seriesFPRs.append(modelFPRs)
    seriesFNRs.append(modelFNRs)
    seriesLabels.append('best boosted tree (k-rounds=50, maxDepth=1) - Sobel Avg features')

    featurizer = BlinkFeaturize.BlinkFeaturize()

    featurizer.CreateFeatureSet(xTrainRaw, yTrain, includeEdgeFeatures=True, includeEdgeFeaturesMax=True)

    xTrain    = featurizer.Featurize(xTrainRaw)
    xValidate = featurizer.Featurize(xValidateRaw)
    xTest     = featurizer.Featurize(xTestRaw)

    model = BoostedTree.BoostedTree()
    model.fit(xTrain, yTrain, maxDepth=1, k=50) 
    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xTest, yTest)
    seriesFPRs.append(modelFPRs)
    seriesFNRs.append(modelFNRs)
    seriesLabels.append('best boosted tree (k-rounds=50, maxDepth=1) - Sobel Avg+Max features')

    Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC Comparison", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="1-SobelFeaturesROC")