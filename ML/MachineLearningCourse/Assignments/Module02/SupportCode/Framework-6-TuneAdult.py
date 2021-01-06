import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLProjectSupport.Adult.AdultDataset as AdultDataset
import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.Assignments.Module02.SupportCode.AdultFeaturize as AdultFeaturize
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

kOutputDirectory = "MachineLearningCourse/Assignments/Module02/Graphs/visualize\\"
kDataPath = "MachineLearningCourse/MLProjectSupport/Adult/dataset/adult.data"

(xRaw, yRaw) = AdultDataset.LoadRawData(kDataPath)

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw)

print("Train is %d samples, %.4f percent >50K." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
print("Validate is %d samples, %.4f percent >50K." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
print("Test is %d samples %.4f percent >50K." % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

featurizer = AdultFeaturize.AdultFeaturize()
featurizer.CreateFeatureSet(xTrainRaw, yTrain, useCategoricalFeatures = True, useNumericFeatures = True)

xTrain    = featurizer.Featurize(xTrainRaw)
xValidate = featurizer.Featurize(xValidateRaw)
xTest     = featurizer.Featurize(xTestRaw)

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

bestModelLG = None
runLogisticRegression = True
if runLogisticRegression:
    stepSizes = [0.1, 1, 10, 100]
    validationAccuracies = []
    validationAccuracyErrorBounds = []
    trainingAccuracies = []
    trainingAccuracyErrorBounds = []
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    for stepSize in stepSizes:
        model = LogisticRegression.LogisticRegression()
        model.fit(xTrain, yTrain, convergence=0.0001, stepSize=stepSize)
        validationAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, model.predict(xValidate))
        lower, upper = ErrorBounds.GetAccuracyBounds(validationAccuracy, len(yValidate), .5)
        trainingAccuracy = EvaluateBinaryClassification.Accuracy(yTrain, model.predict(xTrain))
        lowerTrain, upperTrain = ErrorBounds.GetAccuracyBounds(trainingAccuracy, len(yTrain), .5)

        validationAccuracies.append(validationAccuracy)
        validationAccuracyErrorBounds.append(validationAccuracy-lower)
        trainingAccuracies.append(trainingAccuracy)
        trainingAccuracyErrorBounds.append(trainingAccuracy-lowerTrain)

        print("stepSize: ", stepSize, " accuracy: ", lower, "-", upper)
        if bestModelLG is None:
            bestModelLG = (model, lower, upper, stepSize)
        elif lower > bestModelLG[2]:
            bestModelLG = (model, lower, upper, stepSize)
    
    Charting.PlotSeriesWithErrorBars([validationAccuracies, trainingAccuracies], [validationAccuracyErrorBounds, trainingAccuracyErrorBounds], ["LR-validation", "LR-training"], stepSizes, chartTitle="Logistic Regression Step Size Search", xAxisTitle="Step Size", yAxisTitle="Accuracy", yBotLimit=0.5, outputDirectory=kOutputDirectory, fileName="5-LogisticRegressionStepSize")

bestModelDT = None
runDecisionTree = True
if runDecisionTree:
    import MachineLearningCourse.MLUtilities.Learners.DecisionTree as DecisionTree
    maxDepths = [1, 10, 50, 100, 500]
    validationAccuracies = []
    validationAccuracyErrorBounds = []
    trainingAccuracies = []
    trainingAccuracyErrorBounds = []
    for maxDepth in maxDepths:
        model = DecisionTree.DecisionTree()
        model.fit(xTrain, yTrain, maxDepth=maxDepth)
        validationAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, model.predict(xValidate))
        lower, upper = ErrorBounds.GetAccuracyBounds(validationAccuracy, len(yValidate), .5)
        trainingAccuracy = EvaluateBinaryClassification.Accuracy(yTrain, model.predict(xTrain))
        lowerTrain, upperTrain = ErrorBounds.GetAccuracyBounds(trainingAccuracy, len(yTrain), .5)

        validationAccuracies.append(validationAccuracy)
        validationAccuracyErrorBounds.append(validationAccuracy-lower)
        trainingAccuracies.append(trainingAccuracy)
        trainingAccuracyErrorBounds.append(trainingAccuracy-lowerTrain)

        print("maxDepth: ", maxDepth, " accuracy: ", lower, "-", upper)
        if bestModelDT is None:
            bestModelDT = (model, lower, upper, maxDepth)
        elif lower > bestModelDT[2]:
            bestModelDT = (model, lower, upper, maxDepth)
    
    Charting.PlotSeriesWithErrorBars([validationAccuracies, trainingAccuracies], [validationAccuracyErrorBounds, trainingAccuracyErrorBounds], ["DT-validation", "DT-training"], maxDepths, chartTitle="Decision Tree Max Depths Search", xAxisTitle="Max Depth", yAxisTitle="Accuracy", yBotLimit=0.5, outputDirectory=kOutputDirectory, fileName="5-DecisionTreeMaxDepth")

bestModelBT = None
runBoostedTree = True
if runBoostedTree:
    import MachineLearningCourse.MLUtilities.Learners.BoostedTree as BoostedTree
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

    Charting.PlotSeriesWithErrorBars([validationAccuracies, trainingAccuracies], [validationAccuracyErrorBounds, trainingAccuracyErrorBounds], ["BT-validation", "BT-training"], kValues, chartTitle="Boosted Decision Tree k-Round Search", xAxisTitle="Boosting Rounds", yAxisTitle="Accuracy", yBotLimit=0.5, outputDirectory=kOutputDirectory, fileName="5-BoostingTreeRound")

ROC = True
if ROC:
    # ROC of best numeric vs. best non-numeric
    seriesFPRs = []
    seriesFNRs = []
    seriesLabels = []

    # Best Logistic Regression Model
    print("logistic regression - step size: ", bestModelLG[3], " accuracy: ", bestModelLG[1], "-", bestModelLG[2])
    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(bestModelLG[0], xTest, yTest)
    seriesFPRs.append(modelFPRs)
    seriesFNRs.append(modelFNRs)
    seriesLabels.append('best logistic regression (step size={})'.format(bestModelLG[3]))

    # Best Decision Tree Model
    print("decision tree - max depth: ", bestModelDT[3], " accuracy: ", bestModelDT[1], "-", bestModelDT[2])
    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(bestModelDT[0], xTest, yTest)
    seriesFPRs.append(modelFPRs)
    seriesFNRs.append(modelFNRs)
    seriesLabels.append('best decision tree (max depth={})'.format(bestModelDT[3]))

    # Best Decision Tree Model
    print("boosted tree - k-rounds: ", bestModelBT[3], " accuracy: ", bestModelBT[1], "-", bestModelBT[2])
    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(bestModelBT[0], xTest, yTest)
    seriesFPRs.append(modelFPRs)
    seriesFNRs.append(modelFNRs)
    seriesLabels.append('best boosted tree (k-rounds={})'.format(bestModelBT[3]))

    Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC Comparison", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="6-AllMethodsROC")
