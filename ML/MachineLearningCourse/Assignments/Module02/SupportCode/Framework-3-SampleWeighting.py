import MachineLearningCourse.MLUtilities.Learners.DecisionTreeWeighted as DecisionTreeWeighted

# some sample tests that call into helper functions in the DecisionTree module. 
#   You may not have implemented the same way, so you might have to adapt these tests.

WeightedEntropyUnitTest = False
if WeightedEntropyUnitTest:
    y = [1, 1, 0, 1, 1, 0, 0, 0, 1, 0]
    
    print("Unweighted: ")
    print(DecisionTreeWeighted.Entropy(y, [ 1.0 for label in y ]))

    print("All 1s get 0 weight: ")
    print(DecisionTreeWeighted.Entropy(y, [ 0.0 if label == 1 else 1.0 for label in y ]))

    print("All 1s get .1 weight: ")
    print(DecisionTreeWeighted.Entropy(y, [ 0.1 if label == 1 else 1.0 for label in y ]))


WeightedSplitUnitTest = False
if WeightedSplitUnitTest:
    x = [[.1], [.2], [.3], [.4], [.5], [.6], [.7], [.8], [.9], [1.0]]
    y = [1, 1, 0, 1, 1, 0, 0, 0, 1, 0]
    
    print("Unweighted: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, [ 1.0 for label in y ], 0))

    print("All 1s get 0 weight: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, [ 0.0 if label == 1 else 1.0 for label in y ], 0))

    print("All 1s get .1 weight: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, [ 0.1 if label == 1 else 1.0 for label in y ], 0))


WeightTreeUnitTest = False
if WeightTreeUnitTest:
    xTrain = [[.1], [.2], [.3], [.4], [.5], [.6], [.7], [.8], [.9], [1.0]]
    yTrain = [1, 1, 0, 1, 1, 0, 0, 0, 1, 0]

    print("Unweighted:")
    model = DecisionTreeWeighted.DecisionTreeWeighted()
    model.fit(xTrain, yTrain, maxDepth = 1)

    model.visualize()

    print("Weighted 1s:")
    model = DecisionTreeWeighted.DecisionTreeWeighted()
    model.fit(xTrain, yTrain, weights=[ 10 if y == 1 else 0.1 for y in yTrain ], maxDepth = 1)

    model.visualize()

    print("Weighted 0s:")
    model = DecisionTreeWeighted.DecisionTreeWeighted()
    model.fit(xTrain, yTrain, weights=[ 1 if y == 0 else 0.1 for y in yTrain ], maxDepth = 1)

    model.visualize()


import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLProjectSupport.Adult.AdultDataset as AdultDataset
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
import MachineLearningCourse.Assignments.Module02.SupportCode.AdultFeaturize as AdultFeaturize
import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.MLUtilities.Learners.DecisionTree as DecisionTree
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

ROC = True
if ROC:
    ### UPDATE this path for your environment
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

    # ROC of best numeric vs. best non-numeric
    seriesFPRs = []
    seriesFNRs = []
    seriesLabels = []

    model = DecisionTree.DecisionTree()
    model.fit(xTrain, yTrain, maxDepth=15)

    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xValidate, yValidate)
    seriesFPRs.append(modelFPRs)
    seriesFNRs.append(modelFNRs)
    seriesLabels.append('best unweighted model w/ numeric features (depth=15)')

    model = DecisionTreeWeighted.DecisionTreeWeighted()
    model.fit(xTrain, yTrain, weights=[ 10 if x[0] < 45 else 1 for x in xTrain ], maxDepth=15)

    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xValidate, yValidate)
    seriesFPRs.append(modelFPRs)
    seriesFNRs.append(modelFNRs)
    seriesLabels.append('weighted model on age < 45 (depth=15)')

    Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC Comparison", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="3-WeightedDecistionTreeROC")


