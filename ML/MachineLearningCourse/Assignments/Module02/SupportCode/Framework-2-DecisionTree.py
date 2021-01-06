import MachineLearningCourse.MLUtilities.Learners.DecisionTree as DecisionTree
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
kOutputDirectory = "MachineLearningCourse/Assignments/Module02/Graphs/visualize\\"

doExamples = False
if doExamples:
    ### Some simple test cases to get you started. You'll have to work out the correct answers yourself.

    print("test simple split")
    x=[[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]]
    y = [1,1,1,1,1,1,0,0,0,1]
    model = DecisionTree.DecisionTree()
    model.fit(x, y)

    model.visualize()

    print("test no split")
    x=[[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]]
    y = [1,1,0,1,1,1,1,1,0,1]
    model = DecisionTree.DecisionTree()
    model.fit(x, y)

    model.visualize()

    print("test numeric feature sort")
    x = [[1,3], [2,2], [19,7], [4,1]]
    y = [1,1,0,0]

    model = DecisionTree.DecisionTree()
    model.fit(x, y)

    model.visualize()

    print("Bigger tree")
    x = [[10, 7], [9,8], [101, 71], [44, 44], [19, 111], [1, 2], [1,3], [2,2], [19,7], [4,1]]
    y = [1,1,0,0,1,1,0,0,1,1]

    model = DecisionTree.DecisionTree()
    model.fit(x, y)

    model.visualize()

## These might help you debug...
doVisualize = False
if doVisualize:
    
    import MachineLearningCourse.MLUtilities.Data.Generators.SampleUniform2D as SampleUniform2D
    import MachineLearningCourse.MLUtilities.Data.Generators.ConceptCircle2D as ConceptCircle2D
    import MachineLearningCourse.MLUtilities.Data.Generators.ConceptSquare2D as ConceptSquare2D
    import MachineLearningCourse.MLUtilities.Data.Generators.ConceptLinear2D as ConceptLinear2D

    generator = SampleUniform2D.SampleUniform2D(seed=100)
    #concept = ConceptSquare2D.ConceptSquare2D(width=.2)
    concept = ConceptLinear2D.ConceptLinear2D(bias=0.05, weights=[0.2, -0.2])
    #concept = ConceptCircle2D.ConceptCircle2D(radius=.3)

    x = generator.generate(100)
    y = concept.predict(x)

    import MachineLearningCourse.MLUtilities.Visualizations.Visualize2D as Visualize2D

    visualize = Visualize2D.Visualize2D(kOutputDirectory, "Generated Concept")

    visualize.Plot2DDataAndBinaryConcept(x,y,concept)
    visualize.Save()

    print("Decision Tree on Generated Concept")
    model = DecisionTree.DecisionTree()
    model.fit(x, y, maxDepth = 2)
    model.visualize()

    visualize = Visualize2D.Visualize2D(kOutputDirectory, "DecisionTree on Generated Concept")

    visualize.Plot2DDataAndBinaryConcept(x,y,model)
    visualize.Save()

doModel = False
if doModel:
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

    xTrain    = featurizer.Featurize(xTrainRaw)
    xValidate = featurizer.Featurize(xValidateRaw)
    xTest     = featurizer.Featurize(xTestRaw)

    featurizerNumeric = AdultFeaturize.AdultFeaturize()
    featurizerNumeric.CreateFeatureSet(xTrainRaw, yTrain, useCategoricalFeatures = True, useNumericFeatures = True)

    xTrainNumeric    = featurizerNumeric.Featurize(xTrainRaw)
    xValidateNumeric = featurizerNumeric.Featurize(xValidateRaw)
    xTestNumeric     = featurizerNumeric.Featurize(xTestRaw)

    ############################
    import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
    import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
    import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation
    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
    import time
    import numpy as np
    
    model = DecisionTree.DecisionTree()
    model.fit(xTrainNumeric,yTrain)
    validationSetAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, model.predict(xValidateNumeric))
    print("numericvalidationSetAccuracy: ", validationSetAccuracy)
    #model.visualize()

    model = DecisionTree.DecisionTree()
    model.fit(xTrain,  yTrain)
    validationSetAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, model.predict(xValidate)) 
    print("validationSetAccuracy: ", validationSetAccuracy)
    #model.visualize()

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

doOptimize = False
useNumeric = True
folds = 2
if doOptimize:
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
    featurizer.CreateFeatureSet(xTrainRaw, yTrain, useCategoricalFeatures = True, useNumericFeatures = useNumeric)

    xTrain    = featurizer.Featurize(xTrainRaw)
    xValidate = featurizer.Featurize(xValidateRaw)
    xTest     = featurizer.Featurize(xTestRaw)

    ############################
    import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
    import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation
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

                model = DecisionTree.DecisionTree()
                model.fit(xTrainI,yTrainI, maxDepth=runSpecification["maxDepth"])

                crossValidationAccuracy.append(EvaluateBinaryClassification.Accuracy(yEvaluateI, model.predict(xEvaluateI)))
    
            mean = np.mean(crossValidationAccuracy)
            runSpecification['crossValidationMean'] = mean
            lower, _ = ErrorBounds.GetAccuracyBounds(np.mean(crossValidationAccuracy), len(yEvaluateI), .95)
            runSpecification['crossValidationErrorBound'] = mean - lower
        
    
        if numberOfFolds == 1:
            model = DecisionTree.DecisionTree()
            model.fit(xTrain,yTrain, maxDepth=runSpecification["maxDepth"])
            validationSetAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, model.predict(xValidate)) 
    
            runSpecification['accuracy'] = validationSetAccuracy
            lower, _ = ErrorBounds.GetAccuracyBounds(validationSetAccuracy, len(yValidate), .95)
            runSpecification['accuracyErrorBound'] = validationSetAccuracy - lower
            runSpecification['crossValidationMean'] = validationSetAccuracy
            runSpecification['crossValidationErrorBound'] = validationSetAccuracy - lower

        endTime = time.time()
        runSpecification['runtime'] = endTime - startTime
    
        return runSpecification

    def ExecuteSweep( sweepSpec, optSpec, plotString, plot=False ):
        evaluationRunSpecifications = []
        for sweepValue in sweepSpec[optSpec]:
            runSpecification = {}
            runSpecification['optimizing'] = optSpec
            runSpecification['maxDepth'] = sweepSpec['maxDepth']
            runSpecification[optSpec] = sweepValue 
            evaluationRunSpecifications.append(runSpecification)

        ## if you want to run in parallel you need to install joblib as described in the lecture notes and adjust the comments on the next three lines...
        from joblib import Parallel, delayed
        evaluations = Parallel(n_jobs=4)(delayed(ExecuteEvaluationRun)(runSpec, xTrain, yTrain, folds) for runSpec in evaluationRunSpecifications)

        if plot:
            xValues = sweepSpec[optSpec]
            series1 = [ runSpec['crossValidationMean'] for runSpec in evaluations ]
            errorBarsForSeries1 = [ runSpec['crossValidationErrorBound'] for runSpec in evaluations]
            if folds > 1:
                chartName = "Cross Validation Accuracy"
                fileName = "CrossValidationAccuracy"
            else:
                chartName = "Validation Accuracy"
                fileName = "ValidationAccuracy"
            if useNumeric:
                Charting.PlotSeriesWithErrorBars([series1], [errorBarsForSeries1], [chartName], xValues, chartTitle=plotString + " " + chartName + " (w/ Numeric Features)", xAxisTitle=plotString + " Values", yAxisTitle=chartName, yBotLimit=0.7, outputDirectory=kOutputDirectory, fileName="2-"+plotString+fileName+"Numeric")
            else:
                Charting.PlotSeriesWithErrorBars([series1], [errorBarsForSeries1], [chartName], xValues, chartTitle=plotString + " " + chartName, xAxisTitle=plotString + " Values", yAxisTitle=chartName, yBotLimit=0.7, outputDirectory=kOutputDirectory, fileName="2-"+plotString+fileName)
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
            'maxDepth': best['maxDepth'],
        }

    bestParameters = {}
    bestParameters['maxDepth'] = [5, 10, 15, 30, 45, 60, 75, len(xTrain[0])]
    bestParameters['numeric'] = False
    specs = ExecuteSweep( bestParameters, "maxDepth", "MaxDepth", True )
    newBestParameters = findBestParameters(specs)
    print("best parameters: ", newBestParameters)

    result = ExecuteEvaluationRun(newBestParameters, xTrain, yTrain, 1)
    print("best parameters: ", result['accuracy'], " +/- ", result['accuracyErrorBound'])


ROC = True
BestDepthNumeric = 10
BestDepth = 60
if ROC:
    import MachineLearningCourse.MLProjectSupport.Adult.AdultDataset as AdultDataset
    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

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

    xTrain    = featurizer.Featurize(xTrainRaw)
    xValidate = featurizer.Featurize(xValidateRaw)
    xTest     = featurizer.Featurize(xTestRaw)

    featurizerNumeric = AdultFeaturize.AdultFeaturize()
    featurizerNumeric.CreateFeatureSet(xTrainRaw, yTrain, useCategoricalFeatures = True, useNumericFeatures = True)

    xTrainNumeric    = featurizerNumeric.Featurize(xTrainRaw)
    xValidateNumeric = featurizerNumeric.Featurize(xValidateRaw)
    xTestNumeric     = featurizerNumeric.Featurize(xTestRaw)

    # ROC of best numeric vs. best non-numeric
    seriesFPRs = []
    seriesFNRs = []
    seriesLabels = []

    model = DecisionTree.DecisionTree()
    model.fit(xTrainNumeric, yTrain, maxDepth=BestDepthNumeric)

    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xTestNumeric, yTest)
    seriesFPRs.append(modelFPRs)
    seriesFNRs.append(modelFNRs)
    seriesLabels.append('best numeric model')

    model = DecisionTree.DecisionTree()
    model.fit(xTrain, yTrain, maxDepth=BestDepth)

    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xTest, yTest)
    seriesFPRs.append(modelFPRs)
    seriesFNRs.append(modelFNRs)
    seriesLabels.append('best non-numeric model')

    Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC Comparison Features", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="2-DecistionTreeROC")