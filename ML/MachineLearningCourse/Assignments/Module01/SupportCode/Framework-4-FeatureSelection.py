kOutputDirectory = "MachineLearningCourse/Assignments/Module01/Graphs/visualize"

import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

kDataPath = "MachineLearningCourse/MLProjectSupport/SMSSpam/dataset/SMSSpamCollection"

(xRaw, yRaw) = SMSSpamDataset.LoadRawData(kDataPath)

import MachineLearningCourse.MLUtilities.Data.Sample as Sample
(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize

findTop10Words = False
if findTop10Words:
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)

    print("Top 10 words by frequency: ", featurizer.FindMostFrequentWords(xTrainRaw, 10))
    print("Top 10 words by mutual information: ", featurizer.FindTopWordsByMutualInformation(xTrainRaw, yTrain, 10))

# set to true when your implementation of the 'FindWords' part of the assignment is working
doModeling = False
if doModeling:
    # Now get into model training
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification

    # The hyperparameters to use with logistic regression for this assignment
    stepSize = 1.0
    convergence = 0.0001

    # Remember to create a new featurizer object/vocabulary for each part of the assignment
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords = 10)
    print( featurizer.vocabulary )
    # Remember to reprocess the raw data whenever you change the featurizer
    xTrain      = featurizer.Featurize(xTrainRaw)
    xValidate   = featurizer.Featurize(xValidateRaw)
    xTest       = featurizer.Featurize(xTestRaw)

    ## Good luck!
    print("Learning the logistic regression model:")
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    logisticRegressionModel = LogisticRegression.LogisticRegression()

    logisticRegressionModel.fit(xTrain, yTrain, stepSize=stepSize, convergence=convergence)
    
    import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
    
    print ("\nLogistic regression model:")
    logisticRegressionModel.visualize()
    #EvaluateBinaryClassification.ExecuteAll(yTrain, logisticRegressionModel.predict(xTrain, classificationThreshold=0.5))
    EvaluateBinaryClassification.ExecuteAll(yValidate, logisticRegressionModel.predict(xValidate, classificationThreshold=0.5))

# do hyperparameter sweeping of feature engineering
doSearch = True
if doSearch:
    trainLosses = []
    validationLosses = []
    lossXLabels = [ 1, 10, 20, 30, 40, 50 ]
    for freq in lossXLabels:
        # Now get into model training
        import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression

        # Remember to create a new featurizer object/vocabulary for each part of the assignment
        featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
        featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords = freq)

        # Remember to reprocess the raw data whenever you change the featurizer
        xTrain      = featurizer.Featurize(xTrainRaw)
        xValidate   = featurizer.Featurize(xValidateRaw)
        xTest       = featurizer.Featurize(xTestRaw)

        import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
        logisticRegressionModel = LogisticRegression.LogisticRegression()

        logisticRegressionModel.fit(xTrain, yTrain, stepSize=1.0, convergence=0.001)
        trainLosses.append(logisticRegressionModel.loss(xTrain, yTrain))
        validationLosses.append(logisticRegressionModel.loss(xValidate, yValidate))  

    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
    # trainLosses, validationLosses, and lossXLabels are parallel arrays with the losses you want to plot at the specified x coordinates
    Charting.PlotSeries([trainLosses, validationLosses], ['Train', 'Validate'], lossXLabels, chartTitle="Num Frequent Words Logistic Regression", xAxisTitle="Num Frequent Words", yAxisTitle="Avg. Loss", outputDirectory=kOutputDirectory, fileName="4-Logistic Regression Num Frequent Words Train vs Validate loss")

    trainLosses = []
    validationLosses = []
    lossXLabels = [ 1, 10, 20, 30, 40, 50 ]
    for freq in lossXLabels:
        # Now get into model training
        import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression

        # Remember to create a new featurizer object/vocabulary for each part of the assignment
        featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
        featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords = freq)

        # Remember to reprocess the raw data whenever you change the featurizer
        xTrain      = featurizer.Featurize(xTrainRaw)
        xValidate   = featurizer.Featurize(xValidateRaw)
        xTest       = featurizer.Featurize(xTestRaw)

        import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
        logisticRegressionModel = LogisticRegression.LogisticRegression()

        logisticRegressionModel.fit(xTrain, yTrain, stepSize=1.0, convergence=0.001)
        trainLosses.append(logisticRegressionModel.loss(xTrain, yTrain))
        validationLosses.append(logisticRegressionModel.loss(xValidate, yValidate))  

    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
    # trainLosses, validationLosses, and lossXLabels are parallel arrays with the losses you want to plot at the specified x coordinates
    Charting.PlotSeries([trainLosses, validationLosses], ['Train', 'Validate'], lossXLabels, chartTitle="Num MI Words Logistic Regression", xAxisTitle="Num MI Words", yAxisTitle="Avg. Loss", outputDirectory=kOutputDirectory, fileName="4-Logistic Regression Num MI Words Train vs Validate loss")