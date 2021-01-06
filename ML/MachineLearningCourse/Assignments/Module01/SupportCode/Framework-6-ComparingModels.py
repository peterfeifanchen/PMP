import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

kDataPath = "MachineLearningCourse/MLProjectSupport/SMSSpam/dataset/SMSSpamCollection"

(xRaw, yRaw) = SMSSpamDataset.LoadRawData(kDataPath)

import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Learners.MostCommonClassModel as MostCommonClassModel
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

doModelEvaluation = False
if doModelEvaluation:
    ######
    ### Build a model and evaluate on validation data
    stepSize = 1.0
    convergence = 0.001

    featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords = 25)

    xTrain      = featurizer.Featurize(xTrainRaw)
    xValidate   = featurizer.Featurize(xValidateRaw)
    xTest       = featurizer.Featurize(xTestRaw)

    frequentModel = LogisticRegression.LogisticRegression()
    frequentModel.fit(xTrain, yTrain, convergence=convergence, stepSize=stepSize, verbose=True)

    ######
    ### Use equation 5.1 from Mitchell to bound the validation set error and the true error
    import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds

    print("Logistic regression with 25 features by mutual information:")
    validationSetAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, frequentModel.predict(xValidate))
    print("Validation set accuracy: %.4f." % (validationSetAccuracy))
    for confidence in [.5, .8, .9, .95, .99]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(validationSetAccuracy, len(xValidate), confidence)    
        print(" %.2f%% accuracy bound: %.4f - %.4f" % (confidence, lowerBound, upperBound))

    ### Compare to most common class model here...
    mostCommonModel = MostCommonClassModel.MostCommonClassModel()
    mostCommonModel.fit(xTrain, yTrain)

    print("MostCommon regression model:")
    validationSetAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, mostCommonModel.predict(xValidate))
    print("Validation set accuracy: %.4f." % (validationSetAccuracy))
    for confidence in [.5, .8]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(validationSetAccuracy, len(xValidate), confidence)    
        print(" %.2f%% accuracy bound: %.4f - %.4f" % (confidence, lowerBound, upperBound))

# Set this to true when you've completed the previous steps and are ready to move on...
doCrossValidation = True
if doCrossValidation:
    import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation    
    numberOfFolds = 5

    import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
    ######
    ### Build a model and evaluate on validation data
    stepSize = 0.1
    convergence = 0.0001

    featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords = 25)

    xTrain      = featurizer.Featurize(xTrainRaw)
    xValidate   = featurizer.Featurize(xValidateRaw)
    xTest       = featurizer.Featurize(xTestRaw)

    validationSetAccuracyMI = []
    validationSetAccuracyCommon = []
    for i in range(numberOfFolds):
        xTrain, yTrain, xEvaluate, yEvaluate = CrossValidation.CrossValidation(xTrain, yTrain, numberOfFolds, i)

        frequentModel = LogisticRegression.LogisticRegression()
        frequentModel.fit(xTrain, yTrain, convergence=convergence, stepSize=stepSize, verbose=True)
        
        mostCommonModel = MostCommonClassModel.MostCommonClassModel()
        mostCommonModel.fit(xTrain, yTrain)
 
        validationSetAccuracyMI.append(EvaluateBinaryClassification.Accuracy(yEvaluate, frequentModel.predict(xEvaluate)))
        validationSetAccuracyCommon.append(EvaluateBinaryClassification.Accuracy(yEvaluate, mostCommonModel.predict(xEvaluate)))

    import numpy as np
    dhat = np.mean(validationSetAccuracyMI) - np.mean(validationSetAccuracyCommon)

    # To find our confidence level that dhat is how much better MI is to Common, we find its t-test deviation from dhat == 0
    # and what the one-sided confidence level for that would be.
    # 1) Find the standard deviation of d
    validationSetAccuracyCommon = np.array(validationSetAccuracyCommon)
    validationSetAccuracyMI = np.array(validationSetAccuracyMI)
    diff = validationSetAccuracyMI - validationSetAccuracyCommon
    error = diff-diff.mean()

    print("MostCommon: {}".format(validationSetAccuracyCommon))
    print("MI: {}".format(validationSetAccuracyMI))
    print("diff: {}".format(diff))
    print("error: {}".format(error))
    
    # alternatively, I think you can do diff.std(ddof=1)/np.sqrt(len(diff)), but the result is not equivalent
    # not sure why these two different approaches yield different values for t-statistic and why one takes
    # into account the length of diff, but not the other. For example diff.std(ddof=0) != err.transpose().dot(error)
    # when that should.
    var = 1 / (numberOfFolds*(numberOfFolds-1)) * error.transpose().dot(error)
    tdev = np.sqrt(var)
    print("tdev: {}".format(tdev))
    # 2) Find the confidence level (one-sided) that the observed difference as a proxy for confidence that MI model is better
    from scipy import stats
    confidence = stats.t.cdf(diff.mean()/tdev, numberOfFolds-1)
    print("Confidence that MI model is better than MostCommon given observed {:.4f} is {:.2f}%".format(diff.mean(), confidence*100))

