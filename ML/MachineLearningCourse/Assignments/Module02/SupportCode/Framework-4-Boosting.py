kOutputDirectory = "MachineLearningCourse/Assignments/Module02/Graphs/visualize\\"
import MachineLearningCourse.MLUtilities.Data.Generators.SampleUniform2D as SampleUniform2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptCircle2D as ConceptCircle2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptSquare2D as ConceptSquare2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptLinear2D as ConceptLinear2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptCompound2D as ConceptCompound2D
import MachineLearningCourse.MLUtilities.Learners.BoostedTree as BoostedTree
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds

## remember this helper function
# Charting.PlotSeriesWithErrorBars([yValues], [errorBars], [series names], xValues, chartTitle=", xAxisTitle="", yAxisTitle="", yBotLimit=0.5, outputDirectory=kOutputDirectory, fileName="")

## generat some synthetic data do help debug your learning code

generator = SampleUniform2D.SampleUniform2D(seed=100)
#conceptSquare = ConceptSquare2D.ConceptSquare2D(width=.2)
conceptLinear = ConceptLinear2D.ConceptLinear2D(bias=0.05, weights=[0.3, -0.3])
conceptCircle = ConceptCircle2D.ConceptCircle2D(radius=.3)

concept = ConceptCompound2D.ConceptCompound2D(concepts = [ conceptLinear, conceptCircle ])

xTest = generator.generate(1000)
yTest = concept.predict(xTest)

xTrain = generator.generate(1000)
yTrain = concept.predict(xTrain)


import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
import MachineLearningCourse.MLUtilities.Visualizations.Visualize2D as Visualize2D

## this code outputs the true concept.
visualize = Visualize2D.Visualize2D(kOutputDirectory, "4-Generated Concept")
visualize.Plot2DDataAndBinaryConcept(xTest,yTest,concept)
visualize.Save()

bestModel = None
kValues = [1, 10, 25, 50, 100]
maxDepth = 1
accuracies = []
errorBarsAccuracy = []
for kv in kValues:
    model = BoostedTree.BoostedTree()
    model.fit(xTrain, yTrain, maxDepth=maxDepth, k=kv)
    accuracy = EvaluateBinaryClassification.Accuracy(yTest, model.predict(xTest))
    lower, upper = ErrorBounds.GetAccuracyBounds(accuracy, len(yTest), .5)
    print(kv, ": ", accuracy)
    accuracies.append(accuracy)
    errorBarsAccuracy.append(accuracy-lower)
    if bestModel is None:
        bestModel = (model, upper)
    elif lower > bestModel[1]:
        bestModel = (model, upper)

Charting.PlotSeriesWithErrorBars([accuracies], [errorBarsAccuracy], ["k-round tuning accuracy"], kValues, chartTitle="Line/Circle Concept Accuracy", xAxisTitle="Boosting Rounds", yAxisTitle="Test Accuracy", yBotLimit=0.5, outputDirectory=kOutputDirectory, fileName="4-BoostingTreeRoundTuning")

## you can use this to visualize what your model is learning.
accuracy = EvaluateBinaryClassification.Accuracy(yTest, bestModel[0].predict(xTest))
lower, upper = ErrorBounds.GetAccuracyBounds(accuracy, len(yTest), .95)
print("accuracy: ", lower, "-", upper)
visualize = Visualize2D.Visualize2D(kOutputDirectory, "4-My Boosted Tree")
visualize.PlotBinaryConcept(model)

# Or you can use it to visualize individual models that you learened, e.g.:
# visualize.PlotBinaryConcept(model->modelLearnedInRound[2])
    
## you might like to see the training or test data too, so you might prefer this to simply calling 'PlotBinaryConcept'
#visualize.Plot2DDataAndBinaryConcept(xTrain,yTrain,model)

# And remember to save
visualize.Save()
