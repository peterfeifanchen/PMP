
import gym
import numpy as np

import random
import MachineLearningCourse.MLUtilities.Reinforcement.QLearning as QLearning
import MachineLearningCourse.Assignments.Module04.SupportCode.GymSupport as GymSupport
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

kOutputDirectory = "MachineLearningCourse/Assignments/Module04/Graphs/"
env = gym.make('CartPole-v0')

## Hyperparameters to tune:
discountRateRange = [ 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999 ]          # Controls the discount rate for future rewards -- this is gamma from 13.10
actionProbabilityBaseRange = [1.1, 1.2, 1.3, 1.4, 1.5]   # This is k from the P(a_i|s) expression from section 13.3.5 and influences how random exploration is
randomActionRateRange = [0.01, 0.05, 0.1, 0.15, 0.2]       # Percent of time the next action selected by GetAction is totally random
learningRateScaleRange = [0.1, 0.05, 0.01, 0.005, 0.001]     # Should be multiplied by visits_n from 13.11.
binsPerDimensionRange = [5, 6, 7, 8, 9, 10]

def trainModel( discountRate, actionProbabilityBase, randomActionRate, learningRateScale, binsPerDimension, trainingIterations=30000, numEvalTrials=20):
    continuousToDiscrete = GymSupport.ContinuousToDiscrete(binsPerDimension, [ -4.8000002e+00, -4, -4.1887903e-01, -4 ], [ 4.8000002e+00, 4, 4.1887903e-01, 4 ])
    qlearner = QLearning.QLearning(stateSpaceShape=continuousToDiscrete.StateSpaceShape(), numActions=env.action_space.n, discountRate=discountRate)

    # Learn the policy
    for trialNumber in range(trainingIterations):
        observation = env.reset()
        totalReward = 0
        for i in range(300):
            #env.render()

            currentState = continuousToDiscrete.Convert(observation)
            action = qlearner.GetAction(currentState, learningMode=True, randomActionRate=randomActionRate, actionProbabilityBase=actionProbabilityBase)

            oldState = continuousToDiscrete.Convert(observation)
            observation, reward, isDone, info = env.step(action)
            newState = continuousToDiscrete.Convert(observation)

            qlearner.ObserveAction(oldState, action, newState, reward, learningRateScale=learningRateScale)
            totalReward += reward
            if isDone:
                if(trialNumber%1000) == 0:
                    print("Training", "run", trialNumber, "steps", i, "reward", totalReward)
                break

    # Evaluate the policy
    totalRewards = []
    for runNumber in range(numEvalTrials):
        observation = env.reset()
        totalReward = 0
        reward = 0
        for i in range(300):
            #renderDone = env.render()

            currentState = continuousToDiscrete.Convert(observation)
            observation, reward, isDone, info = env.step(qlearner.GetAction(currentState, learningMode=False))

            totalReward += reward

            if isDone:
                #renderDone = env.render()
                print("evaluation: run ", runNumber, "steps", i, "reward", totalReward)
                totalRewards.append(totalReward)
                break
    
    return totalRewards

runDiscountRateRange = False
if runDiscountRateRange:
    accResults = []
    accError = []
    for i in discountRateRange:
        results = trainModel( discountRate=i, actionProbabilityBase=1.25, randomActionRate=0.1, learningRateScale=0.01, binsPerDimension=8)
        acc = np.mean(results)
        err = np.std(results)
        accResults.append(acc)
        accError.append(err)
    
    Charting.PlotSeriesWithErrorBars([accResults], [accError], ["CartPole Score (Mean) 20 trials"], discountRateRange, useMarkers=False, chartTitle="CartPole Discount Rate Effect on Learning", xAxisTitle="Discount Rate", yAxisTitle="Score", yBotLimit=20, outputDirectory=kOutputDirectory, fileName="1-DiscountRate")

runActionProbability = False
if runActionProbability:
    accResults = []
    accError = []
    for i in actionProbabilityBaseRange:
        results = trainModel( discountRate=0.75, actionProbabilityBase=i, randomActionRate=0.1, learningRateScale=0.01, binsPerDimension=8)
        acc = np.mean(results)
        err = np.std(results)
        accResults.append(acc)
        accError.append(err)
    
    Charting.PlotSeriesWithErrorBars([accResults], [accError], ["CartPole Score (Mean) 20 trials"], actionProbabilityBaseRange, useMarkers=False, chartTitle="CartPole Action Probability Base Effect on Learning", xAxisTitle="Action Probability Base", yAxisTitle="Score", yBotLimit=20, outputDirectory=kOutputDirectory, fileName="1-ActionProbabilityBase")

runRandomAction = False
if runRandomAction:
    accResults = []
    accError = []
    for i in randomActionRateRange:
        results = trainModel( discountRate=0.75, actionProbabilityBase=1.2, randomActionRate=i, learningRateScale=0.01, binsPerDimension=8)
        acc = np.mean(results)
        err = np.std(results)
        accResults.append(acc)
        accError.append(err)
    
    Charting.PlotSeriesWithErrorBars([accResults], [accError], ["CartPole Score (Mean) 20 trials"], randomActionRateRange, useMarkers=False, chartTitle="CartPole Random Action Probability Effect on Learning", xAxisTitle="Random Action Probability", yAxisTitle="Score", yBotLimit=20, outputDirectory=kOutputDirectory, fileName="1-RandomActionProbability")

runLearningRateScale = False
if runLearningRateScale:
    accResults = []
    accError = []
    for i in learningRateScaleRange:
        results = trainModel( discountRate=0.75, actionProbabilityBase=1.2, randomActionRate=0.01, learningRateScale=i, binsPerDimension=8)
        acc = np.mean(results)
        err = np.std(results)
        accResults.append(acc)
        accError.append(err)
    
    Charting.PlotSeriesWithErrorBars([accResults], [accError], ["CartPole Score (Mean) 20 trials"], learningRateScaleRange, useMarkers=False, chartTitle="CartPole Learning Rate Scale Effect on Learning", xAxisTitle="Learning Rate Scale", yAxisTitle="Score", yBotLimit=20, outputDirectory=kOutputDirectory, fileName="1-LearningRateScale")

runBinsPerDim = False
if runBinsPerDim:
    accResults = []
    accError = []
    for i in binsPerDimensionRange:
        results = trainModel( discountRate=0.75, actionProbabilityBase=1.2, randomActionRate=0.01, learningRateScale=0.05, binsPerDimension=i)
        acc = np.mean(results)
        err = np.std(results)
        accResults.append(acc)
        accError.append(err)
    
    Charting.PlotSeriesWithErrorBars([accResults], [accError], ["CartPole Score (Mean) 20 trials"], binsPerDimensionRange, useMarkers=False, chartTitle="CartPole Bin State Quantization Effect on Learning", xAxisTitle="# of Bins", yAxisTitle="Score", yBotLimit=20, outputDirectory=kOutputDirectory, fileName="1-Bins")

final = True
if final:
    results = trainModel( discountRate=0.75, actionProbabilityBase=1.2, randomActionRate=0.01, learningRateScale=0.05, binsPerDimension=8, numEvalTrials=100 )
    print("Final Accuracy:", np.mean(results), "+/-", np.std(results))