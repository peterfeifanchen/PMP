import math
import random

class QLearning(object):
    """Simple table based q learning"""
    def __init__(self, stateSpaceShape, numActions, discountRate=.5):
        random.seed()
        self.numActions = numActions
        self.stateSpaceShape = stateSpaceShape
        self.discountRate = discountRate
        self.qTable = {}
        self.qTableVisits = {}
        for i in range(stateSpaceShape[0]*stateSpaceShape[1]*stateSpaceShape[2]*stateSpaceShape[3]):
            self.qTable[i] = [0.0] * numActions
            self.qTableVisits[i] = 0

    def hashState(self, state ):
        return state[0] + state[1] * self.stateSpaceShape[0] + state[2] * self.stateSpaceShape[0] + state[3] * self.stateSpaceShape[0]

    def GetAction(self, state, randomActionRate=0.0, actionProbabilityBase=math.e, learningMode=True):
        hState = self.hashState(state)
        if learningMode:
            # Use randomActionRate, actionProbabilityBase to calculate the action to take
            # from the current state
            randomActionProb = random.random()
            if randomActionProb > randomActionRate:
                actionProbability0 = math.pow( actionProbabilityBase, self.qTable[hState][0] )
                actionProbability1 = math.pow( actionProbabilityBase, self.qTable[hState][1] )
                probabilityThreshold = actionProbability0 / ( actionProbability0 + actionProbability1 )
                return int( random.random() > probabilityThreshold )
            else:
                return int(random.random() > 0.5)
        else:
            # Take the best action from current state
            if self.qTable[hState][0] > self.qTable[hState][1]:
                return 0
            else:
                return 1

    def ObserveAction(self, oldState, action, newState, reward, learningRateScale=0):
        hOldState = self.hashState(oldState)
        hNewState = self.hashState(newState)
        updateWeight = 1.0 / ( 1.0 + learningRateScale * self.qTableVisits[hOldState])
        if self.qTable[hNewState][0] > self.qTable[hNewState][1]:
            maxQNewState = self.qTable[hNewState][0]
        else:
            maxQNewState = self.qTable[hNewState][1]
        self.qTable[hOldState][action] = (1 - updateWeight) * self.qTable[hOldState][action] + updateWeight * ( reward + self.discountRate * maxQNewState )
        self.qTableVisits[hOldState] += 1
        return 0