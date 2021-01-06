import collections
import math
import time
import numpy as np

def Entropy(y, w):
    totalWeightSum = sum(w)
    if totalWeightSum < 0.000001:
        return 0.0
    w_numpy = np.array(w)
    y_numpy = np.array(y)
    # we take advantage of the fact one of the labels is zero
    label1WeightSum = np.transpose(y_numpy).dot(w_numpy)
    
    probOf1 = label1WeightSum/totalWeightSum
    probOf0 = 1 - probOf1
    if probOf1 < 0.000000001 or probOf0 < 0.00000001:
        return 0.0
    return -(probOf1 * np.log(probOf1) + probOf0 * np.log(probOf0))

def FindBestSplitOnFeature(x, y, w, featureIndex):
    if len(y) < 2:
        # there aren't enough samples so there is no split to make
        return None
        
    # HINT here is how to get an array that has indexes into the data (x, y) arrays in sorted 
    # order based on the value of the feature at featureIndex
    indexesInSortedOrder = sorted(range(len(x)), key = lambda i : x[i][featureIndex])
    
    # so x[indexesInSortedOrder[0]] will be the training sample with the smalles value of 'featureIndex'
    # and y[indexesInSortedOrder[0]] will be the associated label
    prevLabel = x[indexesInSortedOrder[0]][featureIndex]
    splitIndex = -1
    bestThreshold = None
    entropyAfterSplit = Entropy(y, w)
    for i in range(len(indexesInSortedOrder)):
        if x[indexesInSortedOrder[i]][featureIndex] != prevLabel:
            prevLabel = x[indexesInSortedOrder[i]][featureIndex]
            thisThreshold = ( x[indexesInSortedOrder[i]][featureIndex] + x[indexesInSortedOrder[i-1]][featureIndex] ) / 2.0
            thisY1 = []
            thisW1 = []
            thisY2 = []
            thisW2 = []
            for j in indexesInSortedOrder[:i]:
                thisY1.append(y[j])
                thisW1.append(w[j])
            for j in indexesInSortedOrder[i:]:
                thisY2.append(y[j])
                thisW2.append(w[j])
            thisEntropy = sum(thisW1)/sum(w)*Entropy(thisY1, thisW1) + sum(thisW2)/sum(w)*Entropy(thisY2, thisW2)
            if thisEntropy < entropyAfterSplit:
                entropyAfterSplit = thisEntropy
                bestThreshold = thisThreshold
                splitIndex = i
    
    if splitIndex == -1:
        return (np.Inf, ((x, y, w), ([], [], [])), entropyAfterSplit) 

    splitData = ( ([],[], []), ([],[],[]) )
    for index in indexesInSortedOrder[:splitIndex]:
        splitData[0][0].append(x[index])
        splitData[0][1].append(y[index])
        splitData[0][2].append(w[index])
    for index in indexesInSortedOrder[splitIndex:]:
        splitData[1][0].append(x[index])
        splitData[1][1].append(y[index])
        splitData[1][2].append(w[index])

    # HINT: might like to return the partitioned data and the
    #  entropy after partitioning based on the threshold
    return (bestThreshold, splitData, entropyAfterSplit)

class TreeNode(object):
    def __init__(self, depth = 0):
        self.depth = depth
        self.labelDistribution = collections.Counter()
        self.labelWeight = collections.Counter()
        self.splitIndex = None
        self.threshold = None
        self.children = []
        self.x = []
        self.y = []
        self.w = []

    def isLeaf(self):
        return self.splitIndex == None

    def addData(self, x, y, w):
        self.x += x
        self.y += y
        self.w += w

        for i in range(len(y)):
            self.labelDistribution[y[i]] += 1
            self.labelWeight[y[i]] += w[i]

    def growTree(self, maxDepth):
        if self.depth == maxDepth:
            return
        if self.labelDistribution[0] == self.labelDistribution[0] + self.labelDistribution[1]:
            return
        if self.labelDistribution[1] == self.labelDistribution[0] + self.labelDistribution[1]:
            return

        numFeatures = len(self.x[0])
        maxEntropyGain = 0.0
        chooseFeature = -1
        nodeThreshold = np.inf
        split = ( ([], []), ([], []))
        for f in range(numFeatures):
            threshold, splitData, entropy = FindBestSplitOnFeature(self.x, self.y, self.w, f) 
            entropyGain = Entropy(self.y, self.w) - entropy       
            if entropyGain > maxEntropyGain:
                maxEntropyGain = entropyGain
                chooseFeature = f
                nodeThreshold = threshold
                split = splitData
        
        if chooseFeature >= 0:
            self.threshold = nodeThreshold
            self.splitIndex = chooseFeature
            partition1 = TreeNode(depth=self.depth+1)
            partition1.addData(split[0][0], split[0][1], split[0][2])
            partition1.growTree(maxDepth)
            partition2 = TreeNode(depth=self.depth+1)
            partition2.addData(split[1][0], split[1][1], split[1][2])
            partition2.growTree(maxDepth)
            self.children = [ partition1, partition2 ]

    def predictProbability(self,x):
        if self.isLeaf():
            totalSamples = self.labelDistribution[1] + self.labelDistribution[0]
            totalSampleWeights = self.labelWeight[1] + self.labelWeight[0]
            smoothing = totalSampleWeights/totalSamples
            return (self.labelWeight[1]+smoothing)/(self.labelWeight[1]+self.labelWeight[0]+2*smoothing)

        if x[self.splitIndex] < self.threshold:
            return self.children[0].predictProbability(x)
        else:
            return self.children[1].predictProbability(x)
    
    def visualize(self, depth=1):
        ## Here is a helper function to visualize the tree (if you choose to use the framework class)
        if self.isLeaf():
            print(self.labelDistribution)

        else:
            print("Split on: %d" % (self.splitIndex))

            # less than
            for _ in range(depth):
                print(' ', end='', flush=True)
            print("< %f -- " % self.threshold, end='', flush=True)
            self.children[0].visualize(depth+1)

            # greater than or equal
            for _ in range(depth):
                print(' ', end='', flush=True)
            print(">= %f -- " % self.threshold, end='', flush=True)
            self.children[1].visualize(depth+1)

    def countNodes(self):
        if self.isLeaf():
            return 1

        else:
            return 1 + self.children[0].countNodes() + self.children[1].countNodes()

class DecisionTreeWeighted(object):
    """Wrapper class for decision tree learning."""

    def __init__(self):
        pass

    def fit(self, x, y, weights = None, maxDepth = 10000, verbose=True):
        self.maxDepth = maxDepth
        if weights is None:
            weights = [1.0 for _ in y ]

        startTime = time.time()

        self.treeNode = TreeNode(depth=0)

        self.treeNode.addData(x,y, weights)
        self.treeNode.growTree(maxDepth)
        
        endTime = time.time()
        runtime = endTime - startTime
        
        if verbose:
            print("Decision Tree completed with %d nodes (%.2f seconds) -- %d features. Hyperparameters: maxDepth=%d." % (self.countNodes(), runtime, len(x[0]), maxDepth))

    def predictProbabilities(self, x):
        y = []

        for example in x:
            y.append(self.treeNode.predictProbability(example))        
            
        return y

    def predict(self, x, classificationThreshold=0.5):
        return [ 1 if probability >= classificationThreshold else 0 for probability in self.predictProbabilities(x) ]

    def visualize(self):
        self.treeNode.visualize()

    def countNodes(self):
        return self.treeNode.countNodes()