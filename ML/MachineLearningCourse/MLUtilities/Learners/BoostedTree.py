import time
import numpy as np
import MachineLearningCourse.MLUtilities.Learners.DecisionTreeWeighted as DecisionTreeWeighted


class BoostedTree(object):
    """Wrapper class for decision tree learning."""

    def __init__(self):
        self.decisionTrees = []

    def fit(self, x, y, maxDepth = 10000, k=10, verbose=True):
        w = [ 1.0/len(y) for _ in y ]
        startTime = time.time()

        for r in range(k):
            # Calculate weights
            w_sum = sum(w)
            w_normalized = [ w_i / w_sum for w_i in w ]

            # Train weighted decision tree
            model = DecisionTreeWeighted.DecisionTreeWeighted()
            model.fit(x, y, w_normalized, maxDepth=maxDepth)
            
            # Calculate model error
            y_hat_np = np.array(model.predict(x))
            y_np = np.array(y)
            err_v = np.abs(y_hat_np-y_np)
            err = np.dot(np.array(w_normalized), err_v)
            if err > 0.5:
                return r - 1

            # Update weights
            beta = err/(1-err)            
            w = list(np.array(w) * np.power(beta, 1-err_v))

            # Store the weighted decision tree
            self.decisionTrees.append((model, beta))
        
        endTime = time.time()
        runtime = endTime - startTime
        
        if verbose:
            print("Boosting Tree completed (%.2f seconds) -- %d features. Hyperparameters: maxDepth=%d, k=%d." % (runtime, len(x[0]), maxDepth, k))

    def predict(self, x, classificationThreshold=0.5):
        hyp_1 = np.array([0]*len(x))
        hyp_0 = np.array([0]*len(x))
        for m in self.decisionTrees:
            y_est = np.array(m[0].predict(x, classificationThreshold))
            model_weight = np.log(1.0/m[1])
            hyp_1 = hyp_1 + y_est * model_weight
            hyp_0 = hyp_0 + (1 - y_est) * model_weight
        
        return [ 1 if hyp_1[i] > hyp_0[i] else 0 for i in range(len(x)) ]