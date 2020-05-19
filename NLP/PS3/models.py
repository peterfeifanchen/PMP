# models.py

import time
import random
import math
import torch
from torch import nn
import embeddings as E
from sentiment_data import *
from collections import Counter, defaultdict
from nltk.util import ngrams

class FeatureExtractor:

    def extract_features(self, ex_words: List[str]) -> List[int]:
        raise NotImplementedError()


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence.
	It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__( self ):
        self.c = Counter()

    def extract_features(self, ex_words):
        """
        Q1: Implement the unigram feature extractor.
        Hint: You may want to use the Counter class.
        """
		# We take a bag of words and add it to the unigram counters.
		# We store these features and their counts as a key-value store,
		# since we don't know ahead of time how many features there are
		# and how to map them to indices.
        for w in ex_words:
            self.c[w.lower()]+=1

		# For features we don't care about, we set it to -1
        # We don't care about '.'
        # self.c['.'] = -1
        common_tokens = ['.', '\'s', 'the', 'to', 'be', 'is', 'are', 'a', 'and',
        '\'ll', 'by', 'its', 'it', 'as', 'from', 'that', 'have', 'has', 'in', 'of',
        'their', 'another', 'you', 'i', 'he', 'his', 'her', 'she', 'was', '\'re', '\'ve',
        ',', 'your', 'an', '\'', '\`', 'with', 'for', 'film', 'this', 'but', 'one', 'at',
        'so', 'movie', 'on']
        
		# We don't care about the least occuring k unigrams once our unigram
        # features are more than a threshold.
        #if len(self.c) > self.threshold:
        #    least_common_unigrams = self.c.most_common()[:-self.k-1:-1]
        #    for unigram in least_common_unigrams:
        #        self.c[unigram[0]] = -1

        f = Counter()
        for w in ex_words:
            if (w.islower() or w == ex_words[0]) and '-' not in w and \
                w.lower() not in common_tokens and self.c[w.lower()] > 1:
                f[w.lower()] = 1		
		# We construct the feature counter		 
        return f

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self):
        self.c = Counter()

    def extract_features(self, ex_words):
        """
        Q3: Implement the unigram feature extractor.
        Hint: You may want to use the Counter class.
        """
        common_tokens = ['.', '\'s', 'the', 'to', 'be', 'is', 'are', 'a', 'and',
        '\'ll', 'by', 'its', 'it', 'as', 'from', 'that', 'have', 'has', 'in', 'of',
        'their', 'another', 'you', 'i', 'he', 'his', 'her', 'she', 'was', '\'re', '\'ve',
        ',', 'your', 'an', '\'', '\`', 'with', 'for', 'film', 'this', 'but', 'one', 'at',
        'so', 'movie', 'on']

        ex_words_filtered = []
        for w in ex_words:
             self.c[w.lower()]+=1
             if w.lower() not in common_tokens:
                 if self.c[w.lower()] > 1:
                     ex_words_filtered.append(w.lower())
                 else:
                     ex_words_filtered.append('<UNK>')

     

        unigrams = list(ngrams(ex_words_filtered, 1))
        bigrams = list(ngrams(ex_words_filtered, 2))
        trigrams = list(ngrams(ex_words_filtered, 3))
        f = Counter()                
        
        for g in unigrams:
            if '-' not in g[0] and g[0] != '<UNK>':
                f[g] = 1

        for g in bigrams:
            f[g] = 1

        for g in trigrams:
            f[g] = 1
        
        return f

class SentimentClassifier(object):

    def featurize(self, ex):
        raise NotImplementedError()

    def forward(self, feat):
        raise NotImplementedError()

    def extract_pred(self, output):
        raise NotImplementedError()

    def update_parameters(self, output, feat, ex, lr):
        raise NotImplementedError()

    def run_train(self, train_data: List[SentimentExample], dev_data: List[SentimentExample], lr=1e-3, epoch=10):
        """
        Training loop.
        """
        print('begin training...')
        train_data = train_data[:]
        for ep in range(epoch):
            start = time.time()
            random.shuffle(train_data)

            if isinstance(self, nn.Module):
                self.train()

            acc = []
            for ex in train_data:
                #print('epoch {}, sample {}'.format(ep, ex))
                feat = self.featurize(ex)
                output = self.forward(feat)
                self.update_parameters(output, feat, ex, lr)
                predicted = self.extract_pred(output)
                acc.append( predicted == ex.label)
            acc = sum(acc) / len(acc)
            if isinstance(self, nn.Module):
                self.eval()

            dev_acc = []
            for ex in dev_data:
                feat = self.featurize(ex)
                output = self.forward(feat)
                predicted = self.extract_pred(output)
                dev_acc.append(predicted == ex.label)
            dev_acc = sum(dev_acc) / len(dev_acc)
            print('epoch {}: train acc = {}, dev acc = {}, time = {}'.format(ep, acc, dev_acc, time.time() - start))
        if isinstance(self, PerceptronClassifier):
            neg_most_common = self.w[0].most_common(10)
            print('highest weighted unigrams (-): {}'.format(neg_most_common))
            neg_least_common = self.w[0].most_common()[:-11:-1]
            print('least weighted unigrams (-): {}'.format(neg_least_common))
            pos_most_common = self.w[1].most_common(10)
            print('highest weighted unigrams (+): {}'.format(pos_most_common))
            pos_least_common = self.w[1].most_common()[:-11:-1]
            print('least weighted unigrams (+): {}'.format(pos_least_common))
    def predict(self, ex: SentimentExample) -> int:
        feat = self.featurize(ex)
        output = self.forward(feat)
        return self.extract_pred(output)


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, ex: SentimentExample) -> int:
        return 1

    def run_train(self, train_data: List[SentimentExample], dev_data: List[SentimentExample], lr=None, epoch=None):
        pass


class PerceptronClassifier(SentimentClassifier):
    """
    Q1: Implement the perceptron classifier.
    """

    def __init__(self, feat_extractor):
        self.feat_extractor = feat_extractor
        # index 0: contains the block weights for - label
        # index 1: contains the block weights for + label
        self.w = [ Counter(), Counter() ]
        self.gamma = 0.1

    def featurize(self, ex):
        """
        Converts an example into features.
        """
        return self.feat_extractor.extract_features(ex.words)

    def forward(self, feat) -> float:
        # compute the activation of the perceptron
        # Dot product of weights and features if label is 0
        p0 = 0
        for f in feat:
            p0 += self.w[0][f]
		# Dot product of weigths and features if label is 1
        p1 = 0
        for f in feat:
            p1 += self.w[1][f]
        return nn.functional.softmax(torch.tensor([p0,p1], dtype=torch.float32), -1)

    def extract_pred(self, output) -> int:
        # compute the prediction of the perceptron given the activation
        return 0 if output[0] > output[1] else 1

    def update_parameters(self, output, feat, ex, lr):
        # update the weight of the perceptron given its activation, the input features, the example, and the learning rate
        for f in feat:
            self.w[ex.label][f] += lr*(1-output[ex.label])*feat[f] - self.gamma*self.w[ex.label][f]
            self.w[1-ex.label][f] += lr*(-output[ex.label])*feat[f] - self.gamma*self.w[ex.label][f]

class FNNClassifier(SentimentClassifier, nn.Module):
    """
    Q4: Implement the multi-layer perceptron classifier.
    """

    def __init__(self, args):
        super().__init__()
        self.glove = E.GloveEmbedding('wikipedia_gigaword', 300, default='zero')
        ### Start of your code
        self.linear1 = nn.Linear(300, 100)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        ### End of your code

        # do not touch this line below
        self.optim = torch.optim.Adam(self.parameters(), args.learning_rate)

    def featurize(self, ex):
        # You do not need to change this function
        # return a [T x D] tensor where each row i contains the D-dimensional embedding for the ith word out of T words
        embs = [self.glove.emb(w.lower()) for w in ex.words]
        return torch.Tensor(embs)

    def forward(self, feat) -> torch.Tensor:
        # compute the activation of the FNN
        feat = feat.unsqueeze(0)
        # sum along the dimension T
        out = feat.sum(1)
        out = self.linear1(out)
        out = self.tanh(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out        

    def extract_pred(self, output) -> int:
        # compute the prediction of the FNN given the activation
        return 1 if output[0][0] > 0.5 else 0

    def update_parameters(self, output, feat, ex, lr):
        # update the weight of the perceptron given its activation, the input features, the example, and the learning rate
        target = torch.Tensor([[ex.label]])
        loss = nn.functional.binary_cross_entropy(input=output,target=target) 
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

class RNNClassifier(FNNClassifier):

    """
    Q5: Implement the RNN classifier.
    """

    def __init__(self, args):
        super().__init__(args)
        # Start of your code
		# NOTE: bidrectional LSTM outputs a concatenated output for each direction
        self.rnn = nn.LSTM(input_size=300, hidden_size=20, num_layers=1, batch_first=True,
             bidirectional=True)
        self.linear = nn.Linear(40,1)
        self.sigmoid = nn.Sigmoid()
        # End of your code
        self.optim = torch.optim.Adam(self.parameters(), args.learning_rate)

    def forward(self, feat):
        feat = feat.unsqueeze(0)
        out, _ = self.rnn(feat) # batch x seq x (hidden_size*num_directions)
        out = out.max(1)[0] # ( max, indices )
        out = self.linear(out)
        out = self.sigmoid(out)
        return out

class MyNNClassifier(FNNClassifier):

    """
    Q6: Implement the your own classifier.
    """

    def __init__(self, args):
        super().__init__(args)
        # Start of your code
        # We can improve RNNClassifier by:
        #	 + dropout
        #    + add num_layers
        #    + hidden_size
        #    + batch to provide stability
        raise NotImplementedError('Your code here')

        # End of your code
        self.optim = torch.optim.Adam(self.parameters(), args.learning_rate)

    def forward(self, feat):
        feat = feat.unsqueeze(0)

        raise NotImplementedError('Your code here')


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You don't need to change this.
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor()
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor()
    else:
        raise Exception("Pass in UNIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = PerceptronClassifier(feat_extractor)
    elif args.model == "FNN":
        model = FNNClassifier(args)
    elif args.model == 'RNN':
        model = RNNClassifier(args)
    elif args.model == 'MyNN':
        model = MyNNClassifier(args)
    else:
        raise NotImplementedError()

    model.run_train(train_exs, dev_exs, lr=args.learning_rate, epoch=args.epoch)
    return model
