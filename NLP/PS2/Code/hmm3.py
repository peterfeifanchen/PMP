#!/usr/local/bin/python3

import argparse
import emoji
import os
import json
import numpy as np
from nltk.lm import Vocabulary
from nltk.metrics import ConfusionMatrix, accuracy
from nltk.tag import hmm
from nltk.util import LazyMap
from nltk.probability import ProbDistI, FreqDist, ConditionalFreqDist,\
	 ConditionalProbDist, LidstoneProbDist

# This is the solution to the bigram HMM tagger

def open_file( txt ):
	with open( txt ) as f:
		data = [ line.rstrip() for line in f ]
	return data

def Twitter_Tagset():
	return {
		"N": "common noun",
		"O": "pronoun (not possessive)",
		"S": "nominal (N, O, ^, Z, L, M) with possessive",
		"^": "proper Noun",
		"Z": "proper Noun + possessive",
		"L": "nominal + Verbal",
		"M": "proper noun + Verbal",
		"V": "verb",
		"A": "adjective",
		"R": "adverb",
		"!": "interjection",
		"D": "determiner",
		"P": "pre or postposition, or subordinating conjunction",
		"&": "coordinating conjunction",
		"T": "verb particle",
		"X": "existential there, predeterminer",
		"Y": "X + verbal",
		"#": "hashtag - topic or category indication",
		"@": "at-mention - another user as recipient of tweet",
		"~": "discourse marker, indications of continuation of a message across multiple tweets",
		"U": "URL or email address",
		"E": "emoticon",
		"$": "numeral",
		",": "punctuation",
		"G": "other abbreviations, foreign words...",
	}

def tag_list( tagged_sents ):
	return [ tag for sent in tagged_sents for (word, tag) in sent ]

def unlabeled_words( sent ):
	return [ word for (word, tag) in sent ] 

def handle_lowfreq_words( vocab ):
	def relabel( labeled_sentence ):
		relabeled_symbols = []
		for word, tag in labeled_sentence:
			# Transform low freq words (e.g. oov cut-off) into a word form.
			# If a word does not match a word form nor is in the vocabulary
			if word in vocab:
				relabeled_symbols.append( (word, tag) )
			elif word[0] is '@':
				relabeled_symbols.append( ("MENTION/", tag) )
			elif word[0] is '#':
				relabeled_symbols.append( ("HASHTAG/", tag) )
			elif 'http' in word:
				relabeled_symbols.append( ("URL/", tag ) )
			elif word.isupper():
				relabeled_symbols.append( ("ALLCAP/", tag) )
			elif word[0].isupper():
				relabeled_symbols.append( ("CAPFIRST/", tag) )
			elif word[0].isnumeric():
				relabeled_symbols.append( ("NUMBER/", tag) )
			elif word[0] in emoji.UNICODE_EMOJI:
				relabeled_symbols.append( ("EMOJI/", tag) )	
			else:
				relabeled_symbols.append( ("UNK/", tag) )	
		return relabeled_symbols

	# This is a bit weird with nltk. We just care about the accuracy, but if we
	# want to call the hmm.HiddenMarkovModelTagger.tag() method, then the transform
	# function needs to work on a list[] of unlabeled symbols too.
	#
	# Further, sometimes, transform is called with map, sometimes, it is not. This
	# is probably a coding error in nltk. We will assume everything is called without
	# map in our impl of transform. This means we can only get accuracy from the test
	# method of hmm.HidenMarkovModel. It's entropy(), tag() all seems to want a 
	# different form of transform function.
	def transform( labeled_symbols ):
		return LazyMap( relabel, labeled_symbols )
	return transform

def unzip_tagged_sents( tagset ):
	n = len(tagset)
	words_set = [[]]*n
	tags_set = [[]]*n
	for sent_idx in range(n):
		words, tags = zip(*tagset[sent_idx])
		words_set[sent_idx] = list(words)
		tags_set[sent_idx] = list(tags)
	return words_set, tags_set	

class HiddenMarkovTrigramTagger( hmm.HiddenMarkovModelTagger ):
	def __init__( self, symbols, states, transitions, outputs, priors ):
		super( HiddenMarkovTrigramTagger, self ).__init__( symbols, states,
			 transitions, outputs, priors )
		self.ncall = 0

	def _create_cache( self ):
		if not self._cache:
			N = len(self._states)
			M = len(self._symbols)
			# cached prior probability
			P = np.zeros( N, np.float32 )
			# cached bigram transition probability
			X = np.zeros( (N, N, N), np.float32)
			# cached emission probability
			O = np.zeros( (N, M), np.float32)
			# Initial (<s>, state1, state2)
			G = np.zeros( (N, N), np.float32 )
			for i in range(N):
				si = self._states[i]
				P[i] = self._priors.logprob(si) 
				for k in range(M):
					O[i, k] = self._output_logprob(si, self._symbols[k])
				for j in range(N):
					G[i,j] = self._transitions[('<s>', si)].logprob( (si, self._states[j] ) )	
					sj = self._states[j]
					for k in range(N):
						X[i,j,k] = self._transitions[(si,sj)].logprob( (sj, self._states[k]) )

			# reverse look up from symbols to int, for easier storage?
			# S[ self._symbols[k] ] = k
			S = {}
			for k in range(M):
				S[ self._symbols[k] ] = k
			self._cache = (P, O, X, G, S)	
	
	def _best_path( self, unlabeled_sequence ):
		T = len(unlabeled_sequence)
		N = len(self._states)
		self._create_cache()
		self.ncall += 1
		#print( self.ncall, unlabeled_sequence )
		P, O, X, G, S = self._cache
			
		V = np.zeros( (T,N,N), np.float32 )
		B = -np.ones( (T,N,N), np.int )

		V[0,0] = P + O[:, S[unlabeled_sequence[0]]]
		if T == 1:
			current = np.argmax( V[0,0] )
			return [ self._states[ current ] ]	
		# <s>, s1, s2...
		for s1 in range(N):
			for s2 in range(N): 
				V[1, s1, s2] = V[0,0,s1] + G[s1,s2] + O[s2, S[unlabeled_sequence[1]]]
	
		for t in range(2,T):
			# :, s1, s2(current)...
			for s1 in range(N):
				for s2 in range(N):
					vs = V[t-1, :, s1] + X[:,s1,s2] 
					best = np.argmax(vs)
					V[t,s1,s2] = vs[best] + O[s2, S[unlabeled_sequence[t]]]
					B[t,s1,s2] = best

		# decoding
		current = np.argmax( V[T-1, :, :] )
		s1 = int(current/N)
		s2 = current - s1 * N
		sequence = [s2, s1]
		for t in range( T-1, 1, -1 ):
			last = B[t, s1, s2]
			sequence.append(last)
			s2 = s1
			s1 = last	
		sequence.reverse()
		return list(map(self._states.__getitem__, sequence))

# Bigram transition model with linear interpolation smoothing on the closed set
class InterpolatedProbDist(ProbDistI):
	def __init__( self, trigram_freq, alpha1=0.85, alpha2=0.1, alpha3=0.05,
		 bigram_freq=ConditionalFreqDist(), unigram_freq=FreqDist() ):
		self.alpha1 = alpha1
		self.alpha2 = alpha2
		self.alpha3 = alpha3
		self.trifreqdist = trigram_freq
		self.bifreqdist = bigram_freq
		self.unifreqdist = unigram_freq

	def prob( self, sample ):
		return self.alpha1*self.trifreqdist.freq(sample[1]) + \
			self.alpha2*self.bifreqdist[sample[0]].freq(sample[1]) + \
			self.alpha3*self.unifreqdist.freq(sample[1])

	def max( self ):
		# Do not need for this purpose
		pass
	
	def samples( self ):
		# Do not need for this purpose
		pass
	
class HW2ProbDist(object):
	def __init__( self, labeled_sequence, states, transform, alpha1, alpha2, alpha3,
		 gammaPrior, gammaEmission ):
		self.init = FreqDist()
		self.transition_trigram = ConditionalFreqDist()
		self.transition_bigram = ConditionalFreqDist()
		self.transition_unigram = FreqDist()
		self.emission = ConditionalFreqDist()

		# hyper-parameters for smoothing	
		self.alpha1 = alpha1
		self.alpha2 = alpha2
		self.alpha3 = alpha3	
		self.gammaPrior = gammaPrior
		self.gammaEmission = gammaEmission
	
		self.states = states
		self.symbols = []
		self.labeled_sequence = transform( labeled_sequence )
		
	def train( self ):
		for sequence in self.labeled_sequence:
			lasts = None
			lastsBigram = None
			for token in sequence:
				state = token[1]
				symbol = token[0]
				if symbol not in self.symbols:
					self.symbols.append( symbol )	
				if lasts is None:
					self.init[state] += 1
					lastsBigram = ( "<s>", "<s>" )
				else:
					self.transition_trigram[lastsBigram][state] += 1
					self.transition_bigram[lasts][state] += 1
					self.transition_unigram[state] += 1
				self.emission[state][symbol] += 1
				lasts = state
				lastsBigram = ( lastsBigram[1], lasts )
			
		N = len(self.states)
		st = LidstoneProbDist( self.init, gamma=self.gammaPrior )
		# We've modified the emission labeled data by replacing low frequency words
		# with ones in hanlde_lowfreq_words. We smooth the zero probabilities of
		# p[state][symbol] with add-K smoothing.
		em = ConditionalProbDist( self.emission, LidstoneProbDist,
			 gamma=self.gammaEmission, bins=len(self.symbols) )
		tr = ConditionalProbDist( self.transition_trigram, InterpolatedProbDist,
			 alpha1=self.alpha1, alpha2=self.alpha2, bigram_freq=self.transition_bigram,
			 unigram_freq=self.transition_unigram ) 
		return st, em, tr

if __name__ == "__main__":
	parser = argparse.ArgumentParser( description='This is an HMM PoS tagger with \
		bigram transition model and interpolating smoothing on emission models' )
	parser.add_argument( '-oov', type=int, default=1, help='cut-off for OOV' )
	parser.add_argument( '-a1', type=float, default=0.85,
		 help='transition model hyperparameter: trigram interpolation' )
	parser.add_argument( '-a2', type=float, default=0.10,
		 help='transition model hyperparameter: bigram interpolation' )
	parser.add_argument( '-a3', type=float, default=0.05,
		 help='transition model hyperparameter: unigram interpolation' )
	parser.add_argument( '-gp', type=float, default=0.1,
		 help='prior hyperparameter: Lidstone Smoothing' )
	parser.add_argument( '-ge', type=float, default=0.1,
		 help='emission model hyperparameter: Lidstone Smoothing' )
	parser.add_argument( '-test', default=False, action='store_true',
		 help='true/false: run on test/dev set' )
	args = parser.parse_args()

	tagged_data = [ json.loads( line ) for line in open_file( "twt.train.json" ) ]
	tagset = Twitter_Tagset()

	#observables - words
	#states - part-of-speech tags
	words, tags = unzip_tagged_sents( tagged_data )

	#vocabulary set
	vocab = Vocabulary( unk_cutoff=args.oov )
	_ = [ vocab.update( sent ) for sent in words ] 

	#bigram and unigram transition model for interpolating smoothing
	hmm_model = HW2ProbDist( labeled_sequence=tagged_data,
		 states=tagset, transform=handle_lowfreq_words( vocab ),
		 alpha1=args.a1, alpha2=args.a2, alpha3=args.a3,
		 gammaPrior=args.gp, gammaEmission=args.ge )	
	init_model, emission_model, transition_model =  hmm_model.train()

	#labeled sequences use MLE model for training, unlabelled sequences use
	#Baum-Welch expectation-maximization for training. Transform calling from
	#within the model is wacky, we do the transform of the labeled and unlabeled
	#datasets outside and just use identity in the model.

	# hyperparameters are trained on twt.dev.json
	# results gathered from twt.test.json	
	if not args.test:
		print("Running on dev")
		test_data = [ json.loads( line ) for line in open_file( "twt.dev.json" ) ] 
	else:
		print("Running on test")
		test_data = [ json.loads( line ) for line in open_file( "twt.test.json" ) ] 
	test_data = handle_lowfreq_words( vocab )( test_data )
	twitter_model = HiddenMarkovTrigramTagger( symbols=hmm_model.symbols,
		states=tagset, transitions=transition_model, outputs=emission_model,
		priors=init_model )

	# Compute the accuracy - we can call this, but then we just do extra decoding
	# work. What we really need is just call nltk.metrics.accuracy on the gold and
	# predicted.
	# twitter_model.test( test_data )
	
	# Compute the confusion matrix, technically we would be doing this twice, as
	# when computing accuracy we would've already done this. It would be more
	# optimal to modify the hmm library. But meh.
	
	gold = tag_list( test_data )
	unlabeled_data = LazyMap(unlabeled_words, test_data)
	predicted_labels = list(LazyMap( twitter_model.tag, unlabeled_data ))
	predicted = tag_list(predicted_labels)
	
	acc = accuracy( gold, predicted )
	print( "Accuracy: ", acc )	
	cm = ConfusionMatrix( gold, predicted )
	print( cm.pretty_format(sort_by_count=True, show_percents=True, truncate=25) )
