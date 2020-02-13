#!/usr/local/bin/python3

import argparse
import numpy as np
from nltk.lm.api import LanguageModel
from nltk.lm import Vocabulary
from nltk.lm.util import log_base2
from nltk.tokenize import word_tokenize
from nltk.util import pad_sequence, ngrams

class HW1LanguageModel( LanguageModel ):
	#NOTE: In this model, we are case-sensitive as it was unclear from the assignment
	#whether that was something that was up to us.
	def __init__( self, vocab, order, k, interpolate ):
		self.order = order
		self.k = k
		self.interpolate = interpolate
		self.vocab = vocab		
		
		self.vocab_count = sum(list(self.vocab.counts.values()))

		self.vocab_unk_count = sum( [ self.vocab.counts[ word ]
				 if word not in self.vocab else 0
				 for word in list(self.vocab.counts.keys()) ] )
		
		# nltk.lm.api.LanguageModel by default uses nltk.lm.counter.NgramCounter
		# which takes in a list of tuples
		super( HW1LanguageModel, self ).__init__( order=self.order,
			 vocabulary=self.vocab )
	
	#NOTE: This function is not directly called. Other functions of LanguageModel
	#such as generate(), perplexity(), entropy() would call score() which masks the
	#input with OOV '<UNK>' token using self.vocab. As part of the call, these func
	#would provide the context. 
	def unmasked_score( self, word, context=None ):
		# Returns the p(word|context). If k is set, it applies Add-K smoothing
		# to each ngram set. If interpolate is set, it applies interpolation to each
		# of the ngram models in the order that the weights were provided. 		
		if not self.interpolate:
			return self.calculate_ngram( word, context ) 
		return self.interpolate_score( word, context ) 

	def interpolate_score( self, word, context=None ):
			
		interpolate_len = len(self.interpolate)
		ctx_len = len(context) if context else 0
		prob = [0] * interpolate_len
		for i in range( 0, interpolate_len ):
			subctx = context[ctx_len-i:ctx_len] if context else None
			prob[i] = self.calculate_ngram( word, subctx if i > 0 else None )
			   
		return np.dot( prob, self.interpolate )

	def calculate_ngram( self, word, context=None ):
		# We grab the sub-count of the given context. If context is None, we return
		# unigram counter. 
		ctx_counts = self.context_counts(context)
		word_counts = ctx_counts[word]
		# TODO: if we have never seen this word in this context and we have no smoothing
		# then we return 0 for now. We will figure out how to deal with this in the future
		if word_counts == 0 and self.k == 0:
			return 0
		norm_counts = ctx_counts.N() #total counts
		n = len(ctx_counts.values()) #number of items

		# TODO: if we have never seen this context, we return 0 for now (not important for
		# assignment. Probably deal with it better in the future.
		if norm_counts == 0 and n == 0:
			return 0
		return ( word_counts + self.k ) / ( norm_counts + n * self.k )

	def calculate_logscore( self, word, context ):
		ctx = self.context_counts( context )
		if ctx.N() > 0 and ( ctx[word] > 0 or self.k > 0 ):
			return self.logscore( word, context )
		else:
			if word in self.vocab:
				return log_base2( self.vocab.counts[ word ] / self.vocab_count )
			else:
				return log_base2( self.vocab_unk_count / self.vocab_count )

	def calculate_perplexity( self, text ):
		# If a context does not exist from the training data. We set it to 1/size of the
		# vocabulary since its perplexity would be even for any word in the vocabulary.	
		entropy = 0
		num = 0
		for sent in text:
			num += len( sent )
			sent_score = [ self.calculate_logscore( ngram[-1], ngram[:-1] ) for ngram in sent ]
			entropy += sum( sent_score )

		return pow( 2.0, -1 * entropy / num )	

def open_file( txt ):
	with open(txt) as f:
		data = [ line.rstrip() for line in f ]
	return data

if __name__ == "__main__":
	parser = argparse.ArgumentParser( description='This is an N-gram language model \
		with smoothing (add-K and/or linear interpolation) and OOV cutoff') 
	parser.add_argument( '-n', type=int, default=1,
		 help='N-gram order' )
	parser.add_argument( '-oov', type=int, default=1,
		 help='cut-off for OOV' )
	parser.add_argument( '--k', type=float, default=0,
		 help='Add-K smoothing, if interpolate is set, Add-K smoothing applies to \
			the highest order N-gram only' )
	parser.add_argument( '--interpolate', nargs='*', type=float, default=None,
		 help="Interpolate weights with smallest ngram first. If interpolate flag \
			is provided, n means up to ngram models combined together using these \
			 linear weights")
	args = parser.parse_args()
	
	# Read training data from brown.train.txt and preprocess for unigram, bigram
	# and trigram training by adding <s> ('START') and </s> ('STOP') sequences.
	training = [ word_tokenize( sent ) for sent in open_file("brown.train.txt")	]
	padded_training = [ list(pad_sequence( sequence, n=args.n,
		pad_left=True, left_pad_symbol="<s>",
		pad_right=True, right_pad_symbol="</s>" )) for sequence in training ]
	
	# nltk.lm.Vocabulary creates a vocabulary count of training_data. A modified
	# vocabulary is created by masking any unigram with count less than
	# unk_cutoff as '<UNK>'. N-gram counts are then constructed from the modified
	# vocabulary set.
	#
	# NOTE: while self.vocab[$UNIGRAM_WITH_COUNT_LESS_THAN_UNK_CUTOFF] will still
	# retain the actual count, the check
	# '$UNIGRAM_WITH_COUNT_LESS_THAN_UNK_CUTOFF' in self.vocab will return false
	# NOTE: there is no count of '<UNK>', but I guess you can get it by looking up
	# all the words in vocab.counts with count less than the cutoff
	modified_vocab = Vocabulary( unk_cutoff=args.oov )
	_ = [ modified_vocab.update( sent ) for sent in padded_training ]
	
	lm = HW1LanguageModel( vocab=modified_vocab, order=args.n, k=args.k,
		 interpolate=args.interpolate ) 

	# vocabulary_text uses modified vocab to replace each ngram in ngram_text that
	# has a word that did not meet the OOV cutoff as '<UNK>'. The modified ngrams
	# are then used to create the counts in lm.counts using NgramCounter. The final
	# counts can be looked up using:
	# 	unigram - lm.counts[ $WORD ]
	# 	bigram - lm.counts[ ($WORD,) ]
	#	trigram - lm.counts[ ($WORD, $WORD, ) ]	
	ngram_text = [ ngrams( sent, args.n ) for sent in padded_training ]	
	lm.fit( ngram_text, vocabulary_text=modified_vocab )
	if args.interpolate:
		for i in range(1, args.n):
			padded_training = [ list(pad_sequence( sequence, n=i,
				pad_left=True, left_pad_symbol="<s>",
				pad_right=True, right_pad_symbol="</s>" )) for sequence in training ]
			ngram_text = [ ngrams( sent, i ) for sent in padded_training ]	
			lm.fit( ngram_text, vocabulary_text=modified_vocab )

	# base class perplexity func takes a list of ngram words, however our list
	# expression generates a list of sentences broken now into ngram tuples. So we
	# write our own perplexity function. We still use self.logscore (but simply undo
	# the avg) because it would replace all OOV words with '<UNK>'. 
	training_text = [ list(ngrams( sent, args.n )) for sent in padded_training ]	
	pp_training = lm.calculate_perplexity( training_text )
	print( "training:", pp_training)			

	# run lm on the dev set
	dev = [ word_tokenize( sent ) for sent in open_file("brown.dev.txt")	]
	padded_dev = [ list(pad_sequence( sequence, n=args.n,
		pad_left=True, left_pad_symbol="<s>",
		pad_right=True, right_pad_symbol="</s>" )) for sequence in dev ]
	dev_text = [ list(ngrams( sent, args.n )) for sent in padded_dev ]
	pp_dev = lm.calculate_perplexity( dev_text )
	print( "dev: ", pp_dev )
	
	# run lm on test set
	test = [ word_tokenize( sent ) for sent in open_file("brown.test.txt")	]
	padded_test = [ list(pad_sequence( sequence, n=args.n,
		pad_left=True, left_pad_symbol="<s>",
		pad_right=True, right_pad_symbol="</s>" )) for sequence in test ]
	test_text = [ list(ngrams( sent, args.n )) for sent in padded_test ]
	pp_test = lm.calculate_perplexity( test_text )
	print( "test: ", pp_test )

	print( "generated sentence: ", lm.generate(num_words=100, text_seed=['<s>','<s>']) )
