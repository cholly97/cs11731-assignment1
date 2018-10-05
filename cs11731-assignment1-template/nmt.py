# coding=utf-8
# training architecture borrowed from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""

import math
import pickle
import sys
import time
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union, Any
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry

# begin yingjinl
import os
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable as Variable
import torch.nn.functional as F
from globals import *
if USE_TF:
    from tf_model import *
else:
    from models import *

# end yingjinl

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class Hypothesis_(object):
    def __init__(self, d_hidden, indices = [1], score = 0.): # 1 is the index of '<s>'
        self.d_hidden = d_hidden
        self.indices = indices
        self.score = score
        self.complete = self.indices[-1] == 2 # 2 is the index of '</s>'

    def indices_to_words(self, id2word):
        return Hypothesis([id2word[i] for i in self.indices], self.score)

class Hypothesis(object):
    def __init__(self, value, score):
        self.value = value
        self.score = score

class NMT(object):

    def __init__(self, embed_size, hidden_size, vocab, batch_size = 64, lr = 1e-4, lr_decay = 0.2, dropout_rate=0):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.src_vocab_size = len( self.vocab.src.word2id )
        self.tar_vocab_size = len( self.vocab.tgt.word2id )
        self.batch_size = batch_size
        # if unidirectional we need to reverse ordr
        self.reverse_encoder = False

        if USE_TF:
            self.tf_model = TF_Model( batch_size, embed_size, hidden_size, lr, lr_decay )
            self.tf_model.initialize()
        else:
            # initialize neural network laBaselineGRUEncoderyers...
            # ONLY WORKS FOR ! LAYER
            self.encoder = UnidirectionalGRUEncoder( self.embed_size, self.hidden_size, 1, self.src_vocab_size, 0.2 )
            self.decoder = BaselineGRUDecoder( self.embed_size, self.hidden_size, 1, self.tar_vocab_size, 0.2, "general" )
            self.encoder.to( DEVICE )
            self.decoder.to( DEVICE )
            self.lr = 1e-4
            self.encoder_optim = optim.SGD( filter( lambda x: x.requires_grad, self.encoder.parameters() ),
                                    lr=self.lr )
            self.decoder_optim = optim.SGD( filter( lambda x: x.requires_grad, self.decoder.parameters() ),
                                    lr=self.lr )

            # create weight for the loss function on tar side to mask out <pad>
            weight = np.ones( self.tar_vocab_size )
            weight[ 0 ] = 0
            weight[ 3 ] = 0
            weight = torch.tensor( weight, dtype = torch.float ).cuda()
            self.loss = nn.CrossEntropyLoss( weight = weight )

    def __call__( self, src_sents , tgt_sents, lr = 1e-4 ):
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences.

        Args:
            src_sents: list of source sentence tokens  List[List[str]]
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`  List[List[str]]

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """

        if USE_TF:
            src_word_indices = self.vocab.src.words2indices( src_sents )
            tar_word_indices = self.vocab.tar.words2indices( tgt_sents )
            # here scores is loss
            scores = self.tf_model.train_one_iter( src_word_indices, tar_word_indices )

        else:
            decoder_input, decoder_hidden, encoder_output = self.encode( src_sents )
            scores = self.decode( decoder_hidden, decoder_input, tgt_sents, encoder_output )

            scores.backward()
            self.encoder_optim.step()
            self.decoder_optim.step()
        return scores

    def encode( self, src_sents ):
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens s: List[List[str]]

        Returns: > Tuple[Tensor, Any]
            src_encodings: hidden states of tokens in source sentences, this could be a variable
                with shape (batch_size, source_sentence_length, encoding_dim), or in orther formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """
        if USE_TF:
            src_batch = self.vocab.src.words2indices( src_sents )
            src_batch = self.pad_batch( src_batch, _type = "src" )
            e_hidden, self.h_s = self.tf_model( src_batch )
            decoder_init_state = np.array( self.vocab.tar.words2indices( [ [ '<s>' ] for i in range( batch_size ) ] ) ).reshape( ( 1, self.batch_size ) ).astype( np.int32 )
        else:
            # change to allow encoder to encoder the entire sequence at once
            #( batch_size, sentence length, embed length )
            src_var = self.vocab.src.words2indices( src_sents )
            src_var = self.pad_batch( src_var )
            src_var = torch.tensor( src_var )
            [ batch_size, sentence_len ] = src_var.size()

            src_var = torch.transpose( src_var, 0, 1 ) # ( sentence_len, batch_size )
            # print("encode sentence len {}".format( sentence_len ) )
            if USE_CUDA: src_var = src_var.cuda()
            encoder_output, e_hidden = self.encoder( src_var, None, batch_size )

            e_0s = self.vocab.tgt.words2indices( [ [ '<s>' for i in range( batch_size ) ] ] )
            e_0s =  torch.tensor( e_0s ).cuda()
            decoder_input = e_0s
            decoder_hidden = e_hidden
            # print( "e_0s shape", e_0s.size() )  
            # print( "Exit encoding" )

        return decoder_input, decoder_hidden, encoder_output

    def decode(self, src_encodings, decoder_init_state, tgt_sents, encoder_output ):
                     # hidden state,decodr input
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """
        if USE_TF:
            print( "this is not implemented" )
            scores = 0
        else:
            scores = 0
            tar_var = self.vocab.tgt.words2indices( tgt_sents )
            tar_var = self.pad_batch( tar_var, _type = "tar" )
            tar_var = torch.tensor( tar_var )
            [ batch_size, sentence_len ] = tar_var.size()
            # print( "decode sentence len {}".format( sentence_len ) )
            tar_var = torch.transpose( tar_var, 0, 1 ) # ( sentence_len, batch_size )
            d_hidden = src_encodings
            d_input = decoder_init_state.cuda() if USE_CUDA else decoder_init_state
            context = self.decoder.decoder_context_init( decoder_init_state )
            for d_i in range( 1, sentence_len ):
                # print( "D_i reach {}, with d_input dim {}".format( d_i, d_input.size() ) )
                d_out, d_out_logit, d_hidden, context = self.decoder( d_input, d_hidden, batch_size, context, encoder_output )
                d_input = tar_var[ d_i ]
                if USE_CUDA: d_input = d_input.cuda()
                # print( d_out_logit.size(), d_input.size() )
                scores += self.loss( d_out_logit, d_input )

        return scores
    # begin yingjinl
    def pad_batch( self, indices_list, _type = "src" ):
        """

        Padd the input batch to make them equal to the length of the longest sentence
        Args:
            indices_list ( batch_size, sentence_len )
        """
        if USE_TF:
            if _type == "src":
                longest_len = SRC_MAX_SIZE
            else:
                longest_len = TAR_MAX_SIZE
        else:
            longest_len = max( map( len, indices_list ) )
        for i in range( len( indices_list ) ):
            indices_list[ i ] += [self.vocab.src.word2id[ "<pad>" ]] * ( longest_len - len( indices_list[ i ] ) )
            indices_list[ i ] = [self.vocab.src.word2id[ "<s>" ]] + indices_list[i] +  [self.vocab.src.word2id[ "</s>" ]] 
            # print(  np.array( indices_list[i] ) )
            if self.reverse_encoder:
                indices_list[ i ].reverse()
        return indices_list
    
    # def embed( self, sentence_list, _type = "src" ):
    #     """
    #     Convert an array of sentence into an np array of word indices
    #     Then perform embedding on those indices according to a embed function
    #     padd sentence with padding if not at the same length
    #     Args:
    #         sentence: a batch of sentence: ( batch_size, sentence_len )
    #                 [
    #                     [ w1, w2, w3, ......wk ],
    #                     [ w1, w2, w3, ..... wk ]
    #                 ]
    #         type: src embedding ot tar embedding

    #     Returns:
    #         sentence_batch:  a tensor of sentence with embeddings
    #                         shape ( batch_size, sentence len, embedding_len )
    #     """
    #     if _type == "src":
    #         vocab_entry = self.vocab.src
    #         embedder = self.src_embedder
    #     elif _type == "tar":
    #         vocab_entry = self.vocab.tgt
    #         embedder = self.tar_embedder
    #     else:
    #         print( "_type not implemented in self.embed()" )


    #     word_indices_list = vocab_entry.words2indices( sentence_list )
    #     word_indices_list = torch.LongTensor( self.pad_batch( word_indices_list, _type = _type ) )
    #     sentence_batch = embedder( word_indices_list )

    #     if USE_CUDA: word_indices_list = word_indices_list.cuda()
    #     return word_indices_list, sentence_batch

    def decode_one_step( self, d_hidden, d_prev_word_batch, last_context, encoder_output = None ):
        """
        Take in previous hidden state and previous decoded word indices,
        return new hidden and decoded current word

        Args:
            d_hidden ( batch_size, hidden_size ): previous hidden state, if it is the start, takes in encoder hidden state
            d_prev_word_batch ( 1, batch_size ): a batch of prev word indices

        Returns:
            d_hidden( batch_size, hidden_size ): the new decoder hidden state
            prob_list( batch_size, tar_vocab_size ): return the scores of batch
        """
        [ _, batch_size ] = d_prev_word_batch.size()
        if USE_TF:
            prob_list, d_hidden = self.tf_model.decode_one_step( d_hidden, d_prev_word_batch, self.h_s )
        else:
            prob_list, _, d_hidden, context = self.decoder( d_prev_word_batch, d_hidden,batch_size, last_context, encoder_output )
        return d_hidden, prob_list, context

    def new_hypotheses(self, hypotheses, beam_size, last_context, encoder_output):
        complete, incomplete = [], []
        for hypothesis in hypotheses:
            partition = complete if hypothesis.complete else incomplete
            partition.append(hypothesis)
        incomplete_len = len( incomplete )

        d_hidden = torch.cat( [ hypothesis.d_hidden for hypothesis in incomplete ], dim = 0 )
        d_prev_word_batch = torch.tensor( [ [ hypothesis.indices[ -1 ] ] for hypothesis in incomplete ] )
        d_hidden, prob_list, context = self.decode_one_step( d_hidden, d_prev_word_batch, last_context, encoder_output = encoder_output )

        all_probs = [ prob.cpu().item() + incomplete[ i ].score
                     for prob in prob_list[ i ] for i in range( incomplete_len ) ] + \
                     [ hypothesis.score for hypothesis in complete ] # len( all_probs ) == V * incomplete_len + complete_len
        top_indices = np.argsort( all_probs )[ -beam_size: ]

        new_hypotheses = []
        V = len(self.vocab.tgt)
        for index in top_indices:
            complete_idx = index - V * incomplete_len
            if complete_idx < 0: # index corresponds to incomplete hypothesis
                hyp_idx, word_idx = divmod(index, V) # = index / V, index % V
                new_hypothesis = Hypothesis_(
                    d_hidden[hyp_idx],
                    incomplete[hyp_idx].indices + [word_idx],
                    all_probs[index])
            else:
                new_hypothesis = complete[complete_idx]

            new_hypotheses.append(new_hypothesis)

        return new_hypotheses, context
    
    def top_1_search( self, src_sent, max_decoding_time_step = 100 ):

        decoder_init_state, src_encodings, encoder_output = self.encode([src_sent])
        context = torch.zeros( 3,2 ).cuda()
        time = 0
        decode_array = []
        d_hidden = src_encodings
        decoder_init_state = decoder_init_state.reshape( (1,1) )
        d_hidden, prob_list, context = self.decode_one_step( d_hidden, decoder_init_state, context, encoder_output = encoder_output)
        _, index = torch.topk( prob_list, 1 )
        decode_array.append( index.cpu().item() )
        while index != 1 and time < max_decoding_time_step:

            d_prev_word_batch = torch.tensor( index ).reshape( (1,1 ) ).cuda()
            d_hidden, prob_list, context = self.decode_one_step(d_hidden, d_prev_word_batch, context, encoder_output = encoder_output)
            _, index = torch.topk( prob_list, 1 )
            decode_array.append( index.cpu().item() )
            time += 1
        res =  [ self.vocab.tgt.id2word[ i ] for i in decode_array ]
        print("res", res )
        return [res]



    def beam_search(self, src_sent, beam_size: int=5, max_decoding_time_step: int=70):
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence : List[str]
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns: -> List[Hypothesis]
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        src_encodings, decoder_init_state, encoder_output = self.encode([src_sent])
        context = torch.zeros( 3,2 ).cuda()
        time = 0
        hypotheses = [Hypothesis_(decoder_init_state) for _ in range(beam_size)]
        while(not all(hypothesis.complete for hypothesis in hypotheses)
              and time < max_decoding_time_step):
            time += 1
            hypotheses, context = self.new_hypotheses(hypotheses, beam_size, context, encoder_output)

        return [hypothesis.indices_to_words(self.vocab.tgt.id2word) for hypothesis in hypotheses]

    def evaluate_ppl(self, dev_data, model, batch_size: int=32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences : List[Any]
            batch_size: batch size

        Returns:
            ppl: the perplexity on dev sentences
        """

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`
        # with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl

    @staticmethod
    def load(model_path: str):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        if os.path.isfile( model_path ):
            cpt = torch.load( model_path )
            model = NMT(embed_size=256,
                        hidden_size=256,
                        dropout_rate=0.2,
                        vocab=pickle.load(open("data/vocab.bin", "rb")))
            model.encoder.load_state_dict( cpt[ "encoder_state" ] )
            model.decoder.load_state_dict( cpt[ "decoder_state" ] )
            model.encoder_optim.load_state_dict( cpt[ "encoder_optim_state" ] )
            model.decoder_optim.load_state_dict( cpt[ "decoder_optim_state" ] )
        else:
            print( "No checkpoint found in {}".format( model_path ) )

        return model

    def save(self, path: str):
        """
        Save current model to file
        """
        cpt = dict()
        cpt[ "encoder_state" ] = self.encoder.state_dict()
        cpt[ "decoder_state" ] = self.decoder.state_dict()
        cpt[ "encoder_optim_state" ] = self.encoder_optim.state_dict()
        cpt[ "decoder_optim_state" ] = self.decoder_optim.state_dict()
        cpt[ "vocab_path" ] = "data/vocab.bin"
        cpt[ "hidden_size" ] = self.hidden_size
        cpt[ "embed_size" ] = self.embed_size
        torch.save( cpt, path )


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def train(args: Dict[str, str]):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    lr = float( args['--lr'] )

    vocab = pickle.load(open(args['--vocab'], 'rb'))

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            print( "train iter: {}".format( train_iter ) )

            batch_size = len(src_sents)

            # (batch_size)
            if USE_TF:
                loss = model(src_sents, tgt_sents, lr = lr)
            else:
                loss = -model(src_sents, tgt_sents).item()

            report_loss += loss
            cum_loss += loss

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         np.exp(cum_loss / cumulative_tgt_words),
                                                                                         cumulative_examples), file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = model.evaluate_ppl(dev_data, model, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # You may also save the optimizer's state
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = float( lr ) * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        model_save_path

                        print('restore parameters of the optimizers', file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    # was_training = model.training

    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        # res = model.top_1_search( src_sent, max_decoding_time_step=max_decoding_time_step )
        example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

        hypotheses.append(example_hyps)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    # begin yingjinl
    print( "load model from {}".format( args[ 'MODEL_PATH' ] ) , file=sys.stderr )
    #end yingjinl change the wierd string formatting
    model = NMT.load( args['MODEL_PATH'] )

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        # begin yingjinl
        print( 'Corpus BLEU: {}'.format( bleu_score ), file=sys.stderr)
        # end yingjinl

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        # yingjinl
        # raise RuntimeError(f'invalid mode')
        raise RuntimeError('invalid mode')
        # end yingjinl


if __name__ == '__main__':
    main()
