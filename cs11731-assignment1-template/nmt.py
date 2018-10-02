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
import torch.nn.functional as F
from models import *
# end yingjinl

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class NMT(object):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # initialize neural network laBaselineGRUEncoderyers...

        # begin yingjinl
        self.src_embedder = nn.Embedding( len( self.vocab.src.word2id ) , self.embed_size )
        self.tar_embedder = nn.Embedding( len( self.vocab.tgt.word2id ) , self.embed_size )

        self.encoder = BaselineGRUEncoder( self.embed_size, self.hidden_size, 2 )
        self.decoder = BaselineGRUDecoder( self.embed_size, self.hidden_size, self.embed_size, 2 )
        self.encoder.to( DEVICE )
        self.decoder.to( DEVICE )
        self.lr = 1e-4
        self.encoder_optim = optim.SGD( filter( lambda x: x.requires_grad, self.encoder.parameters() ),
                                  lr=self.lr )
        self.decoder_optim = optim.SGD( filter( lambda x: x.requires_grad, self.decoder.parameters() ),
                                  lr=self.lr )
        # if USE_CUDA:
        #     self.encoder_optim.cuda()
        #     self.decoder_optim.cuda()
        self.loss = nn.NLLLoss()
        # end yingjinl


    def __call__( self, src_sents , tgt_sents ):
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
        src_encodings, decoder_init_state = self.encode( src_sents )
        scores = self.decode(src_encodings, decoder_init_state, tgt_sents)

        # begin yingjinl
        scores.backward()
        self.encoder_optim.step()
        self.decoder_optim.step()
        # end yingjinl

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
        # begin yingjinl

        #( batch_size, sentence length, embed length )
        _, src_embed = self.embed( src_sents )
        [ batch_size, sentence_len, embed_len ] = src_embed.size()

        src_var = src_embed.view( ( sentence_len, batch_size, embed_len ) )
        # print("encode sentence len {}".format( sentence_len ) )
        if USE_CUDA: src_var = src_var.cuda()
        e_hidden = self.encoder.initial_hidden( batch_size )
        for e_i in range( sentence_len ):
            _, e_hidden = self.encoder( src_var[ e_i ], e_hidden, batch_size )

        _, e_0s = self.embed( [ [ '<s>' ] for i in range( batch_size ) ] )
        decoder_init_state = torch.tensor( e_0s )
        # print( "Exit encoding" )
        return e_hidden, decoder_init_state
        # end yingjinl

    def decode(self, src_encodings, decoder_init_state, tgt_sents):
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
        scores = 0
        true_indices, _ = self.embed( tgt_sents )
        [ batch_size, sentence_len ] = true_indices.size()
        # print( "decode sentence len {}".format( sentence_len ) )
        tar_var = true_indices.view( sentence_len, batch_size )
        if USE_CUDA: tar_var.cuda()

        d_input = decoder_init_state.cuda() if USE_CUDA else decoder_init_state
        d_hidden = src_encodings
        for d_i in range( sentence_len ):
            # print( "D_i reach {}, with d_input dim {}".format( d_i, d_input.size() ) )
            d_out, d_hidden = self.decoder( d_input, d_hidden, batch_size )
            # code taken from https://github.com/pengyuchen/PyTorch-Batch-Seq2seq/blob/master/seq2seq_translation_tutorial.py
            topv, topi = d_out.data.topk( 1, dim = 1 )
            # Cast from torch.cuda.LongTensor to regular Long tensor
            d_input = self.tar_embedder( topi.type( torch.LongTensor ) )
            if USE_CUDA: d_input = d_input.cuda()
            scores += self.loss( d_out, tar_var[ d_i, : ] )
            
        # print( "Exit decode" )
        return scores
    # begin yingjinl
    def pad_batch( self, indices_list ):
        """

        Padd the input batch to make them equal to the length of the longest sentence
        Args:
            indices_list ( batch_size, sentence_len )
        """
        longest_len = max( map( len, indices_list ) )
        for i in range( len( indices_list ) ):
            indices_list[ i ] += [ self.vocab.src.word2id[ "<pad>" ] ] * ( longest_len - len( indices_list[ i ] ) )
        return indices_list

    def embed( self, sentence_list, _type = "src" ):
        """
        Convert an array of sentence into an np array of word indices
        Then perform embedding on those indices according to a embed function
        padd sentence with padding if not at the same length
        Args:
            sentence: a batch of sentence: ( batch_size, sentence_len )
                    [
                        [ w1, w2, w3, ......wk ],
                        [ w1, w2, w3, ..... wk ]
                    ]
            type: src embedding ot tar embedding

        Returns:
            sentence_batch:  a tensor of sentence with embeddings
                            shape ( batch_size, sentence len, embedding_len )
        """
        if _type == "src":
            vocab_entry = self.vocab.src
            embedder = self.src_embedder
        elif _type == "tar":
            vocab_entry = self.vocab.tgt
            embedder = self.tar_embedder
        else:
            print( "_type not implemented in self.embed()" )


        word_indices_list = vocab_entry.words2indices( sentence_list )
        word_indices_list = torch.LongTensor( self.pad_batch( word_indices_list ) )
        sentence_batch = embedder( word_indices_list )

        if USE_CUDA: word_indices_list = word_indices_list.cuda()
        return word_indices_list, sentence_batch



    # this is a test comment
    # end yingjinl

    class Hypothesis(object):
        def __init__(self, d_hidden, value = ['<s>'], score = 0.):
            self.d_hidden = d_hidden
            self.value = value
            self.score = score
            self.incomplete = self.value[-1] == '</s>'

        def completed(self):
            return self.value[-1] == '</s>'

        def generate_candidates(self, model):
            if self.incomplete:
                d_input = [model.embed(self.value[-1:])]
                d_out, self.d_hidden = model.decoder.forward(d_input, self.d_hidden, 1)
                scores = torch.nn.functional.log_softmax(d_out, dim = 0)
                return scores + self.score
            return self.score

    def divmod_(idx, lens):
        ret = 0
        while(idx >= lens[ret]):
            idx -= lens[ret]
            ret += 1
        return ret, idx

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

        src_encodings, decoder_init_state = encode([src_sent])

        time = 0
        hypotheses = [Hypothesis(decoder_init_state) for i in range(beam_size)]
        while(time < max_decoding_time_step):
            time += 1
            all_candidates = [hypothesis.generate_candidates(self) for hypothesis in hypotheses]
            lens = [len(candidates) for candidates in all_candidates] # 1 or |V_tgt|

            scores, indices = torch.topk(input = torch.cat(all_candidates, dim = 0), k = beam_size, dim = 0)

            new_hypotheses = []
            for index in indices:
                hyp_idx, word_idx = divmod_(index, lens)
                hypothesis = hypotheses[hyp_idx]
                new_hypotheses.append(
                    Hypothesis(hypothesis.d_hidden,
                               hypothesis.value + [vocab.tgt.id2word[word_idx]],
                               float(scores[index]))
                    if hypothesis.incomplete else hypothesis)

            hypotheses = new_hypotheses
            if sum(hypothesis.incomplete for hypothesis in hypotheses) == 0:
                break

        return hypotheses

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
            self.encoder.load_state_dict( cpt[ "encoder_state" ] )
            self.decoder.load_state_dict( cpt[ "decoder_state" ] )
            self.encoder_optim.load_state_dict( cpt[ "encoder_optim_state" ] )
            self.decoder_optim.load_state_dict( cpt[ "decoder_optim_state" ] )
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
    lr = args['--lr']

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
                        lr = lr * float(args['--lr-decay'])
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
    was_training = model.training

    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
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
