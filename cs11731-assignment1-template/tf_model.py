import os
import tensorflow as tf
import tensorflow.nn.rnn_cell as rnn
import numpy as np
import time

from globals import *
from tf_utils import *


# tensorflow implementation of the hw1 reference paper https://arxiv.org/pdf/1508.04025.pdf
# some details of implementation borrows from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class TF_Model( object ):

    def __init__( self, batch_size, embed_size, hidden_size, lr, lr_decay, dropout_rate=0.2, num_layers = 2,
                  initializer = tf.random_uniform_initializer( -1.0, 1.0 ), teacher_sup = True ):

        self.sess = None
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.src_max_size = SRC_MAX_SIZE
        self.tar_max_size = TAR_MAX_SIZE
        self.src_minval = SRC_MIN_VOCAB_VAL
        self.tar_minval = TAR_MIN_VOCAB_VAL
        self.src_maxval = SRC_MAX_VOCAB_VAL  # s_nwords
        self.tar_maxval = TAR_MAX_VOCAB_VAL  # t_nwords
        self.lr = lr
        self.lr_decay = lr_decay
        self.initializer = initializer
        self.is_sample = False
        self.teacher_sup = teacher_sup
        self.attention_type = 'dot' # 'dot' | 'general' | 'concat'
        self.build_variable()
        self.initialize()

    def initialize( self ):
        self.sess = tf.Session()
        self.sess.run( [ tf.global_variables_initializer() ] )


    def build_variable( self ):

        self.construct_input()
        self.build_projection()
        self.build_attention()
        self.build_network()
        self.build_graph()
        self.saver = tf.train.Saver()

    def build_projection( self ):
        with tf.variable_scope( "encoder" ) as scope:

            self.src_embed_matrix = tf.get_variable( "src_embed_matrix", shape = [ self.src_maxval, self.embed_size ], initializer = self.initializer )
            self.src_proj_w = tf.get_variable( "src_proj_w", shape=[ self.embed_size, self.hidden_size ], initializer = self.initializer )
            self.src_proj_b = tf.get_variable( "src_proj_b", shape=[ self.hidden_size ], initializer = self.initializer )

        with tf.variable_scope( "decoder" ) as scope:
            # input projection
            self.tar_embed_matrix = tf.get_variable( "tar_embed_matrix", shape=[ self.tar_maxval, self.embed_size ], initializer = self.initializer )
            self.tar_proj_w = tf.get_variable( "tar_proj_w", shape=[ self.embed_size, self.hidden_size], initializer = self.initializer )
            self.tar_proj_b = tf.get_variable( "tar_proj_b", shape=[ self.hidden_size ], initializer = self.initializer )
            # output projection TODO
            self.proj_w = tf.get_variable( "proj_w", shape = [ self.hidden_size, self.embed_size ], initializer = self.initializer )
            self.proj_b = tf.get_variable( "b", shape = [ self.embed_size ], initializer = self.initializer )
            self.proj_wo = tf.get_variable("proj_wo", shape = [ self.embed_size, self.tar_maxval ], initializer = self.initializer )
            self.proj_bo = tf.get_variable( "proj_bo", shape = [ self.tar_maxval ], initializer = self.initializer )

    def construct_input( self ):
        self.src_input = tf.placeholder( tf.int32, [ self.batch_size, self.src_max_size ], name = "src_input" )
        self.tar_input = tf.placeholder( tf.int32, [ self.batch_size, self.tar_max_size ], name = "tar_input" )
        # sample should always just provide begining of the word
        self.tar_sample = tf.placeholder( tf.int32, [ 1, self.batch_size ] )
        self.tar_prev_state_c = tf.placeholder( tf.float32, [ self.batch_size, self.hidden_size ] )
        self.tar_prev_state_m = tf.placeholder( tf.float32, [ self.batch_size, self.hidden_size ] )
        self.h_s_sample = tf.placeholder( tf.float32, [ self.src_max_size, self.batch_size, self.hidden_size ] )
        self.lr = tf.placeholder( tf.float32, [], name = "lr" )

    def build_attention( self ):
        with tf.variable_scope( "decoder" ) as scope:
            self.w_c = tf.get_variable( "w_c", shape = [ 2 * self.hidden_size, self.hidden_size ], initializer = self.initializer )
            self.b_c = tf.get_variable( "b_c", shape = [ self.hidden_size ], initializer = self.initializer )
            if self.attention_type == 'dot':
                pass
            elif self.attention_type == 'general':
                self.w_a = tf.get_variable( "w_a" shape = [ self.hidden_size, self.hidden_size ], initializer = self.initializer )
            elif self.attention_type == 'concat':
                self.w_a = tf.get_variable( "w_a" shape = [ 2 * self.hidden_size, self.hidden_size ], initializer = self.initializer )
                self.v_a = tf.get_variable( "v_a" shape = [ self.hidden_size, 1 ], initializer = self.initializer )
            else:
                pass

    def build_network( self ):
        with tf.variable_scope( "encoder" ) as scope:
            layer = rnn.LSTMCell( self.hidden_size, state_is_tuple = True )
            layer = rnn.DropoutWrapper( layer, output_keep_prob = ( 1 - self.dropout_rate ) )
            self.encoder = rnn.MultiRNNCell( [ layer ] * self.num_layers, state_is_tuple = True )

        with tf.variable_scope( "decoder" ) as scope:
        	layers = []
        	for i in range( self.num_layers - 1 ):
        		layer = append( rnn.LSTMCell( self.hidden_size, state_is_tuple = True ) )
        		layer = rnn.DropoutWrapper( layer, output_keep_prob = ( 1 - self.dropout_rate ) )
        		layers.append( layer )
            atten_cell = rnn.LSTMCell( self.hidden_size, state_is_tuple = True )
            atten_cell = tf.contrib.seq2seq.AttentionWrapper( atten_cell, atten_mech, attention_size = self.hidden_size )
            atten_cell = rnn.DropoutWrapper( atten_cell, output_keep_prob = ( 1 - self.dropout_rate ) )
            layers.append( atten_cell )
            self.decoder = rnn.MultiRNNCell( layers, state_is_tuple = True )

    def build_graph( self ):
        # embed input for src and tar
        with tf.variable_scope( "encoder", reuse = True ) as scope:
            source_xs = tf.nn.embedding_lookup( self.src_embed_matrix, self.src_input )
            # source_xs = tf.split( source_xs, self.src_max_size, 1 )
            source_xs = tf.matmul( x, self.src_proj_w ) + self.src_proj_b

            source_xs = tf.reshape( source_xs, [self.src_max_size, self.batch_size, self.embed_size ] )
            # ( src_max_size, batch_size, embed_size )
            # for step in range( self.src_max_size ):
            #     # reuse variables for each encoding step
            #     if step > 0: tf.get_variable_scope().reuse_variables()
            #     # x = tf.squeeze( source_xs[ step ], axis = [ 1 ] )
            #     x = source_xs[ step ]
            #     x = tf.matmul( x, self.src_proj_w ) + self.src_proj_b
            #     # x: ( batch_size, hidden size )
            #     e_output, e_state = self.encoder( x, e_state )
            #     h_s.append( e_output )
            # the final hidden state of the encoder

	        # self.e_hidden = e_state
	        # self.h_s = tf.stack( h_s )

	        # hs shape( seq_len, batch_size, hidden_size )
	        self.h_s, self.e_hidden = tf.nn.dynamic_rnn( self.encoder, source_xs, time_major = True, dtype=tf.float32 )


        # this is TODO, should we have zero initial?
        d_state = self.e_hidden
        with tf.variable_scope( "decoder" ) as scope:
            target_xs = tf.nn.embedding_lookup( self.tar_embed_matrix, self.tar_input )
            # target_xs = tf.split( 1, self.tar_max_size, target_xs )
            target_xs = tf.matmul( self.tar_input )
            target_xs = tf.reshape( target_xs, [ self.tar_max_size, self.batch_size, self.embed_size ] )
        #     logit_list, prob_list = [], []
        #     for step in range( self.tar_max_size ):
        #         # reuse variables for each encoding step
        #         if step > 0: tf.get_variable_scope().reuse_variables()
        #         # if step == 0 or ( not self.is_sample ):
        #         #     x = tf.squeeze( target_xs[ step ], [ 1 ] )
        #         x = target_xs[ step ]
        #         x = tf.matmul( x, self.tar_proj_w ) + self.tar_proj_b
        #         h_t, d_state = self.decoder( x, d_state )
        #         h_t_telda = self.attention( h_t, self.h_s )

        #         output_embed = tf.matmul( h_t_telda, self.proj_w ) + self.proj_b
        #         logit = tf.matmul( output_embed, self.proj_wo ) + self.proj_bo
        #         logit_list.append( logit )
        #         prob  = tf.nn.softmax( logit )
        #         prob_list.append( prob )
        # # get rid of the end of sentence mark? TODO
        # logit_list = logit_list
        # logit_list = tf.stack( logit_list )
        # get rid of begining of the sentence? TODO
        # convert target into one hot labels to do softmax loss
        self.train_decode, self.e_hidden = tf.nn.dynamic_rnn( self.decoder, target_xs, time_major = True,
        	                                                  initial_state = self.e_hidden, dtype=tf.float32 )
        tar = tf.one_hot( self.tar_input, self.tar_maxval )
        tar = tf.reshape( tar, [ self.tar_max_size, self.batch_size, self.tar_maxval ] )
        # soft max cross entropy loss
        self.loss  = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits_v2( labels = tar, logits = logit_list ) )
        # ( batch_size, sentence_len, prob embed )
        self.probs = tf.transpose( tf.stack( prob_list ), [ 1, 0, 2 ] )
        self.optimizer = tf.train.GradientDescentOptimizer( self.lr ).minimize( self.loss )

        # construct sampling decoder:

        with tf.variable_scope( "decoder", reuse = True ):
            tar_sample = tf.nn.embedding_lookup( self.tar_embed_matrix, self.tar_sample )
            tar_sample = tf.reshape( tar_sample, [ self.batch_size, self.embed_size ] )
            tar_sample = tf.matmul( tar_sample, self.tar_proj_w ) + self.tar_proj_b
            h_t, d_state = self.decoder( tar_sample, ( self.tar_prev_state_c, self.tar_prev_state_m ) )
            h_t_telda = self.attention( h_t, self.h_s )

            output_embed = tf.matmul( h_t_telda, self.proj_w ) + self.proj_b
            logit = tf.matmul( output_embed, self.proj_wo ) + self.proj_bo
            self.prob_sample = tf.nn.softmax( logit )
            self.d_state_sample = d_state


    def attention( self, h_t, h_s ):
        # first choice of the attention
        scores = tf.reduce_sum( tf.multiply( h_s, h_t ), 2 )
        a_t = tf.nn.softmax( tf.transpose( scores ) )
        a_t = tf.expand_dims( a_t, 2 )
        c_t = tf.matmul( tf.transpose( h_s, perm=[ 1,2,0 ] ), a_t )
        c_t = tf.squeeze( c_t, [ 2 ] )
        h_t_telda = tf.tanh( tf.matmul( tf.concat( [h_t, c_t], axis = 1 ), self.w_c ) + self.b_c )

        return h_t_telda

    def score( self, h_t, h_s ):
        if self.attention_type == 'dot':
            return tf.reduce_sum( tf.mul( h_s, h_t ), 2 )
        elif self.attention_type == 'general':
            return tf.reduce_sum( tf.mul( tf.matmul(h_s, self.w_a), h_t ), 2 )
        elif self.attention_type == 'concat':
            return tf.squeeze( tf.matmul(
                tf.tanh( tf.matmul( tf.concat( h_s, h_t, 1 ), self.w_a ) ),
                self.v_a ) )
        else:
            print( 'incorrect attention type' )
            return

    def train_one_iter( self, src_batch, tar_batch, lr ):
        loss, _ = self.sess.run( [self.loss, self.optimizer ],
                                  feed_dict = { self.src_input: src_batch,
                                                self.tar_input: tar_batch,
                                                self.lr: lr } )
        return loss

    def encode_src( self, src_batch ):
        e_hidden, h_s = self.sess.run( [ self.e_hidden, self.h_s ], feed_dict = {
                                                        self.src_input: src_batch
         } )
        return e_hidden, h_s


    def decode_one_step( self, d_hidden, d_prev_word, h_s ):
        prob, d_hidden = self.sess.run( [ self.prob_sample, self.d_state_sample ],
                                feed_dict = {
                                    self.h_s_sample: h_s,
                                    self.tar_sample: d_prev_word,
                                    self.tar_prev_state_c: d_hidden[ 0 ],
                                    self.tar_prev_state_m: d_hidden[ 1 ]
                                } )
        return prob, d_hidden
