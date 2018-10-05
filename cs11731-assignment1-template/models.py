# pytorch construction structure borrow from
# https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable as Variable
# Borrow from PYTORCH tutorial as this is a easy way to do things in cuda
DEVICE = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
USE_CUDA = True

class BaselineGRUEncoder( nn.Module ):

    def __init__( self, input_size, hidden_size, num_layer ):
        super( BaselineGRUEncoder, self ).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.encoder = nn.GRU( input_size, hidden_size, num_layers = num_layer )

    def initial_hidden( self, batch_size ):
        # treat initial state as variable that can be trained
        # initial_state = torch.autograd.Variable( torch.zeros( 1, batch_size, self.hidden_size, device=DEVICE ) )
        initial_state = torch.autograd.Variable( torch.zeros( 1, batch_size, self.hidden_size, device=DEVICE ) )
        return initial_state

    def forward( self, embedded_input, hidden, batch_size ):
        embedded_input = embedded_input.view( 1, batch_size, self.input_size )
        output = embedded_input
        output, hidden = self.encoder( output, hidden )
            # print( "Finish Encoder Layer {}".format( l ) )
            # print( "Output size: {} hidden size: {}".format( output.size(), hidden.size() ) )
        return output, hidden

class BaselineGRUDecoder( nn.Module ):
    def __init__( self, input_size, hidden_size, num_layer, tar_vocab_size, dropout, attention_mode ):
        super( BaselineGRUDecoder, self ).__init__()
        self.input_size = input_size
        self.num_layer = num_layer
        self.inut_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.tar_vocab_size = tar_vocab_size
        self.embedder = nn.Embedding( tar_vocab_size, input_size )
        self.decoder = nn.GRU( input_size, hidden_size, num_layers = num_layer, dropout = dropout )
        self.decode_to_out = nn.Linear( hidden_size, hidden_size )
        self.out = nn.Linear( hidden_size, tar_vocab_size )
        self.softmax = nn.LogSoftmax( dim=1 )
        self.attention = Attention( self.hidden_size, method = attention_mode )

    def decoder_context_init( self, inputs ):
        [ _, batch_size ] = inputs.size()
        return Variable( torch.zeros( 1, batch_size, self.hidden_size ) ).cuda()
    # decoder's input should start from the SOS token
    def forward( self, inputs, hidden, batch_size, last_context, encoder_output ):
        inputs = self.embedder( inputs.view( ( 1, batch_size ) ) )
        seq_len, _, e_hidden_size = encoder_output.size()
        # inputs = inputs.view( 1, batch_size, self.input_size )
        # inputs = inputs.view( batch_size, self.input_size )
        # last_context = last_context.view( ( 1, batch_size, self.hidden_size ) )
        # print( "decoder true input size", self.input_size, self.hidden_size  )
        # print( "output size :", output.size() )
        # print( "Hidden size", hidden.size() )
        output, hidden = self.decoder( inputs, hidden )

        output_logits =  self.out( F.relu( self.decode_to_out( output[ 0 ]  ) ) )
        output = self.softmax( output_logits )
        return output, output_logits, hidden, last_context


class Attention(nn.Module):
    """Attention nn module that is responsible for computing the alignment scores."""

    def __init__(self,  hidden_size, method = 'dot'):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        # Define layers
        if self.method == 'general':
            self.attention = nn.Linear(self.hidden_size, self.hidden_size, bias = False)
        elif self.method == 'concat':
            self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size, bias = False)
            self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_size))


    def forward(self, hidden, encoder_outputs):
        """Attend all encoder inputs conditioned on the previous hidden state of the decoder.
        
        After creating variables to store the attention energies, calculate their 
        values for each encoder output and return the normalized values.
        
        Args:
            hidden: ( 1, input_size )
            encoder_outputs: list of encoder outputs
            
        Returns:
             Normalized (0..1) energy values, re-sized to 1 x 1 x seq_len
        """
        [ _, batch_size, _ ] = hidden.size()
        seq_len = len(encoder_outputs)
        energies = Variable(torch.zeros((seq_len, batch_size)) ).cuda()
        for i in range(seq_len):
            energies[i] = self._score(hidden, encoder_outputs[i])
        return F.softmax(energies, dim = 0)

    def _score(self, hidden, encoder_output):
        """Calculate the relevance of a particular encoder output in respect to the decoder hidden."""
        
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
        elif self.method == 'general':
            energy = self.attention( torch.squeeze( encoder_output, 0 ) )
            # print( "hidden", hidden.size() )
            if len( energy.size() ) == 1:
                energy = energy.unsqueeze( 0 )
            energy = torch.matmul( hidden[0], torch.t( energy )  )
            # torch.matmul( torch.t( hidden ),  energy  )
            energy = torch.diag( energy )
        elif self.method == 'concat':
            energy = self.attention(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
        return energy

class UnidirectionalGRUEncoder( nn.Module ):

    def __init__( self, input_size, hidden_size, num_layer, src_vocab_size, dropout ):
        super( UnidirectionalGRUEncoder, self ).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layer = num_layer
        self.src_vocab_size = src_vocab_size
        self.embedder = nn.Embedding( src_vocab_size , input_size )
        self.encoder = nn.GRU( input_size, hidden_size, num_layers = num_layer, bidirectional = False, dropout = dropout )
        
    def forward( self, embedded_input, hidden, batch_size ):
        output = self.embedder( embedded_input )  
        if hidden is None:
            output, hidden = self.encoder( output )
        else:
            output, hidden = self.encoder( output, hidden )
            # print( "Finish Encoder Layer {}".format( l ) )
            # print( "Output size: {} hidden size: {}".format( output.size(), hidden.size() ) )
        # concat the hidden layers of both direction together and reduce dimension
        # hidden = torch.cat( [ hidden[i] for i in range( self.num_layer * 2) ], dim = 1  )
        # hidden = F.leaky_relu( hidden )
        # hidden = torch.reshape( hidden, ( batch_size, self.hidden_size ) )
        # hidden = hidden.unsqueeze( 0 )
        return output, hidden

class BidirectionalGRUEncoder( nn.Module ):

    def __init__( self, input_size, hidden_size, num_layer, src_vocab_size, dropout ):
        super( BidirectionalGRUEncoder, self ).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layer = num_layer
        self.src_vocab_size = src_vocab_size
        self.embedder = nn.Embedding( src_vocab_size , input_size )
        self.encoder = nn.GRU( input_size, hidden_size, num_layers = num_layer, bidirectional = True, dropout = dropout )
        self.hidden_state_combine_linear = nn.Linear( 2*num_layer * self.hidden_size, self.hidden_size )

        self.output_combine_linear = nn.Linear( 2*num_layer * self.hidden_size, self.hidden_size )


    def forward( self, embedded_input, hidden, batch_size ):
        output = self.embedder( embedded_input )  
        if hidden is None:
            output, hidden = self.encoder( output )
        else:
            output, hidden = self.encoder( output, hidden )
            # print( "Finish Encoder Layer {}".format( l ) )
            # print( "Output size: {} hidden size: {}".format( output.size(), hidden.size() ) )
        # concat the hidden layers of both direction together and reduce dimension
        hidden = torch.cat( [ hidden[i] for i in range( self.num_layer * 2) ], dim = 1  )
        hidden = self.hidden_state_combine_linear( hidden )
        # hidden = F.leaky_relu( hidden )
        hidden = torch.reshape( hidden, ( 1, batch_size, self.hidden_size ) )
        output = self.output_combine_linear( output )
        return output, hidden

class AtttentGRUDecoder( nn.Module ):
    def __init__( self, input_size, hidden_size, num_layer, tar_vocab_size, dropout, attention_mode ):
        super( AtttentGRUDecoder, self ).__init__()
        self.input_size = input_size
        self.num_layer = num_layer
        self.inut_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.tar_vocab_size = tar_vocab_size
        self.embedder = nn.Embedding( tar_vocab_size, input_size )
        self.decoder = nn.GRU( input_size, hidden_size, num_layers = num_layer, dropout = dropout )
        self.out = nn.Linear( hidden_size , tar_vocab_size )
        self.context_to_out = nn.Linear( hidden_size + hidden_size, hidden_size )
        self.softmax = nn.LogSoftmax( dim=1 )
        self.attention = Attention( self.hidden_size, method = attention_mode )


    def decoder_context_init( self, inputs ):
        [ _, batch_size ] = inputs.size()
        return Variable( torch.zeros( 1, batch_size, self.hidden_size ) ).cuda()
    # decoder's input should start from the SOS token
    def forward( self, inputs, hidden, batch_size, last_context, encoder_output ):
        inputs = self.embedder( inputs.reshape( ( 1, batch_size ) ) )
        seq_len, _, e_hidden_size = encoder_output.size()
        output, hidden = self.decoder( inputs, hidden )

        attention_weights = self.attention( output, encoder_output ).permute( 1,0 ).unsqueeze( 1 )
        # ( batch_size, 1, sequence_len )
        # print( "attention weights", attention_weights.size() )
        # print( "encoder output", encoder_output.permute( 1,0,2 ).size() )
        context = torch.bmm( attention_weights, encoder_output.permute( 1, 0, 2 ) )
        # ( batch_size, 1, hidden )
        context = context.permute( 1, 0, 2 )
        # ( 1, batch_size, hidden )
        output = torch.cat( ( output, context ), 2 )
        # output_logits = F.tanh( self.context_to_out( output[ 0 ] ) )
        output_logits = torch.tanh( self.context_to_out( output[ 0 ] ) )
        # ( batch_size, hidden )
        output_logits =  self.out( output_logits )
        output = self.softmax( output_logits )
        # batch_size, vocab_size
        return output, output_logits, hidden, context