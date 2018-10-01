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

# Borrow from PYTORCH tutorial as this is a easy way to do things in cuda
DEVICE = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
USE_CUDA = True

class BaselineGRUEncoder( nn.Module ):

    def __init__( self, input_size, hidden_size, num_layer ):
        super( BaselineGRUEncoder, self ).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.encoder = nn.GRU( input_size, hidden_size )

    def initial_hidden( self, batch_size ):
        # treat initial state as variable that can be trained
        # initial_state = torch.autograd.Variable( torch.zeros( 1, batch_size, self.hidden_size, device=DEVICE ) )
        initial_state = torch.autograd.Variable( torch.zeros( 1, batch_size, self.hidden_size, device=DEVICE ) )
        return initial_state

    def forward( self, embedded_input, hidden, batch_size ):
        embedded_input = embedded_input.view( 1, batch_size, self.input_size )
        output = embedded_input
        for l in range( self.num_layer ):
            output, hidden = self.encoder( output, hidden )
            # print( "Finish Encoder Layer {}".format( l ) )
            # print( "Output size: {} hidden size: {}".format( output.size(), hidden.size() ) )
        return output, hidden

class BaselineGRUDecoder( nn.Module ):
    def __init__( self, input_size, hidden_size, output_size, num_layer ):
        super( BaselineGRUDecoder, self ).__init__()
        self.input_size = input_size
        self.num_layer = num_layer
        self.decoder = nn.GRU( input_size, hidden_size )
        self.out = nn.Linear( hidden_size, output_size )
        self.softmax = nn.LogSoftmax( dim=1 )

    # decoder's input should start from the SOS token
    def forward( self, embedded_input, hidden, batch_size ):
        embedded_input = embedded_input.view( 1, batch_size, self.input_size )
        output = embedded_input
        for _ in range( self.num_layer ):
            output = F.relu( output )
            output, hidden = self.decoder( output, hidden )
        output = self.softmax( self.out( output[ 0 ] ) )
        return output, hidden
