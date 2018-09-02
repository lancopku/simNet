import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import numpy as np

#=========================================simNet=========================================
class AttentiveCNN( nn.Module ):
    def __init__( self, hidden_size ):
        super( AttentiveCNN, self ).__init__()
        
        # ResNet-152 backend
        resnet = models.resnet152()
        modules = list( resnet.children() )[ :-2 ] # delete the last fc layer and avg pool.
        resnet_conv = nn.Sequential( *modules ) # last conv feature
        
        self.resnet_conv = resnet_conv
        self.affine_VI = nn.Linear( 2048, hidden_size ) # reduce the dimension
        
        # Dropout before affine transformation
        self.dropout = nn.Dropout( 0.5 )
        
        self.init_weights()

    def init_weights( self ):
        """Initialize the weights."""
        init.kaiming_uniform( self.affine_VI.weight, mode='fan_in' )
        self.affine_VI.bias.data.fill_( 0 )
        
    def forward( self, images ):
        '''
        Input: images
        Output: V=[v_1, ..., v_n], v_g
        '''
        # Last conv layer feature map
        A = self.resnet_conv( images )
        
        # V = [ v_1, v_2, ..., v_49 ]
        V = A.view( A.size( 0 ), A.size( 1 ), -1 ).transpose( 1,2 )
        V = F.relu( self.affine_VI( self.dropout( V ) ) )

        return V

# Encoder Block
class EncoderBlock( nn.Module ):
    def __init__( self, embed_size, hidden_size, vocab_size ):
        super( EncoderBlock, self ).__init__()
        
        self.affine_ZV = nn.Linear(hidden_size, 49)  # W_zv_output_attention
        self.affine_Zh = nn.Linear(hidden_size, 49)  # W_zh_output_attention
        self.affine_alphaz = nn.Linear(49, 1)  # w_alphaz_output_attention

        self.affine_QT = nn.Linear(embed_size, 5)  # W_Qt
        self.affine_Qh = nn.Linear(hidden_size, 5)  # W_Qh
        self.affine_betaq = nn.Linear(5, 1)  # u_betaq

        self.affine_sq = nn.Linear(embed_size, embed_size)  # W_sq
        self.affine_sh = nn.Linear(hidden_size, embed_size)  # W_sh

        self.affine_Ss = nn.Linear(embed_size, 5)  # W_Ss
        self.affine_Sr = nn.Linear(embed_size, 5)  # W_Sr

        self.affine_sz = nn.Linear(hidden_size, embed_size)

        # Final Caption generator
        self.mlp = nn.Linear( embed_size, vocab_size )
        
        # Dropout layer inside Affine Transformation
        self.dropout = nn.Dropout( 0.5 )
        
        self.hidden_size = hidden_size
        self.init_weights()
        
    def init_weights( self ):
        '''
        """Initialize the weights."""
        '''

        init.xavier_uniform(self.affine_ZV.weight)
        self.affine_ZV.bias.data.fill_(0)
        init.xavier_uniform(self.affine_Zh.weight)
        self.affine_Zh.bias.data.fill_(0)
        init.xavier_uniform(self.affine_alphaz.weight)
        self.affine_alphaz.bias.data.fill_(0)

        init.xavier_uniform(self.affine_QT.weight)
        self.affine_QT.bias.data.fill_(0)
        init.xavier_uniform(self.affine_Qh.weight)
        self.affine_Qh.bias.data.fill_(0)
        init.xavier_uniform(self.affine_betaq.weight)
        self.affine_betaq.bias.data.fill_(0)

        init.xavier_uniform(self.affine_sq.weight)
        self.affine_sq.bias.data.fill_(0)
        init.xavier_uniform(self.affine_sh.weight)
        self.affine_sh.bias.data.fill_(0)

        init.xavier_uniform(self.affine_Ss.weight)
        self.affine_Ss.bias.data.fill_(0)
        init.xavier_uniform(self.affine_Sr.weight)
        self.affine_Sr.bias.data.fill_(0)

        init.xavier_uniform(self.affine_sz.weight)
        self.affine_sz.bias.data.fill_(0)

        init.kaiming_normal( self.mlp.weight, mode='fan_in' )
        self.mlp.bias.data.fill_( 0 )
        
    def forward( self, epoch, h_t, V, T ):

        '''
        Input: V=[v_1, v_2, ... v_k], h_t from LSTM and T from Topic Extractor
        Output: A probability indicating how likely the corresponding word in vocabulary D is the current output word
        '''

        # -------------------------Output Attention :z_t_output--------------------------------------------------------------------
        # W_ZV * V + W_Zh * h_t * 1^T
        content_V = self.affine_ZV(self.dropout(V)).unsqueeze(1) + self.affine_Zh(self.dropout(h_t)).unsqueeze(2)

        # visual_t = W_alphaz * tanh( content_V )
        visual_t = self.affine_alphaz(self.dropout(F.tanh(content_V))).squeeze(3)
        alpha_t = F.softmax(visual_t.view(-1, visual_t.size(2))).view(visual_t.size(0), visual_t.size(1), -1)

        z_t = torch.bmm(alpha_t, V).squeeze(2)
        r_t = F.tanh(self.affine_sz(self.dropout(z_t)))

        # -------------------------Topic Attention  :q_t--------------------------------------------------------------------
        content_T = self.affine_QT(self.dropout(T)).unsqueeze(1) + self.affine_Qh(self.dropout(h_t)).unsqueeze(2)

        # topic_t = W_betaq * tanh( content_T )
        topic_t = self.affine_betaq(self.dropout(F.tanh(content_T))).squeeze(3)
        beta_t = F.softmax(topic_t.view(-1, topic_t.size(2))).view(topic_t.size(0), topic_t.size(1), -1)

        q_t = torch.bmm(beta_t, T).squeeze(2)
        s_t = F.tanh(self.affine_sq(self.dropout(q_t)) + self.affine_sh(self.dropout(h_t)))

        # ------------------------------------------Merging Gate----------------------------------------------------
        for ip in range(r_t.size(1)):

            # compute socre_s_t
            s_t_ip = s_t[:, ip, :].contiguous().view(s_t.size(0), 1, s_t.size(2))
            s_t_extended = torch.cat([s_t_ip] * 5, 1)

            content_s_t = self.affine_Ss( s_t_extended ).unsqueeze(1) + self.affine_Qh( h_t ).unsqueeze(2)
            score_s_t = self.affine_betaq( F.tanh( content_s_t ) ).squeeze(3)

            if ip == 0:
                score_s = score_s_t[0][0][0].view(1, 1, 1)
            else:
                score_s = torch.cat([score_s, score_s_t[0][ip][0].view(1, 1, 1)], 1)

            # compute socre_r_t
            r_t_ip = r_t[:, ip, :].contiguous().view(r_t.size(0), 1, r_t.size(2))
            r_t_extended = torch.cat([r_t_ip] * 5, 1)

            content_r_t = self.affine_Sr( r_t_extended ).unsqueeze(1) + self.affine_Qh( h_t ).unsqueeze(2)
            score_r_t = self.affine_betaq( F.tanh( content_r_t ) ).squeeze(3)

            if ip == 0:
                score_r = score_r_t[0][0][0].view(1, 1, 1)
            else:
                score_r = torch.cat([score_r, score_r_t[0][ip][0].view(1, 1, 1)], 1)

        # First train the model without visual attention for 15 epoch
        if epoch <= 20:
            gama_t = 1.0
        else:
            gama_t = F.sigmoid(score_s - score_r)
        
        # Final score along vocabulary
        #scores = self.mlp( self.dropout( c_t ) )
        c_t = gama_t * s_t + (1-gama_t) * r_t
        scores = self.mlp( self.dropout( c_t ) )

        return scores

# Caption Decoder
class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size):
        super(Decoder, self).__init__()

        # word embedding
        self.embed = nn.Embedding(vocab_size, embed_size)
        # LSTM decoder: input = [ w_t; v_input ] => 2 x word_embed_size;
        self.LSTM = nn.LSTM(embed_size * 2, hidden_size, 1, batch_first=True)

        # Save hidden_size for hidden and cell variable
        self.hidden_size = hidden_size

        # Encoder Block: Final scores for caption sampling
        self.encoder = EncoderBlock(embed_size, hidden_size, vocab_size)
        
        # reduce the feature map dimension
        self.affine_b = nn.Linear( hidden_size, embed_size )

        # input_attention weights
        self.affine_ZV_input = nn.Linear(embed_size, 49 )  # W_ZV_input_attention
        self.affine_Zh_input = nn.Linear(hidden_size, 49 )  # W_Zh_input_attention
        self.affine_alphaz_input = nn.Linear(49, 1 )  # w_alphaz_input_attention

        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        init.kaiming_uniform( self.affine_b.weight, mode='fan_in' )
        self.affine_b.bias.data.fill_( 0 )

        init.xavier_uniform(self.affine_ZV_input.weight)
        self.affine_ZV_input.bias.data.fill_( 0 )
        init.xavier_uniform(self.affine_Zh_input.weight)
        self.affine_Zh_input.bias.data.fill_( 0 )
        init.xavier_uniform(self.affine_alphaz_input.weight)
        self.affine_alphaz_input.bias.data.fill_( 0 )

    def forward(self, epoch, V, captions, T, states=None):
        
        # Reduce the feature map dimension
        V_input = F.relu( self.affine_b( self.dropout( V ) ) )
        v_g = torch.mean( V_input,dim=1 )

        # Word Embedding
        embeddings = self.embed( captions )

        # Topic Embedding
        T = self.embed( T )

        # x_t = embeddings
        x = embeddings

        # Hiddens: Batch x seq_len x hidden_size
        if torch.cuda.is_available():
            hiddens = Variable(torch.zeros(x.size(0), x.size(1), self.hidden_size).cuda())
        else:
            hiddens = Variable(torch.zeros(x.size(0), x.size(1), self.hidden_size))

        # Recurrent Block
        for time_step in range(x.size(1)):

            # Feed in x_t one at a time
            x_t = x[:, time_step, :]
            x_t = x_t.unsqueeze(1)

            #-----input attention-----
            if time_step == 0:
                x_t = torch.cat((x_t, v_g.unsqueeze(1).expand_as(x_t)), dim=2)
            else:
                # W_ZV * V + W_Zh * h_t * 1^T
                content_v_input = self.affine_ZV_input(self.dropout(V_input)).unsqueeze(1) + self.affine_Zh_input(self.dropout(h_t)).unsqueeze(2)

                # visual_t = W_alphaz * tanh( content_v_input )
                visual_t_input = self.affine_alphaz_input(self.dropout(F.tanh(content_v_input))).squeeze(3)
                alpha_t_input = F.softmax(visual_t_input.view(-1, visual_t_input.size(2))).view(visual_t_input.size(0),visual_t_input.size(1), -1)
                z_t_input = torch.bmm(alpha_t_input, V_input).squeeze(2)

                #x_t =[embeddings;z_t_input]
                x_t = torch.cat((x_t, z_t_input), dim=2)

            h_t, states = self.LSTM(x_t, states)

            # Save hidden
            hiddens[:, time_step, :] = h_t

        # Data parallelism for Encoder block
        if torch.cuda.device_count() > 1:
            device_ids = range( torch.cuda.device_count() )
            encoder_block_parallel = nn.DataParallel( self.encoder, device_ids=device_ids )
            scores = encoder_block_parallel( epoch, hiddens, V, T )
        else:
            scores = self.encoder( epoch, hiddens, V, T )

        # Return states for Caption Sampling purpose
        return scores, states

# Whole Architecture with Image Encoder and Caption decoder        
class Encoder2Decoder( nn.Module ):
    def __init__( self, embed_size, vocab_size, hidden_size ):
        super( Encoder2Decoder, self ).__init__()
        
        # Image CNN encoder and simNet Decoder
        self.encoder = AttentiveCNN( hidden_size )
        self.decoder = Decoder( embed_size, vocab_size, hidden_size )

    def forward( self, epoch, images, captions, lengths, T ):

        if torch.cuda.device_count() > 1:
            device_ids = range( torch.cuda.device_count() )
            encoder_parallel = torch.nn.DataParallel( self.encoder, device_ids=device_ids )
            V = encoder_parallel( images ) 
        else:
            V = self.encoder( images )

        # Language Modeling on word prediction
        scores, _ = self.decoder( epoch, V, captions, T )

        # Pack it to make criterion calculation more efficient
        packed_scores = pack_padded_sequence( scores, lengths, batch_first=True )

        return packed_scores
    
    # Caption generator
    def sampler( self, epoch, images, T, max_len=20 ):
        """
        Samples captions for given image features.
        """
        
        # Data parallelism if multiple GPUs
        if torch.cuda.device_count() > 1:
            device_ids = range( torch.cuda.device_count() )
            encoder_parallel = torch.nn.DataParallel( self.encoder, device_ids=device_ids )
            V = encoder_parallel( images ) 
        else:    
            V = self.encoder( images )
            
        # Build the starting token Variable <start> (index 1): B x 1
        if torch.cuda.is_available():
            captions = Variable( torch.LongTensor( images.size( 0 ), 1 ).fill_( 1 ).cuda() )
        else:
            captions = Variable( torch.LongTensor( images.size( 0 ), 1 ).fill_( 1 ) )

        # Get generated caption idx list
        sampled_ids = []

        # Initial hidden states
        states = None

        for i in range( max_len ):
            scores, states = self.decoder( epoch, V, captions, T, states )
            captions = scores.max( 2 )[ 1 ]

            # Save sampled word
            sampled_ids.append( captions )

        # caption: B x max_len
        sampled_ids = torch.cat( sampled_ids, dim=1 )

        return sampled_ids
