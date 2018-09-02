import math
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from model import Encoder2Decoder
from build_vocab import Vocabulary
from torch.autograd import Variable 
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence

def to_var( x, volatile=False ):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable( x, volatile=volatile )

def main( args ):
    # To reproduce training results
    torch.manual_seed( args.seed )
    if torch.cuda.is_available():
        torch.cuda.manual_seed( args.seed )
 
    # Create model directory
    if not os.path.exists( args.model_path ):
        os.makedirs( args.model_path )
    
    # Image Preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([ 
        transforms.RandomCrop( args.crop_size ),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(( 0.485, 0.456, 0.406 ), 
                             ( 0.229, 0.224, 0.225 ))])
    
    # Load vocabulary wrapper.
    with open( args.vocab_path, 'rb') as f:
        vocab = pickle.load( f )

    # Load pretrained model or build from scratch
    simNet = Encoder2Decoder( args.embed_size, len( vocab ), args.hidden_size )
    
    if args.pretrained:
        simNet.load_state_dict( torch.load( args.pretrained ) )
        # Get starting epoch #, note that model is named as '...your path to model/algoname-epoch#.pkl'
        # A little messy here.
        start_epoch = int( args.pretrained.split('/')[-1].split('-')[1].split('.')[0] ) + 1
        
    elif args.pretrained_cnn:
        pretrained_dict = torch.load( args.pretrained_cnn )
        model_dict=simNet.state_dict()
        
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update( pretrained_dict )
        simNet.load_state_dict( model_dict )
        
        start_epoch = 1
        
    else:
        start_epoch = 1

    # Parameter optimization
    params = list( simNet.encoder.affine_VI.parameters() )  + list( simNet.decoder.parameters() )
    
    # Will decay later    
    learning_rate = args.learning_rate
    
    # Language Modeling Loss
    LMcriterion = nn.CrossEntropyLoss()
    
    # Change to GPU mode if available
    if torch.cuda.is_available():
        simNet.cuda()
        LMcriterion.cuda()
    # Load image_topic
    topic = json.load( open( args.topic_path , 'r' ) )
    # Build training data loader
    data_loader = get_loader(args.image_dir, args.caption_path, topic, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    # Train the Models
    total_step = len( data_loader )
    
    # Start Training 
    for epoch in range( start_epoch, args.num_epochs + 1 ):
        if epoch == args.visual_attention_epoch:
            print 'Starting Training Visual Attention'
            
        # Start Learning Rate Decay
        if epoch > args.lr_decay:
                
            frac = float( epoch - args.lr_decay ) / args.learning_rate_decay_every
            decay_factor = math.pow( 0.5, frac )

            # Decay the learning rate
            learning_rate = args.learning_rate * decay_factor
        
        print 'Learning Rate for Epoch %d: %.6f'%( epoch, learning_rate )

        optimizer = torch.optim.Adam( params, lr=learning_rate, betas=( args.alpha, args.beta ) )

        # Language Modeling Training
        print '------------------Training for Epoch %d----------------'%( epoch )
        for i, ( images, captions, lengths, _, _, T ) in enumerate( data_loader ):

            # Set mini-batch dataset
            images = to_var( images )
            captions = to_var( captions )
            T = to_var( T )
            lengths = [ cap_len - 1  for cap_len in lengths ]
            targets = pack_padded_sequence( captions[:,1:], lengths, batch_first=True )[0]

            # Forward, Backward and Optimize
            simNet.train()
            simNet.zero_grad()

            packed_scores = simNet( epoch, images, captions, lengths, T )

            # Compute loss and backprop
            loss = LMcriterion( packed_scores[0], targets )
            loss.backward()
            
            # Gradient clipping for gradient exploding problem in LSTM
            for p in simNet.decoder.LSTM.parameters():
                p.data.clamp_( -args.clip, args.clip )
            
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print 'Epoch [%d/%d], Step [%d/%d], CrossEntropy Loss: %.4f'%( epoch, args.num_epochs, i, total_step, loss.data[0] )
                
        # Save the simNet model after each epoch
        torch.save( simNet.state_dict(), 
                    os.path.join( args.model_path, 
                    'simNet-%d.pkl'%( epoch ) ) )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '-f', default='self', help='To make it runnable in jupyter' )
    parser.add_argument( '--model_path', type=str, default='./models',
                         help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/coco2014' ,
                        help='directory for training images')
    parser.add_argument('--caption_path', type=str,
                        default='./data/annotations/karpathy_split_train.json',
                        help='path for train annotation json file')
    parser.add_argument('--topic_path', type=str,
                        default='./data/topics/image_topic.json',
                        help='path for image topic json file')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing log info')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed for model reproduction')
    
    # ---------------------------Hyper Parameter Setup------------------------------------
    # Optimizer Adam parameter
    parser.add_argument( '--alpha', type=float, default=0.8,
                         help='alpha in Adam' )
    parser.add_argument( '--beta', type=float, default=0.999,
                         help='beta in Adam' )
    parser.add_argument( '--learning_rate', type=float, default=4e-4,
                         help='learning rate for the whole model' )
    
    # LSTM hyper parameters
    parser.add_argument( '--embed_size', type=int, default=256,
                         help='dimension of word embedding vectors' )
    parser.add_argument( '--hidden_size', type=int, default=512,
                         help='dimension of lstm hidden states' )
    
    # Training details
    parser.add_argument( '--pretrained', type=str, default='', help='start from checkpoint or scratch' )
    parser.add_argument( '--pretrained_cnn', type=str, default='models/pretrained_cnn.pkl', help='load pertraind_cnn parameters' )
    parser.add_argument( '--num_epochs', type=int, default=30 )
    parser.add_argument( '--batch_size', type=int, default=80 )
    parser.add_argument( '--num_workers', type=int, default=4 )
    parser.add_argument( '--clip', type=float, default=0.1 )
    parser.add_argument( '--visual_attention_epoch', type=int, default=20, help='epoch at which to start training visual_attention' )
    parser.add_argument( '--lr_decay', type=int, default=20, help='epoch at which to start lr decay' )
    parser.add_argument( '--learning_rate_decay_every', type=int, default=50,
                         help='decay learning rate at every this number')

    
    args = parser.parse_args()
    
    print '------------------------Model and Training Details--------------------------'
    print(args)
    
    # Start training
    main( args )
