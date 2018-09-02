import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import glob
import pickle
from build_vocab import Vocabulary
from model import Encoder2Decoder
from torch.autograd import Variable 
from torchvision import transforms, datasets
from coco.pycocotools.coco import COCO
from coco.pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt

# Variable wrapper
def to_var(x, volatile=False):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable( x, volatile=volatile )

# MS COCO evaluation data loader
class CocoEvalLoader( datasets.ImageFolder ):
    def __init__( self, root, ann_path, topic_path, transform=None, target_transform=None,
                 loader=datasets.folder.default_loader ):
        '''
        Customized COCO loader to get Image ids and Image Filenames
        root: path for images
        ann_path: path for the annotation file (e.g., caption_val2014.json)
        '''
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.imgs = json.load( open( ann_path, 'r' ) )['images']
        self.image_topic = json.load(open( topic_path , 'r'))

    def __getitem__(self, index):

        filename = self.imgs[ index ]['file_name']
        img_id = self.imgs[ index ]['id']
        
        # Filename for the image
        if 'val2014' in filename.lower():
            path = os.path.join( self.root, 'val2014' , filename )
        elif 'train2014' in filename.lower():
            path = os.path.join( self.root, 'train2014' , filename )
        else:
            path = os.path.join( self.root, 'test2014', filename )
            
        # Load the vocabulary
        with open( 'vocab.pkl', 'rb' ) as f:
            vocab = pickle.load( f )

        img = self.loader( path )
        if self.transform is not None:
            img = self.transform( img )
            
        # Load the image topic
        T_val = []
        for topic in self.image_topic:
            if topic['image_id'] == img_id:
                image_topic = topic['image_concepts']
                T_val.extend([vocab(token) for token in image_topic])
                break
                
        T_val = torch.LongTensor(T_val)
        
        return img, img_id, filename, T_val

# MSCOCO Evaluation function
def main( args ):
    
    '''
    model: trained model to be evaluated
    args: parameters
    '''
    # Load vocabulary wrapper.
    with open( args.vocab_path, 'rb') as f:
        vocab = pickle.load( f )
    # Load trained model
    model = Encoder2Decoder( args.embed_size, len(vocab), args.hidden_size )
    model.load_state_dict(torch.load(args.trained))

    # Change to GPU mode if available
    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    
    transform = transforms.Compose([ 
        transforms.Resize( (args.crop_size, args.crop_size) ),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    # Wrapper the COCO VAL dataset
    eval_data_loader = torch.utils.data.DataLoader( 
        CocoEvalLoader( args.image_dir, args.caption_test_path, args.topic_path, transform ),
        batch_size = args.eval_size, 
        shuffle = False, num_workers = args.num_workers,
        drop_last = False )  
    epoch = int( args.trained.split('/')[-1].split('-')[1].split('.')[0] )
    
    # Generated captions to be compared with GT
    results = []
    print '---------------------Start evaluation on MS-COCO dataset-----------------------'
    for i, (images, image_ids, _, T_val ) in enumerate( eval_data_loader ):
        
        images = to_var( images )
        T_val = to_var( T_val )
        generated_captions = model.sampler( epoch, images, T_val )

        if torch.cuda.is_available():
            captions = generated_captions.cpu().data.numpy()
        else:
            captions = generated_captions.data.numpy()

        # Build caption based on Vocabulary and the '<end>' token
        for image_idx in range( captions.shape[0] ):
            
            sampled_ids = captions[ image_idx ]
            sampled_caption = []
            
            for word_id in sampled_ids:
                
                word = vocab.idx2word[ word_id ]
                if word == '<end>':
                    break
                else:
                    sampled_caption.append( word )
            
            sentence = ' '.join( sampled_caption )
            
            temp = { 'image_id': int( image_ids[ image_idx ] ), 'caption': sentence}
            results.append( temp )
        
        # Disp evaluation process
        if (i+1) % 10 == 0:
            print '[%d/%d]'%( (i+1),len( eval_data_loader ) ) 

    print '------------------------Caption Generated-------------------------------------'
            
    # Evaluate the results based on the COCO API
    resFile = args.save_path
    json.dump( results, open( resFile , 'w' ) )
    
    annFile = args.caption_test_path
    coco = COCO( annFile )
    cocoRes = coco.loadRes( resFile )
    
    cocoEval = COCOEvalCap( coco, cocoRes )
    cocoEval.params['image_id'] = cocoRes.getImgIds() 
    cocoEval.evaluate()

    print '-----------Evaluation performance on MS-COCO dataset----------'
    for metric, score in cocoEval.eval.items():
        print '%s: %.4f'%( metric, score )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='self', help='To make it runnable in jupyter')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/coco2014',
                        help='directory for resized training images')
    parser.add_argument('--caption_test_path', type=str,
                        default='./data/annotations/karpathy_split_test.json',
                        help='path for test annotation json file')
    parser.add_argument('--topic_path', type=str,
                        default='./data/topics/image_topic.json',
                        help='path for test topic json file')

    # ---------------------------Hyper Parameter Setup------------------------------------
    parser.add_argument('--save_path', type=str, default='model_generated_caption.json')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors, also dimension of v_g')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--trained', type=str, default='./models/simNet-30.pkl',
                        help='start from checkpoint or scratch')
    parser.add_argument('--eval_size', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    print '------------------------Model and Testing Details--------------------------'
    print(args)

    # Start training
    main(args)
