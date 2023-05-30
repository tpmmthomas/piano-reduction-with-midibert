import argparse
import numpy as np
import random
import pickle
import os
import copy
import shutil
import json
from sklearn.metrics import confusion_matrix
from cm_fig import save_cm_fig

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import BertConfig

from model import MidiBert
from finetune_trainer import FinetuneTrainer
from finetune_dataset import FinetuneDataset
from finetune_model import TokenClassification, SequenceClassification

from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct


def get_args():
    parser = argparse.ArgumentParser(description='')

    ### mode ###
    parser.add_argument('--task', choices=['melody', 'velocity','composer', 'emotion', 'custom', 'reduction'], required=True)
    
    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='../../dict/CP.pkl')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--case', type=str, default='')

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--class_num', type=int)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
    parser.add_argument("--index_layer", type=int, default=12, help="number of layers")
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    
    ### cuda ###
    parser.add_argument('--cpu', action="store_true")  # default: false
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,1,2,3], help="CUDA device ids")

    args = parser.parse_args()

    root = 'result/finetune/'

    if args.task == 'melody' or args.task=='custom':
        args.class_num = 4
        args.ckpt = root + 'melody_default/model_best.ckpt' if args.ckpt=='' else args.ckpt
    elif args.task=='reduction':
        args.class_num = 3
        args.ckpt = root + 'reduction_/model_best.ckpt' if args.ckpt=='' else args.ckpt
    elif args.task == 'velocity':
        args.class_num = 7
        args.ckpt = root + 'velocity_default/model_best.ckpt' if args.ckpt=='' else args.ckpt
    elif args.task == 'composer':
        args.class_num = 8
        args.ckpt = root + 'composer_default/model_best.ckpt' if args.ckpt=='' else args.ckpt
    elif args.task == 'emotion':
        args.class_num = 4
        args.ckpt = root + 'emotion_default/model_best.ckpt' if args.ckpt=='' else args.ckpt
    elif args.task == 'reduction':
        args.class_num = 3
        args.ckpt = root + 'reduction_/model_best.ckpt' if args.ckpt=='' else args.ckpt

    return args


def load_data(dataset, task, case=None):
    data_root = '../../data/CP/'

    if dataset == 'emotion':
        dataset = 'emopia'

    if dataset =='casestudy':
        X = np.load(os.path.join(data_root, f'{case}.npy'), allow_pickle=True)
        return X
    elif dataset =='custom':
        X = np.load(os.path.join(data_root, f'{dataset}.npy'), allow_pickle=True)
        return X
    else:
        if dataset not in ['pop909', 'composer', 'emopia']:
            print('dataset {} not supported'.format(dataset))
            exit(1)
        
        X_train = np.load(os.path.join(data_root, f'{dataset}_train.npy'), allow_pickle=True)
        X_val = np.load(os.path.join(data_root, f'{dataset}_valid.npy'), allow_pickle=True)
        X_test = np.load(os.path.join(data_root, f'{dataset}_test.npy'), allow_pickle=True)

        print('X_train: {}, X_valid: {}, X_test: {}'.format(X_train.shape, X_val.shape, X_test.shape))

        if dataset == 'pop909':
            y_train = np.load(os.path.join(data_root, f'{dataset}_train_{task[:3]}ans.npy'), allow_pickle=True)
            y_val = np.load(os.path.join(data_root, f'{dataset}_valid_{task[:3]}ans.npy'), allow_pickle=True)
            y_test = np.load(os.path.join(data_root, f'{dataset}_test_{task[:3]}ans.npy'), allow_pickle=True)
        else:
            y_train = np.load(os.path.join(data_root, f'{dataset}_train_ans.npy'), allow_pickle=True)
            y_val = np.load(os.path.join(data_root, f'{dataset}_valid_ans.npy'), allow_pickle=True)
            y_test = np.load(os.path.join(data_root, f'{dataset}_test_ans.npy'), allow_pickle=True)

        print('y_train: {}, y_valid: {}, y_test: {}'.format(y_train.shape, y_val.shape, y_test.shape))

        return X_train, X_val, X_test, y_train, y_val, y_test


def conf_mat(_y, output, task):
    if task == 'melody':
        target_names = ['M','B','A']
        seq = False
    elif task == 'velocity':
        target_names = ['pp','p','mp','mf','f','ff']
        seq = False
    elif task == 'composer':
        target_names = ['M', 'C', 'E','H','W','J','S','Y']
        seq = True
    elif task == 'emotion':
        target_names = ['HAHV', 'HALV', 'LALV', 'LAHV']
        seq = True
        
    output = output.detach().cpu().numpy()
    output = output.reshape(-1,1)
    _y = _y.reshape(-1,1)
    
    
    cm = confusion_matrix(_y, output) 
    print(cm)
    
    _title = 'BERT (CP): ' + task + ' task'
    
    save_cm_fig(cm, classes=target_names, normalize=False,
                title=_title, seq=seq)


def main():
    args = get_args()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nBuilding BERT model")
    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                                position_embedding_type='relative_key_query',
                                hidden_size=args.hs)

    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
    
    print("\nLoading Dataset") 
    if args.task == 'custom':
        dataset='custom'
        model = TokenClassification(midibert, args.class_num, args.hs)
        seq_class = False
    elif args.task == 'melody' or args.task == 'velocity':
        dataset = 'pop909'
        model = TokenClassification(midibert, args.class_num, args.hs)
        seq_class = False
    elif args.task == 'composer' or args.task == 'emotion':
        dataset = args.task
        model = SequenceClassification(midibert, args.class_num, args.hs)
        seq_class = True
    elif args.task=='reduction':
        dataset='casestudy'
        model = TokenClassification(midibert, args.class_num, args.hs)
        seq_class = False
        
    if args.task == 'custom' or args.task=='reduction':
        X = load_data(dataset,args.task,args.case).astype(np.int64)
        y=np.zeros((X.shape[0],X.shape[1])).astype(np.int64) #just place holder, meaningless, bez we r focusing on prediction
        predictset = FinetuneDataset(X=X,y=y)
        predict_loader = DataLoader(predictset, batch_size=args.batch_size, num_workers=args.num_workers)
        print("   len of predict_loader",len(predict_loader))
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = load_data(dataset, args.task)
        trainset = FinetuneDataset(X=X_train, y=y_train)
        validset = FinetuneDataset(X=X_val, y=y_val) 
        testset = FinetuneDataset(X=X_test, y=y_test) 
        train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        print("   len of train_loader",len(train_loader))
        valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
        print("   len of valid_loader",len(valid_loader))
        test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
        print("   len of test_loader",len(test_loader))

    
    print('\nLoad ckpt from', args.ckpt)  
    best_mdl = args.ckpt 
    checkpoint = torch.load(best_mdl, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])


    index_layer = int(args.index_layer)-13
    print("\nCreating Finetune Trainer using index layer", index_layer)
    if args.task=='custom' or args.task=='reduction':        
        a=(X.shape[0],X.shape[1])
        print('!',X.shape,a)
        trainer = FinetuneTrainer(midibert, None,None, predict_loader, index_layer, args.lr, args.class_num,
                                args.hs, a, args.cpu, args.cuda_devices, model, seq_class)
    else:
        print('!',X_test.shape,y_test.shape)
        trainer = FinetuneTrainer(midibert, train_loader, valid_loader, test_loader, index_layer, args.lr, args.class_num,
                                args.hs, y_test.shape, args.cpu, args.cuda_devices, model, seq_class)
  
    
    test_loss, test_acc, all_output = trainer.test()
    if args.task =='custom':
        # print(all_output) #prediction
        
        #visualize
        for ttype in [1,3]:
            out = mid_parser.MidiFile()
            out.ticks_per_beat = 480
            out.instruments = [ct.Instrument(program=0,is_drum=False,name='post-processed piano')]
            current_beat=-1
            for idx1,i in enumerate(all_output):
                for idx2,j in enumerate(i):                    
                    n=X[idx1][idx2]
                    if n[0]==0:
                        current_beat+=1
                        
                    if ttype==3:
                        if all_output[idx1][idx2]==3:
                            out.instruments[0].notes.append(ct.Note(start=current_beat*4*480+n[1]*120,end=current_beat*4*480+n[1]*120+n[3]*60,pitch=n[2]+22,velocity=30))
                    else:
                        if all_output[idx1][idx2]!=3:
                            out.instruments[0].notes.append(ct.Note(start=current_beat*4*480+n[1]*120,end=current_beat*4*480+n[1]*120+n[3]*60,pitch=n[2]+22,velocity=30))
                    
            out.dump('test'+str(ttype)+'.mid')
            print(current_beat)
            
        #reconstruct the original score
        out = mid_parser.MidiFile()
        out.ticks_per_beat = 480
        out.instruments = [ct.Instrument(program=0,is_drum=False,name='post-processed piano')]
        current_beat=-1

        for idx1,i in enumerate(all_output):
            for idx2,j in enumerate(i):                
                n=X[idx1][idx2]
                if n[0]==0:
                    current_beat+=1
                out.instruments[0].notes.append(ct.Note(start=current_beat*4*480+n[1]*120,end=current_beat*4*480+n[1]*120+n[3]*60,pitch=n[2]+22,velocity=30))
                
        out.dump('org.mid')
        print(current_beat)
            
    elif args.task=='reduction':
        ccount=0
        out = mid_parser.MidiFile()
        out.ticks_per_beat = 480
        out.instruments = [ct.Instrument(program=0,is_drum=False,name='reduction')]
        current_beat=-1
        for idx1,i in enumerate(all_output):
            for idx2,j in enumerate(i):                    
                n=X[idx1][idx2]
                if n[0]==0:
                    current_beat+=1
                if all_output[idx1][idx2]==2:
                   ccount+=1
                if all_output[idx1][idx2]==1:
                    out.instruments[0].notes.append(ct.Note(start=current_beat*4*480+n[1]*120,end=current_beat*4*480+n[1]*120+n[3]*60,pitch=n[2]+22,velocity=30))
        out.dump('reduction.mid')
        print(f'reduced {ccount} notes')
        
        out = mid_parser.MidiFile()
        out.ticks_per_beat = 480
        out.instruments = [ct.Instrument(program=0,is_drum=False,name='reduction')]
        current_beat=-1
        for idx1,i in enumerate(all_output):
            for idx2,j in enumerate(i):                    
                n=X[idx1][idx2]
                if n[0]==0:
                    current_beat+=1


                if all_output[idx1][idx2]!=0:
                    out.instruments[0].notes.append(ct.Note(start=current_beat*4*480+n[1]*120,end=current_beat*4*480+n[1]*120+n[3]*60,pitch=n[2]+22,velocity=30))
        out.dump('original.mid')
    else:
        print('test loss: {}, test_acc: {}'.format(test_loss, test_acc))
        conf_mat(y_test, all_output, args.task)

if __name__ == '__main__':
    main()
