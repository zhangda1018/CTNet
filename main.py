# -*- coding: utf-8 -*-
"""
@author: iopenzd
"""
import utils, argparse, warnings, sys, shutil, torch, os, numpy as np, math
warnings.filterwarnings("ignore")
from data.Multi.datautils import *
from train import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ArticularyWordRecognition')
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--nlayers', type=int, default=4)
parser.add_argument('--emb_size', type=int, default=128)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--task_rate', type=float, default=0.5)  # balance rate
parser.add_argument('--masking_ratio', type=float, default=0.05)
parser.add_argument('--lamb', type=float, default=0.8)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--ratio_highest_attention', type=float, default=0.5) # thresholding
parser.add_argument('--avg', type=str, default='macro')  
parser.add_argument('--dropout', type=float, default=0.01)
parser.add_argument('--nhid', type=int, default=128)
parser.add_argument('--nhid_task', type=int, default=128)
parser.add_argument('--nhid_re', type=int, default=128)
parser.add_argument('--task_type', type=str, default='classification')
parser.add_argument('--Path', type=str,
                    default=r'', )
args = parser.parse_args()


# Multi:
# ArticularyWordRecognition, AtrialFibrillation, BasicMotions, CharacterTrajectories, Cricket, DuckDuckGeese, EigenWorms, Epilepsy, EthanolConcentration
# ERing, FaceDetection, FingerMovements, HandMovementDirection, Handwriting, Heartbeat, InsectWingbeat, JapaneseVowels, Libras, LSST, MotorImagery,
# NATOPS, PenDigits, PEMS-SF, Phoneme, RacketSports, SelfRegulationSCP1, SelfRegulationSCP2, SpokenArabicDigits, StandWalkJump, UWaveGestureLibrary

def main():
    prop = utils.get_prop(args)

    print('Data loading start...')
    if prop['task_type'] == 'classification':
        X_train, y_train, X_test, y_test = load_UCR(prop['Path'], prop['dataset'])
        print('Data loading complete...')

    print('Data preprocessing start...')
    X_train_cls, y_train_cls, X_test, y_test = utils.preprocess(prop, X_train, y_train, X_test, y_test)
    print(X_train_cls.shape, y_train_cls.shape, X_test.shape, y_test.shape)
    print('Data preprocessing complete...')

    prop['nclasses'] = torch.max(y_train_cls).item() + 1 if prop['task_type'] == 'classification' else None
    prop['dataset'], prop['seq_len'], prop['input_size'] = prop['dataset'], X_train_cls.shape[1], X_train_cls.shape[2]
    prop['device'] = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    print('Initializing model...')
    model, optimizer, criterion_re, criterion_task, best_model, best_optimizer = utils.initialize_training(prop)
    print('Model intialized...')

    print('Training start...')
    utils.training(model, optimizer, criterion_re, criterion_task, best_model, best_optimizer, X_train_cls, y_train_cls, X_test, y_test, prop)
    print('Training complete...')


if __name__ == "__main__":
    main()
