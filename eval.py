from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import DataLoader
import numpy as np
from args import get_args
import random
import os
from youcook_dataloader import Youcook_DataLoader
from model import Net
from metrics import compute_metrics, print_computed_metrics
from gensim.models.keyedvectors import KeyedVectors
import pickle
import glob
from lsmdc_dataloader import LSMDC_DataLoader
from msrvtt_dataloader import MSRVTT_DataLoader


args = get_args()
if args.verbose:
    print(args)

assert args.pretrain_path != '', 'Need to specify pretrain_path argument'

# predefining random initial seeds
th.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

print('Loading word vectors: {}'.format(args.word2vec_path))
we = KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
print('done')


if args.eval_youcook:
    dataset_val = Youcook_DataLoader(
        data=args.youcook_val_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        n_pair=1,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
    )
if args.eval_lsmdc:
    dataset_lsmdc = LSMDC_DataLoader(
        csv_path=args.lsmdc_test_csv_path,
        features_path=args.lsmdc_test_features_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
    )
    dataloader_lsmdc = DataLoader(
        dataset_lsmdc,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
    )
if args.eval_msrvtt:
    msrvtt_testset = MSRVTT_DataLoader(
        csv_path='/sequoia/data2/dataset/MSR-VTT_Dataset/test_sentences.csv',
        features_path=args.msrvtt_test_features_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=3000,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
net = Net(
    video_dim=args.feature_dim,
    embd_dim=args.embd_dim,
    we_dim=args.we_dim,
    max_words=args.max_words,
)
net.eval()
# Optimizers + Loss
if args.gpu_mode:
    net.cuda()

if args.verbose:
    print('Starting evaluation loop ...')

def Eval_msrvtt(model, eval_dataloader):
    model.eval()
    print ('Evaluating Text-Video retrieval on MSRVTT data')
    with th.no_grad():
        for i_batch, data in enumerate(eval_dataloader):
            text = data['text'].cuda() if args.gpu_mode else data['text']
            vid = data['video_id']
            video = data['video'].cuda() if args.gpu_mode else data['video']
            m = model(video, text)
            m = m.cpu().detach().numpy()
            metrics = compute_metrics(m)
            print_computed_metrics(metrics)

def Eval_lsmdc(model, eval_dataloader):
    model.eval()
    print ('Evaluating Text-Video retrieval on LSMDC data')
    with th.no_grad():
        for i_batch, data in enumerate(eval_dataloader):
            text = data['text'].cuda() if args.gpu_mode else data['text']
            video = data['video'].cuda() if args.gpu_mode else data['video']
            vid = data['video_id']
            m = model(video, text)
            m = m.cpu().detach().numpy()
            metrics = compute_metrics(m)
            print_computed_metrics(metrics)


def Eval_youcook(model, eval_dataloader):
    model.eval()
    print ('Evaluating Text-Video retrieval on Youcook data')
    with th.no_grad():
        for i_batch, data in enumerate(eval_dataloader):
            text = data['text'].cuda() if args.gpu_mode else data['text']
            video = data['video'].cuda() if args.gpu_mode else data['video']
            vid = data['video_id']
            m = model(video, text)
            m  = m.cpu().detach().numpy()
            metrics = compute_metrics(m)
            print_computed_metrics(metrics)

all_checkpoints = glob.glob(args.pretrain_path)

for c in all_checkpoints:
    print('Eval checkpoint: {}'.format(c))
    print('Loading checkpoint: {}'.format(c))
    net.load_checkpoint(c)
    if args.eval_youcook:
        Eval_youcook(net, dataloader_val)
    if args.eval_msrvtt:
        Eval_msrvtt(net, dataloader_msrvtt)
    if args.eval_lsmdc:
        Eval_lsmdc(net, dataloader_lsmdc)
