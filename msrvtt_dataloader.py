from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import Dataset
import pickle
import torch.nn.functional as F
import numpy as np
import re
import pandas as pd
from collections import defaultdict
from torch.utils.data.dataloader import default_collate
import json
import random

class MSRVTT_DataLoader(Dataset):
    """MSRVTT dataset loader."""

    def __init__(
            self,
            csv_path,
            features_path,
            we,
            we_dim=300,
            max_words=30,
    ):
        """
        Args:
        """
        self.data = pd.read_csv(csv_path)
        self.features = th.load(features_path)
        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words
        self.max_video = 30

    def __len__(self):
        return len(self.data)

    def custom_collate(self, batch):
        return default_collate(batch)

    def _zero_pad_tensor(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
            return np.concatenate((tensor, zero), axis=0)

    def _tokenize_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_we(self, words):
        words = [word for word in words if word in self.we.vocab]
        if words:
            we = self._zero_pad_tensor(self.we[words], self.max_words)
            return th.from_numpy(we)
        else:
            return th.zeros(self.max_words, self.we_dim)

    def __getitem__(self, idx):
        vid = self.data['video_id'].values[idx]
        sentence = self.data['sentence'].values[idx]
        caption = self._words_to_we(self._tokenize_text(sentence))
        feat_2d = F.normalize(self.features['2d'][vid].float(), dim=0)
        feat_3d = F.normalize(self.features['3d'][vid].float(), dim=0)
        video = th.cat((feat_2d, feat_3d))
        return {'video': video, 'text': caption, 'video_id': vid}

        return {'2d': feat_2d, '3d': feat_3d, 'text': caption, 'video_id': vid}



class MSRVTT_TrainDataLoader(Dataset):
    """MSRVTT train dataset loader."""

    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            we,
            we_dim=300,
            max_words=30,
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.features = th.load(features_path)
        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words
        self.sentences = defaultdict(list)
        for s in self.data['sentences']:
            self.sentences[s['video_id']].append(s['caption'])

    def __len__(self):
        return len(self.csv)

    def custom_collate(self, batch):
        return default_collate(batch)

    def _zero_pad_tensor(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
            return np.concatenate((tensor, zero), axis=0)

    def _tokenize_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_we(self, words):
        words = [word for word in words if word in self.we.vocab]
        if words:
            we = self._zero_pad_tensor(self.we[words], self.max_words)
            return th.from_numpy(we)
        else:
            return th.zeros(self.max_words, self.we_dim)

    def __getitem__(self, idx):
        vid = self.csv['video_id'].values[idx]
        rind = random.randint(0, len(self.sentences[vid]) - 1)
        sentence = self.sentences[vid][rind]
        feat_2d = F.normalize(self.features['2d'][vid].float(), dim=0)
        feat_3d = F.normalize(self.features['3d'][vid].float(), dim=0)
        video = th.cat((feat_2d, feat_3d))
        caption = self._words_to_we(self._tokenize_text(sentence))
        return {'video': video, 'text': caption}
