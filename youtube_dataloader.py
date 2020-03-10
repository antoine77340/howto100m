from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
import re
import random


class Youtube_DataLoader(Dataset):
    """Youtube dataset loader."""

    def __init__(
            self,
            csv,
            features_path,
            caption,
            we,
            min_time=10.0,
            features_path_3D=None,
            feature_framerate=1.0,
            feature_framerate_3D=24.0 / 16.0,
            we_dim=300,
            max_words=30,
            min_words=0,
            n_pair=1,
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(csv)
        self.features_path_2D = features_path
        self.features_path_3D = features_path_3D
        self.caption = caption
        self.min_time = min_time
        self.feature_framerate = feature_framerate
        self.feature_framerate_3D = feature_framerate_3D
        self.we_dim = we_dim
        self.max_words = max_words
        self.min_words = min_words
        self.we = we
        self.n_pair = n_pair
        self.fps = {'2d': feature_framerate, '3d': feature_framerate_3D}
        self.feature_path = {'2d': features_path}
        if features_path_3D != '':
            self.feature_path['3d'] = features_path_3D

    def __len__(self):
        return len(self.csv)

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

    def _get_text(self, caption, n_pair_max):
        n_caption = len(caption['start'])
        k = n_pair_max
        starts = np.zeros(k)
        ends = np.zeros(k)
        text = th.zeros(k, self.max_words, self.we_dim)
        r_ind = np.random.choice(range(n_caption), k, replace=True)

        for i in range(k):
            ind = r_ind[i]
            text[i], starts[i], ends[i] = self._get_single_text(caption, ind)

        return text, starts, ends


    def _get_single_text(self, caption, ind):
        start, end = ind, ind
        words = self._tokenize_text(caption['text'][ind])
        diff = caption['end'][end] - caption['start'][start]
        while len(words) < self.min_words or diff < self.min_time:
            if start > 0 and end < len(caption['end']) - 1:
                next_words = self._tokenize_text(caption['text'][end + 1])
                prev_words = self._tokenize_text(caption['text'][start - 1])
                d1 = caption['end'][end + 1] - caption['start'][start]
                d2 = caption['end'][end] - caption['start'][start - 1]
                if (self.min_time > 0 and d2 <= d1) or \
                    (self.min_time == 0 and len(next_words) <= len(prev_words)):
                    start -= 1
                    words.extend(prev_words)
                else:
                    end += 1
                    words.extend(next_words)
            elif start > 0:
                words.extend(self._tokenize_text(caption['text'][start - 1]))
                start -= 1
            elif end < len(caption['end']) - 1:
                words.extend(self._tokenize_text(caption['text'][end + 1]))
                end += 1
            else:
                break
            diff = caption['end'][end] - caption['start'][start]
        return self._words_to_we(words), \
            caption['start'][start], caption['end'][end]


    def _get_video(self, vid_path, s, e):
        feature_path = {}
        video = {}
        output = {}
        dim = 0
        for k in self.feature_path:
            feature_path[k] = os.path.join(self.feature_path[k], vid_path)
            video[k] = th.from_numpy(np.load(feature_path[k])).float()
            output[k] = th.zeros(len(s), video[k].shape[-1])

            for i in range(len(s)):
                start = int(s[i] * self.fps[k])
                end = int(e[i] * self.fps[k]) + 1
                slice = video[k][start:end]
                if len(slice) < 1:
                    print("video_id: {}, start: {}, end: {}".format(
                        feature_path[k], start, end))
                else:
                    output[k][i] = F.normalize(th.max(slice, dim=0)[0], dim=0)

        return th.cat([output[k] for k in output], dim=1)


    def __getitem__(self, idx):
        video_id = self.csv['video_id'].values[idx]
        task = str(self.csv['task'].values[idx])
        vid_path = self.csv['path'].values[idx]
        text, starts, ends = self._get_text(self.caption[video_id], self.n_pair)
        video = self._get_video(vid_path, starts, ends)
        return {'video': video, 'text': text, 'video_id': video_id}
