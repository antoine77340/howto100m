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
from collections import defaultdict
from torch.utils.data.dataloader import default_collate

class Youcook_DataLoader(Dataset):
    """Youcook dataset loader."""

    def __init__(
            self,
            data,
            we,
            we_dim=300,
            max_words=30,
            n_pair=6,
    ):
        """
        Args:
        """
        self.data = pickle.load(open(data, 'rb'))
        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words
        self.n_pair = n_pair
        if self.n_pair > 1:
            self.videos = defaultdict(list)
            for i in range(len(self.data)):
                video_id = self.data[i]['id'].split('_')
                video_id = '_'.join(video_id[:-1])
                self.videos[video_id].append(i)
            self.unique_videos = list(self.videos.keys())

    def __len__(self):
        if self.n_pair > 1:
            return len(self.unique_videos)
        else:
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
        if self.n_pair > 1:
            vid = self.unique_videos[idx]
            ind = np.random.choice(self.videos[vid], size=self.n_pair, replace=True)
            video = th.zeros(self.n_pair, 4096)
            caption = th.zeros(self.n_pair, self.max_words, self.we_dim)


        if self.n_pair > 1:
            for i in range(self.n_pair):
                idx_c = ind[i]
                feat_2d = F.normalize(th.from_numpy(self.data[idx_c]['2d']).float(), dim=0)
                feat_3d = F.normalize(th.from_numpy(self.data[idx_c]['3d']).float(), dim=0)
                video[i] = th.cat((feat_2d, feat_3d))
                cap = self.data[idx_c]['caption']
                caption[i] = self._words_to_we(self._tokenize_text(cap))
        else:
            feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d']).float(), dim=0)
            feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d']).float(), dim=0)
            video = th.cat((feat_2d, feat_3d))
            cap = self.data[idx]['caption']
            caption = self._words_to_we(self._tokenize_text(cap))

        return {'video': video, 'text': caption, 'video_id': self.data[idx]['id']}
