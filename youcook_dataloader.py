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
    ):
        """
        Args:
        """
        self.data = pickle.load(open(data, 'rb'))
        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words

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
        feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d']).float(), dim=0)
        feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d']).float(), dim=0)
        video = th.cat((feat_2d, feat_3d))
        cap = self.data[idx]['caption']
        caption = self._words_to_we(self._tokenize_text(cap))
        return {'video': video, 'text': caption, 'video_id': self.data[idx]['id']}
