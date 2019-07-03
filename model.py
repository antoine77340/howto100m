from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch.nn as nn
import torch as th
import torch.nn.functional as F
import re

class Net(nn.Module):
    def __init__(
            self,
            embd_dim=1024,
            video_dim=2048,
            n_pair=1,
            we_dim=300,
            max_words=30,
            sentence_dim=-1,
            we=None,
    ):
        super(Net, self).__init__()
        if sentence_dim <= 0:
            self.text_pooling = Sentence_Maxpool(we_dim, embd_dim)
        else:
            self.text_pooling = Sentence_Maxpool(we_dim, sentence_dim)
        self.GU_text = Gated_Embedding_Unit(
            self.text_pooling.out_dim, embd_dim, gating=True)
        self.GU_video = Gated_Embedding_Unit(
            video_dim, embd_dim, gating=True)
        self.n_pair = n_pair
        self.embd_dim = embd_dim
        self.we = we
        self.we_dim = we_dim


    def save_checkpoint(self, path):
        th.save(self.state_dict(), path)

    def load_checkpoint(self, path, cpu=False):
        if cpu:
            self.load_state_dict(th.load(path,
                map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(th.load(path))

    def forward(self, video, text):
        video = self.GU_video(video)
        text = self.GU_text(self.text_pooling(text))
        return th.matmul(text, video.t())



class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension, gating=True):
        super(Gated_Embedding_Unit, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)
        self.gating = gating

    def forward(self, x):
        x = self.fc(x)
        if self.gating:
            x = self.cg(x)
        x = F.normalize(x)
        return x

class Sentence_Maxpool(nn.Module):
    def __init__(self, word_dimension, output_dim, relu=True):
        super(Sentence_Maxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)
        self.out_dim = output_dim
        self.relu = relu

    def forward(self, x):
        x = self.fc(x)
        if self.relu:
            x = F.relu(x)
        return th.max(x, dim=1)[0]


class Context_Gating(nn.Module):
    def __init__(self, dimension, add_batch_norm=False):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)

    def forward(self, x):
        x1 = self.fc(x)
        if self.add_batch_norm:
            x1 = self.batch_norm(x1)
        x = th.cat((x, x1), 1)
        return F.glu(x, 1)
