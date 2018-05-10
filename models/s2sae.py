'''
 @Date  : 2018/1/10
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn 
 @homepage: shumingma.com
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import utils
import models


class s2sae(nn.Module):

    def __init__(self, config, module):
        super(s2sae, self).__init__()
        self.s2s = models.seq2seq(config)
        self.ae = getattr(models, module)(config, use_attention=False, encoder=self.s2s.encoder)

    def forward(self, src, src_len, dec, targets, ae_dec, ae_target):
        s2s_loss, s2s_output = self.s2s(src, src_len, dec, targets)
        ae_loss, ae_output = self.ae(src, src_len, ae_dec, ae_target)
        return s2s_loss, ae_loss, s2s_output, ae_output

    def sample(self, src, src_len):
        return self.s2s.sample(src, src_len)

    def beam_sample(self, src, src_len, beam_size=1):
        return self.s2s.beam_sample(src, src_len, beam_size)