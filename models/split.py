'''
 @Date  : 2018/1/9
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn
 @homepage: shumingma.com
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils
import models


class split(models.seq2seq):

    def __init__(self, config):
        super(split, self).__init__(config, use_attention=False)
        self.split_num = 10
        self.position_embedding = nn.Embedding(200, config.hidden_size)
        self.mlp = nn.Sequential(nn.Linear(config.hidden_size*3, config.hidden_size*2),
                                 nn.Tanh(),
                                 nn.Linear(config.hidden_size*2, config.hidden_size*2),
                                 nn.Tanh())

    def forward(self, src, src_len, dec, targets):
        src = src.t()
        dec = dec.t()
        targets = targets.t()

        contexts, enc_state = self.encoder(src, src_len.data.tolist())
        if self.decoder.attention is not None:
            self.decoder.attention.init_context(context=contexts)
        outputs, state = [], enc_state

        for i, input in enumerate(dec.split(1)):
            if (i+1) % self.split_num == 0:
                state = self.update_state(i, enc_state)
            output, state, attn_weights = self.decoder(input.squeeze(0), state)
            outputs.append(output)
        outputs = torch.stack(outputs)

        loss = self.compute_loss(outputs, targets)
        return loss, outputs

    def update_state(self, i, enc_state, volatile=False):
        eh, ec = enc_state
        position = Variable(torch.ones(eh.size(0), eh.size(1)).long().fill_(i), volatile=volatile).cuda()
        position = self.position_embedding(position)
        h, c = self.mlp(torch.cat([eh, ec, position], dim=-1)).chunk(2, dim=-1)
        return h+eh, c+ec

    def sample(self, src, src_len):

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, reverse_indices = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        bos = Variable(torch.ones(src.size(0)).long().fill_(utils.BOS), volatile=True)
        src = src.t()

        if self.use_cuda:
            bos = bos.cuda()

        contexts, enc_state = self.encoder(src, lengths.data.tolist())
        if self.decoder.attention is not None:
            self.decoder.attention.init_context(context=contexts)
        inputs, outputs, attn_matrix = [bos], [], []
        state = enc_state
        for i in range(self.config.max_time_step):
            if (i+1) % self.split_num == 0:
                state = self.update_state(i, enc_state, volatile=True)
            output, state, attn_weights = self.decoder(inputs[i], state)
            predicted = output.max(1)[1]
            inputs += [predicted]
            outputs += [predicted]
            attn_matrix += [attn_weights]

        outputs = torch.stack(outputs)
        sample_ids = torch.index_select(outputs, dim=1, index=reverse_indices).t().data

        if self.decoder.attention is not None:
            attn_matrix = torch.stack(attn_matrix)
            alignments = attn_matrix.max(2)[1]
            alignments = torch.index_select(alignments, dim=1, index=reverse_indices).t().data
        else:
            alignments = None

        #sample_ids = [sample[:length] for length, sample in zip(src_len.data.tolist(), sample_ids)]

        return sample_ids, alignments

    def beam_sample(self, src, src_len, beam_size=1):

        # (1) Run the encoder on the src.

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        src = src.t()
        batch_size = src.size(1)
        contexts, encState = self.encoder(src, lengths.data.tolist())

        #  (1b) Initialize for the decoder.
        def var(a):
            return Variable(a, volatile=True)

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Repeat everything beam_size times.
        contexts = rvar(contexts.data)

        if self.config.cell == 'lstm':
            decState = (rvar(encState[0].data), rvar(encState[1].data))
        else:
            decState = rvar(encState.data)
        #decState.repeat_beam_size_times(beam_size)
        beam = [models.Beam(beam_size, n_best=1,
                          cuda=self.use_cuda, length_norm=self.config.length_norm)
                for __ in range(batch_size)]
        if self.decoder.attention is not None:
            self.decoder.attention.init_context(contexts)

        # (2) run the decoder to generate sentences, using beam search.

        for i in range(self.config.max_time_step):

            if (i+1) % 5 == 0:
                decState = self.update_state(decState, encState)

            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam])
                      .t().contiguous().view(-1))

            # Run one step.
            output, decState, attn = self.decoder(inp, decState)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(self.log_softmax(output))
            attn = unbottle(attn)
                # beam x tgt_vocab

            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):
                b.advance(output.data[:, j], attn.data[:, j])
                b.beam_update(decState, j)

        # (3) Package everything up.
        allHyps, allScores, allAttn = [], [], []

        for j in ind.data:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])

        return allHyps, allAttn
