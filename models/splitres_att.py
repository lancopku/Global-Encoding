'''
 @Date  : 2018/1/4
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn 
 @homepage: shumingma.com
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils
import models


class splitres_att(models.seq2seq):

    def __init__(self, config, use_attention=True, encoder=None, decoder=None):
        super(splitres_att, self).__init__(config, use_attention=use_attention, encoder=encoder)
        self.split_num = config.split_num
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.Sigmoid())
        self.linear_out = nn.Sequential(nn.Linear(2*config.hidden_size, config.hidden_size), nn.SELU(), nn.Linear(config.hidden_size, config.hidden_size), nn.SELU())

    def forward(self, src, src_len, dec, targets):
        src = src.t()
        dec = dec.t()
        targets = targets.t()

        contexts, enc_state, embeds = self.encoder(src, src_len.data.tolist())
        if self.decoder.attention is not None:
            self.decoder.attention.init_context(context=contexts)
        outputs, state = [], enc_state

        for i, input in enumerate(dec.split(1)):
            if (i+1) % self.split_num == 0:
                attn_state = torch.bmm(attn_weights.unsqueeze(1), contexts.transpose(0,1)).squeeze(1) # batch * size
                state = self.update_state(state, (attn_state, attn_state))
            #if i == 0:
            inp = input.squeeze(0)
            #else:
            #    inp = outputs[-1].max(1)[1]
            output, state, attn_weights = self.decoder(inp, state, embeds)
            outputs.append(output)
        outputs = torch.stack(outputs)

        loss = self.compute_loss(outputs, targets)
        return loss, outputs

    def update_state(self, state, enc_state):
        h, c = state
        #gate_h, gate_c = self.linear(h), self.linear(c)
        eh, ec = enc_state
        #eh, ec = gate_h * eh, gate_c * ec
        #print(h.size(), c.size(), eh.size(), ec.size())
        h_, c_ = self.linear_out(torch.cat([h, eh.unsqueeze(0)], -1)), self.linear_out(torch.cat([c, ec.unsqueeze(0)], -1))
        #h_, c_ =  self.linear_out(torch.cat([h, eh.unsqueeze(0)], -1)), self.linear_out(torch.cat([c, ec.unsqueeze(0))
        return h_, c_

    def sample(self, src, src_len):

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, reverse_indices = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        bos = Variable(torch.ones(src.size(0)).long().fill_(utils.BOS), volatile=True)
        src = src.t()

        if self.use_cuda:
            bos = bos.cuda()

        contexts, enc_state, embeds = self.encoder(src, lengths.data.tolist())
        if self.decoder.attention is not None:
            self.decoder.attention.init_context(context=contexts)
        inputs, outputs, attn_matrix = [bos], [], []
        state = enc_state
        for i in range(self.config.max_time_step):
            if (i+1) % self.split_num == 0:
                attn_state = torch.bmm(attn_weights.unsqueeze(1), contexts.transpose(0,1)).squeeze(1) # batch * size
                state = self.update_state(state, (attn_state, attn_state))
            output, state, attn_weights = self.decoder(inputs[i], state, embeds)
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
        contexts, encState, embeds = self.encoder(src, lengths.data.tolist())

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
        contexts = rvar(contexts.transpose(0, 1).data)
        embeds = rvar(embeds.data)

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

            if (i+1) % self.split_num == 0:
                #attn = bottle(attn)
                attn_state = torch.bmm(attn.unsqueeze(2), contexts).squeeze(2) # batch * size
                decState = self.update_state(decState, (attn_state, attn_state))

            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam])
                      .t().contiguous().view(-1))

            # Run one step.
            output, decState, attn = self.decoder(inp, decState, embeds)
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
