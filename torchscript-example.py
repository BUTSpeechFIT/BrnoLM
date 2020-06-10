#!/usr/bin/env python3
import torch

lm = torch.load('/mnt/matylda5/ibenes/projects/chime6/lms/base-training-workdir/best.lm')
sentence = 'a run cube </s>'
print(sentence)
print('orig lm score:', lm.single_sentence_nll(sentence.split(), None))

orig_model = lm.model
s_model = torch.jit.script(orig_model)
lm.model = s_model

print('s_model lm score:', lm.single_sentence_nll(sentence.split(), None))

orig_dec = lm.decoder
s_dec = torch.jit.script(orig_dec)
lm.decoder = s_dec

print('s_model s_dec lm score:', lm.single_sentence_nll(sentence.split(), None))
