#!/usr/bin/env python3
import torch
from brnolm.language_models import language_model

lm = torch.load('/mnt/matylda5/ibenes/projects/chime6/lms/base-training-workdir/best.lm', map_location='cpu')
sentence = 'a run cube </s>'
print(sentence)
print('orig lm score:', lm.single_sentence_nll(sentence.split(), None))

whole_lm_path = 'tmp/whole_lm.zip'
language_model.torchscript_export(lm, whole_lm_path)
new_lm = language_model.torchscript_import(whole_lm_path)
print('new model lm score:', new_lm.single_sentence_nll(sentence.split(), None))
