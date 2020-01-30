from test.common import TestCase

import io
import torch

from brnolm.language_models.vocab import Vocabulary

from brnolm.data_pipeline.reading import get_independent_lines


def get_stream(string):
    data_source = io.StringIO()
    data_source.write(string)
    data_source.seek(0)

    return data_source


class IndependentSentecesTests(TestCase):
    def setUp(self):
        self.vocab = Vocabulary('<unk>', 0)
        self.vocab.add_from_text('a b c')

    def test_single_word(self):
        f = get_stream('a\n')
        lines = get_independent_lines(f, self.vocab)
        self.assertEqual(lines, [torch.tensor([1])])
