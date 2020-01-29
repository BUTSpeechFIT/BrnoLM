from .common import TestCase


from brnolm.data_pipeline.masked import masked_tensor_from_sentences


class MaskedDataCreationTests(TestCase):
    def test_requires_batch(self):
        self.assertRaises(ValueError, masked_tensor_from_sentences, [0])
