import torch


class LanguageModel(torch.nn.Module):
    def __init__(self, model, decoder, vocab):
        super().__init__()

        self.model = model
        self.decoder = decoder
        self.vocab = vocab

        self.forward = model.forward

    def single_sentence_nll(self, sentence, prefix):
        sentence_ids = [self.vocab[c] for c in sentence]
        device = next(self.parameters()).device

        if prefix:
            prefix_id = self.vocab[prefix]
            tensor = torch.tensor([prefix_id] + sentence_ids).view(1, -1).to(device)
        else:
            tensor = torch.tensor(sentence_ids).view(1, -1).to(device)

        h0 = self.model.init_hidden(1)
        o, _ = self.model(tensor[:, :-1], h0)

        if prefix:
            nll, _ = self.decoder.neg_log_prob(o, tensor[:, 1:])
        else:
            prepended_o = torch.cat([h0[0][0].unsqueeze(0), o], dim=1)
            nll, _ = self.decoder.neg_log_prob(prepended_o, tensor)

        return nll.item()

    def batch_nll(self, sentences, prefix):
        return [self.single_sentence_nll(s, prefix) for s in sentences]
