import torch
from brnolm.data_pipeline.masked import masked_tensor_from_sentences


class LanguageModel(torch.nn.Module):
    def __init__(self, model, decoder, vocab):
        super().__init__()

        self.model = model
        self.decoder = decoder
        self.vocab = vocab

    def forward(self, x, h0):
        o, h = self.model(x, h0)
        return self.decoder(o), h

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
            prepended_o = torch.cat([h0[0][0].unsqueeze(1), o], dim=1)
            nll, _ = self.decoder.neg_log_prob(prepended_o, tensor)

        return nll.item()

    def batch_nll(self, sentences, prefix):
        if not sentences:
            return []

        idx_seqs = [[self.vocab[w] for w in s] for s in sentences]

        if prefix:
            prefix_idx = self.vocab[prefix]

            for s_idxs in idx_seqs:
                s_idxs.insert(0, prefix_idx)

        masked_nlllh = self.batch_nll_idxs(idx_seqs, predict_first=not prefix)

        return masked_nlllh.sum(dim=1).detach().numpy().tolist()

    def batch_nll_idxs(self, idxs, predict_first=True):
        input, target, mask = masked_tensor_from_sentences(idxs)
        batch_size = input.shape[0]

        if predict_first:
            first_inputs = input[:, 0].view(-1, 1)
            target = torch.cat([first_inputs, target], dim=1)

            batch_of_ones = torch.ones((batch_size, 1), dtype=torch.int64)
            mask = torch.cat([batch_of_ones, mask], dim=1)

        h0 = self.model.init_hidden(batch_size)
        o, _ = self.model(input, h0)
        if predict_first:
            o0 = h0[0][0].unsqueeze(1)
            o = torch.cat([o0, o], dim=1)

        all_nlllh = self.decoder.neg_log_prob_raw(o, target)

        return all_nlllh * mask
