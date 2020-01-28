#!/usr/bin/env python
import argparse
import sys

from brnolm.language_models.vocab import vocab_from_kaldi_wordlist

ST_WORDS = 0
ST_OOV_INTEREST = 1
ST_OOV_OTHER = 2


def words_from_idx(idx_list):
    transcript = []
    state = ST_WORDS
    for idx in idxes:
        if state == ST_WORDS:
            if idx == oov_start_idx + args.interest_constant:
                state = ST_OOV_INTEREST
            elif idx == oov_start_idx:
                state = ST_OOV_OTHER
            elif idx == oov_end_idx + args.interest_constant:
                raise ValueError("Unacceptable end of OOV-OI within WORDS ({}, key {})".format(line_no, key))
            elif idx == oov_end_idx:
                raise ValueError("Unacceptable end of OOV-NI within WORDS ({}, key {})".format(line_no, key))
            else:
                transcript.append(decoder_vocabulary.i2w(idx))
        elif state == ST_OOV_INTEREST:
            if idx == oov_end_idx + args.interest_constant:
                transcript.append(args.unk_oi)
                state = ST_WORDS
            elif idx == oov_end_idx:
                raise ValueError("Unacceptable end of OOV-NI within OOV-OI ({}, key {})".format(line_no, key))
            elif idx == oov_start_idx + args.interest_constant:
                raise ValueError("Unacceptable start of OOV-OI within OOV-OI ({}, key {})".format(line_no, key))
            elif idx == oov_start_idx:
                raise ValueError("Unacceptable start of OOV-NI within OOV-OI ({}, key {})".format(line_no, key))
            else:
                pass
        elif state == ST_OOV_OTHER:
            if idx == oov_end_idx:
                transcript.append(args.unk)
                state = ST_WORDS
            elif idx == oov_end_idx + args.interest_constant:
                raise ValueError("Unacceptable end of OOV-OI within OOV-NI ({}, key {})".format(line_no, key))
            elif idx == oov_start_idx + args.interest_constant:
                raise ValueError("Unacceptable start of OOV-OI within OOV-NI ({}, key {})".format(line_no, key))
            elif idx == oov_start_idx:
                raise ValueError("Unacceptable start of OOV-NI within OOV-NI ({}, key {})".format(line_no, key))
            else:
                pass
        else:
            raise RuntimeError("got into an impossible state {}".format(state))

    if state == ST_OOV_INTEREST:
        raise ValueError("Incomplete OOV of interest on line '{}'".format(idx_list))
    elif state == ST_OOV_OTHER:
        raise ValueError("Incomplete OOV (not of interest) on line '{}'".format(idx_list))

    return transcript


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unk', default="<UNK>")
    parser.add_argument('--unk-oi', default="<UNK-OI>")
    parser.add_argument('--oov-start', required=True)
    parser.add_argument('--oov-end', required=True)
    parser.add_argument('--interest-constant', type=int, required=True)
    parser.add_argument('--decoder-wordlist', required=True)
    args = parser.parse_args()

    with open(args.decoder_wordlist) as f:
        decoder_vocabulary = vocab_from_kaldi_wordlist(f, unk_word=args.unk)

    oov_start_idx = decoder_vocabulary[args.oov_start]
    oov_end_idx = decoder_vocabulary[args.oov_end]

    for line_no, line in enumerate(sys.stdin):
        fields = line.split()
        key = fields[0]
        idxes = [int(idx) for idx in fields[1:]]

        try:
            transcript = words_from_idx(idxes)
        except ValueError:
            sys.stderr.write("WARNING: there was a problem with input line {} (counting from 0)\n".format(line_no))
            continue

        sys.stdout.write("{} {}\n".format(key, " ".join(str(w) for w in transcript)))
