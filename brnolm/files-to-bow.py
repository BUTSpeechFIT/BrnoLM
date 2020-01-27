#!/usr/bin/env python
 
import argparse

import lstm_model
import vocab
import language_model

import scipy.io as sio
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch LSTM Language Model')
    parser.add_argument('--filelist', type=str,  required=True,
                        help='list of files with documents (one per line)')
    parser.add_argument('--load', type=str,  required=True,
                        help='path to corresponding NN LM')
    parser.add_argument('--save', type=str,  required=True,
                        help='path to mtx file')
    args = parser.parse_args()

    print("loading model...")
    with open(args.load, 'rb') as f:
        lm = language_model.load(f)
    print(lm.model)
    vocab = lm.vocab

    documents = []
    with open(args.filelist) as fl:
        filenames = fl.read().split()
        for filename in filenames:
            with open(filename) as f:
                documents.append(f.read())
         
    cvect = CountVectorizer(documents, analyzer='word', lowercase=False, vocabulary=vocab)

    document_bows = cvect.fit_transform(documents)
    vocab = cvect.get_feature_names()
    print('document_bows:', document_bows.shape)

    sio.mmwrite(args.save, document_bows.T)
