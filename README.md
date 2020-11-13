# BrnoLM
A neural language modeling toolkit built on PyTorch.

This is a scientific piece of code, so expect rough edges.

BrnoLM has so far powered language modeling in the following papers:
* Beneš et al. [Text Augmentation for Language Models in High Error Recognition Scenario](https://arxiv.org/pdf/2011.06056.pdf)
* Žmolíková et al. [BUT System for CHiME-6 Challenge](https://www.fit.vutbr.cz/research/groups/speech/publi/2020/zmolikova_CHiME_2020_abstract.pdf)
* Beneš et al. [i-vectors in language modeling: An efficient way of domain adaptation for feed-forward models](http://www.fit.vutbr.cz/research/groups/speech/publi/2018/benes_interspeech2018_1070.pdf)
* Beneš et al. [Unsupervised Language Model Adaptation for Speech Recognition with no Extra Resources](http://www.fit.vutbr.cz/research/groups/speech/publi/2019/benes_DAGA_2019.pdf)


## Installation
To install, clone this repository and exploit the provided `setup.py`, e.g.:

```
git clone git@github.com:BUTSpeechFIT/BrnoLM.git
cd BrnoLM
pip install . # or, if you don't care about environmental pollution: python setup.py install
```

If you want to edit the sources, [pip with `-e`](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs) or [setup.py develop](https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode).

Occasionally, a PIP version is produced, so you can simply `pip install brnolm` to obtain the last pre-packed version.


### Requirements
The above way of installation takes care of dependencies.
If you want to prepare an environment yourself, know that BrnoLM requires:

```
    torch
    numpy
    scikit-learn
```
Exact tested versions are provided in `setup.py`.
