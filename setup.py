#!/usr/bin/env python3

from setuptools import setup


def get_long_desc():
    with open('README.md') as f:
        return f.read()


setup(
    name='brnolm',
    version='0.2.0',
    python_requires='>=3.6',
    packages=[
        'brnolm',
        'brnolm/data_pipeline',
        'brnolm/language_models',
        'brnolm/oov_clustering',
        'brnolm/runtime',
        'brnolm/smm_itf',
    ],
    install_requires=[
        'torch>=1.4',
        'numpy',
        'scikit-learn',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url='https://github.com/BUTSpeechFIT/BrnoLM',
    description='A language modeling toolkit',
    long_description=get_long_desc(),
    long_description_content_type='text/markdown',
    author='Karel Benes',
    author_email='ibenes@fit.vutbr.cz',
)
