#!/usr/bin/env python3

from setuptools import setup

setup(
    name='brnolm',
    version='0.1',
    packages=['brnolm'],
    install_requires=[
        'torch>=1.4',
        'numpy',
        'scikit-learn',
    ]
)
