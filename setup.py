#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：csc_scope 
# @File    ：setup.py
# @Author  ：sl
# @Date    ：2022/10/22 9:07


from __future__ import print_function

import io
import sys

import os
from setuptools import setup, find_packages

__version__ = "0.0.1"
exec(open('csc_eda/version.py').read())

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for csc_eda.')


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def get_package_data_files(package, data, package_dir=None):
    """
    Helps to list all specified files in package including files in directories
    since `package_data` ignores directories.
    """
    if package_dir is None:
        package_dir = os.path.join(*package.split('.'))
    all_files = []
    for f in data:
        path = os.path.join(package_dir, f)
        if os.path.isfile(path):
            all_files.append(f)
            continue
        for root, _dirs, files in os.walk(path, followlinks=True):
            root = os.path.relpath(root, package_dir)
            for file in files:
                file = os.path.join(root, file)
                if file not in all_files:
                    all_files.append(file)
    return all_files


with open('requirements.txt', 'r', encoding='utf-8') as f:
    reqs = f.read()

setup(
    name='csc_eda',
    version=__version__,
    description='Csc eda',
    # long_description=readme,
    long_description=read("README.md"),
    license='Apache License 2.0',
    url='https://github.com/csc_eda',
    author='sl',
    author_email='sl@qq.com',
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    keywords='NLP,csc, eda',
    setup_requires=[],
    install_requires=reqs.strip().split('\n'),
    # packages=find_packages(),
    packages=find_packages(
        where='.',
        exclude=('data*', 'docs*','tests*', 'outputs*', 'examples*', 'tests*', 'applications*', 'model_zoo*')),
    # package_dir={'cscscope': 'cscscope'},
    # package_data={
    #     'financial_ner': ['*.*', '../LICENSE', '../README.*', '../*.txt', 'data/*', 'data/en/en.json.gz',
    #                     'utils/*.', 'bert/*', 'deep_context/*', 'conv_seq2seq/*', 'seq2seq_attention/*',
    #                     'transformer/*', 'electra/*']}
)
