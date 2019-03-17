#!/usr/bin/env python3

"""
Pierre-Louis Guhur - TP2 SL MVA
-------------------------------

Learning a PCFG grammar from a training set
"""


from utils import *
from argparse import ArgumentParser
from pathlib import Path
import os

from nltk.grammar import induce_pcfg, Nonterminal






              

if __name__ == "__main__":

    parser = ArgumentParser(description="build a PCFG grammar")
    parser.add_argument("--database", "-d", type=Path, help="path to the SEQUOIA file")
    parser.add_argument("--output", "-o", type=Path, help="folder to output results")
    args = parser.parse_args()
    print(args)

    dataset = Dataset(args.database)
    os.makedirs(args.output, exist_ok=True)

    print("Training set...")
    trainset = dataset.get_grammar(TRAINING)
    grammar = learning_pcfg(trainset)
    save(grammar, args.output / "trainset.txt")

    print("Validation set...")
    validset = dataset.get_grammar(VALIDATION)
    grammar = learning_pcfg(validset)
    save(grammar, args.output / "validset.txt")

    print("Test set...")
    testset = dataset.get_grammar(TESTING)
    grammar = learning_pcfg(testset)
    save(grammar, args.output / "testset.txt")
