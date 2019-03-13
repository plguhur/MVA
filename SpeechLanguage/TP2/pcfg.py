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


def chomsky_normal_form(grammar):
    new_rule = dict(rules)
    for rule in rules:
        list_of_nn_symbols = rules[rule]
        for set_of_symbols in list_of_nn_symbols:
            if len(set_of_symbols)>2:
                new_rule[rule].remove(set_of_symbols)
                new_symbol = list(set_of_symbols)
                while len(new_symbol)!=2:
                    concatenation = tuple([new_symbol[0],new_symbol[1]])
                    new_symbol = [new_symbol[0] +'+' +new_symbol[1]] + new_symbol[2:]
                    new_rule[new_symbol[0]] =  set()
                    new_rule[new_symbol[0]].add(concatenation)
                    new_probabilities_rules[tuple_proba] = 1
                
                
                new_rule[rule].add(tuple(new_symbol))
    return new_rule


def learning_PCFG(rules):
    S = Nonterminal('SENT')
    return induce_pcfg(S, rules).productions()


              

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
    grammar = learning_PCFG(trainset)
    save(grammar, args.output / "trainset.txt")

    print("Validation set...")
    validset = dataset.get_grammar(VALIDATION)
    grammar = learning_PCFG(validset)
    save(grammar, args.output / "validset.txt")

    print("Test set...")
    testset = dataset.get_grammar(TESTING)
    grammar = learning_PCFG(testset)
    save(grammar, args.output / "testset.txt")
