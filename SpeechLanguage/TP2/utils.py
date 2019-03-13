#!/usr/bin/env python3

"""
Pierre-Louis Guhur - TP2 SL MVA
-------------------------------

Utils fonctions for NLP
"""

from pathlib import Path 
from argparse import ArgumentParser
from nltk import Tree
from nltk.treetransforms import chomsky_normal_form
import re
import itertools 
import os
import random

TRAINING, VALIDATION, TESTING = range(3)

def rm_labels(lines):
    """ Remove functional labels
    >>> rm_labels(['(PP-MOD (P En)', '(NP-SUJ (DET la)']) 
    ['(PP (P En)', '(NP (DET la)'])
    """
    for i in range(len(lines)):
        lines[i] = re.sub(r"(\([A-Z]+)-[A-Z]+", r"\1", lines[i])  
    return lines
    
def save(list_str, filename):
    with open(filename, "w+") as f:
        for string in list_str:
            f.write("%s\n" % string)

def load_grammar(filename):
    grammar = load(filename)
    grammar = 

def load(filename):
    with open(filename, "r") as f:
         return f.readlines()

def parse_rules(lines, stem=True):
    if stem:
        rm_labels(lines)
    for i in range(len(lines)):
        lines[i] = Tree.fromstring(lines[i])
   
def production_rules(rules, binarization=True):
    if binarization:
        for t in rules:
            chomsky_normal_form(t)

    rules = [t.productions() for t in rules]
    return list(itertools.chain.from_iterable(rules))


def get_leaves(rules):
    return [" ".join(t.leaves()) for t in rules]


class Dataset:

    def __init__(self, path, split=[0.8,0.1,0.1], seed=1):
        random.seed(seed)

        with open(path, "r") as f:
            db = f.readlines()
        random.shuffle(db)
        parse_rules(db)

	# Splitting 
        self.datasets = []
        n_data = len(db)
        start = 0
        for s in split:
            end = start + int(s*n_data) + 1
            self.datasets.append(db[start:end])
            start = end


    def get_grammar(self, idx=TRAINING):
        return production_rules(self.datasets[idx])

    def get_sentences(self, idx=TESTING):
        return get_leaves(self.datasets[idx])


if __name__ == "__main__":
   parser = ArgumentParser(description="Manage data")
   parser.add_argument("--database", "-d", 
           default=Path("data/sequoia-corpus+fct.mrg_strict"),
           type=Path, help="Path to SEQUOIA")
   parser.add_argument("--eval", "-e", 
           default=Path("results/eval"), 
           type=Path, help="Path to evaluation file")
   parser.add_argument("--split", "-s", 
           default=Path("results/split"), 
           type=Path, help="Path to splitting database")
 
   args = parser.parse_args()
   print(args)
                       
   dataset = Dataset(args.database)
   os.makedirs(args.eval, exist_ok=True)
   os.makedirs(args.split, exist_ok=True)

   db = Dataset(args.database)
   trainset = db.get_grammar(TRAINING)
   
   print("Training..")
   training = db.get_sentences(TRAINING)
   save(training, args.eval / "trainset.txt")
   save(db.datasets[TRAINING], args.split / "trainset.txt")

   print("Validation...")
   validset = db.get_sentences(VALIDATION)
   save(validset, args.eval / "validset.txt")
   save(db.datasets[VALIDATION], args.split / "validset.txt")

   print("Testing...")
   testset = db.get_sentences(TESTING)
   save(testset, args.eval / "testset.txt")
   save(db.datasets[TESTING], args.split / "testset.txt")

 
