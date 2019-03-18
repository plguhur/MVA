#!/usr/bin/env python3

"""
Pierre-Louis Guhur - TP2 SL MVA
-------------------------------

Parse a sentence using the CYK algorithm
"""


from utils import *
from oov import OOV
from argparse import ArgumentParser
from pathlib import Path
import os
from nltk import word_tokenize 
from nltk.grammar import induce_pcfg, Nonterminal, is_nonterminal
from tqdm import tqdm          

    
class CYK:
    def __init__(self,grammar):

        self._nonterm = dict()
        self._term = dict()
        self.grammar = grammar
        self.non_terminal = set()
        self.leaves_rules = set()
        self.unary_rules = set()
        self.binary_rules = set()

        self.nonTerminal()
        self.binaryRules()
        self.leavesRules()
        self.unaryRules()    


    def nonTerminal(self):
        for prod in self.grammar:
            if is_nonterminal(prod.lhs()):
                self.non_terminal.add(prod.lhs())
                for i in range(len(prod.rhs())):
                    y = prod.rhs()[i]
                    if is_nonterminal(y):
                        self.non_terminal.add(y)

    def leavesRules(self):
        for prod in self.grammar:
            if type(prod.rhs()[0]) == str:
                self.leaves_rules.add(prod)


    def unaryRules(self):
        for prod in self.grammar:
            if len(prod.rhs())==1 and type(prod.rhs()[0])!=str:
                self.unary_rules.add(prod)


    def binaryRules(self):
        for prod in self.grammar:
            if len(prod.rhs())==2:
                self.binary_rules.add(prod)

    
    def backtrack(self,sentence,nodes,begin,end,label):
        begin_ = begin
        end_ = end
        label_ = label
        n = len(sentence)
        if nodes[0][n][Nonterminal('SENT')] == 0:
            return None
        next_ =  nodes[begin_][end_][label_]
        if next_ == 0:
            if (begin_,end_,label_) in self._term:
                t = Tree(str(label_),[self._term[(begin_,end_,label_)]])
                return t
        if type(next_)!=tuple:
            label_ = next_
            t1 = self.backtrack(sentence,nodes,begin_,end_,label_)
            t = Tree(str(label),[t1])
            return t
        else:
            (split, B, C) = next_
            t1 = self.backtrack(sentence,nodes,begin,split,B)
            t2 = self.backtrack(sentence,nodes,split,end,C)
            t = Tree(str(label_),[t1,t2])
            return t


    def parse(self,sentence):
        for nonterm in self.non_terminal:
            self._nonterm[nonterm] = 0
        n = len(sentence)
        
        score = [[self._nonterm.copy() for i in range(n+1)] for j in range(n+1)]
        nodes = [[self._nonterm.copy() for i in range(n+1)] for j in range(n+1)]
        for i in range(n):
            for prod in self.leaves_rules:
                if prod.rhs()[0] == sentence[i]:
                    score[i][i+1][prod.lhs()] = prod.prob()
                    self._term[(i,i+1,prod.lhs())] = sentence[i]
            added = True
            while added:
                added = False
                for prod in self.unary_rules:
                    if (score[i][i+1][prod.rhs()[0]]>0):
                        A = prod.lhs()
                        B = prod.rhs()[0]
                        proba = prod.prob()*score[i][i+1][B]
                        if proba > score[i][i+1][A]:
                            score[i][i+1][A] = proba
                            nodes[i][i+1][A] = B
                            added = True
        for span in range(2,n+1):
            for begin in range(0,n-span+1):
                end = begin+span
                for split in range(begin+1,end):
                    for prod in self.binary_rules:
                        A = prod.lhs()
                        B = prod.rhs()[0]
                        C = prod.rhs()[1]
                        proba = score[begin][split][B]*score[split][end][C]*prod.prob()
                        if proba > score[begin][end][A]:
                            score[begin][end][A] = proba
                            nodes[begin][end][A] = (split,B,C)
                added = True
                while added:
                    added = False
                    for prod in self.unary_rules:
                        A = prod.lhs()
                        B = prod.rhs()[0]
                        proba = prod.prob()*score[begin][end][B]
                        if proba > score[begin][end][A]:
                            score[begin][end][A] = proba
                            nodes[begin][end][A] = B
                            added = True
#         print(t)
        t = self.backtrack(sentence,nodes,0,n,Nonterminal('SENT'))
        Tree.un_chomsky_normal_form(t)
        return(t)
    
    def parse3(self,sentence):
        for nonterm in self.non_terminal:
            self._nonterm[nonterm] = 0
        n = len(sentence)
        nonterm = self._nonterm.copy()
        nonterm2idx = {r:i for i, w in enumerate(nonterm)}
        score = np.zeros((n+1,n+1, len(nonterm)))
        nodes = np.zeros((n+1,n+1, len(nonterm)))
        
        for i in range(n):
            for prod in self.leaves_rules:
                if prod.rhs()[0] == sentence[i]:
                    score[i,i+1,nonterm2idx[prod.lhs()]] = prod.prob()
                    self._term[(i,i+1,prod.lhs())] = sentence[i]
            added = True
            while added:
                added = False
                for prod in self.unary_rules:
                    if (score[i][i+1][prod.rhs()[0]]>0):
                        A = prod.lhs()
                        B = prod.rhs()[0]
                        proba = prod.prob()*score[i][i+1][B]
                        if proba > score[i][i+1][A]:
                            score[i][i+1][A] = proba
                            nodes[i][i+1][A] = B
                            added = True
        for span in range(2,n+1):
            for begin in range(0,n-span+1):
                end = begin+span
                for split in range(begin+1,end):
                    for prod in self.binary_rules:
                        A = prod.lhs()
                        B = prod.rhs()[0]
                        C = prod.rhs()[1]
                        proba = score[begin][split][B]*score[split][end][C]*prod.prob()
                        if proba > score[begin][end][A]:
                            score[begin][end][A] = proba
                            nodes[begin][end][A] = (split,B,C)
                added = True
                while added:
                    added = False
                    for prod in self.unary_rules:
                        A = prod.lhs()
                        B = prod.rhs()[0]
                        proba = prod.prob()*score[begin][end][B]
                        if proba > score[begin][end][A]:
                            score[begin][end][A] = proba
                            nodes[begin][end][A] = B
                            added = True
#         print(t)
        t = self.backtrack(sentence,nodes,0,n,Nonterminal('SENT'))
        Tree.un_chomsky_normal_form(t)
        return(t)

    def parse2(self, sentence):
        n = len(sentence)
        r = len(self.non_terminal)
        P = np.zeros((n, n, r))
        back = np.zeros((n, n, r))
        for s in range(n):
#             P(1, s, 
              
            for prod in self.leaves_rules:
                if prod.rhs()[0] == sentence[s]:
#                     P(1, score[i][i+1][prod.lhs()] = prod.prob()
                    self._term[(i,i+1,prod.lhs())] = sentence[i]
            added = True
            while added:
                added = False
                for prod in self.unary_rules:
                    if (score[i][i+1][prod.rhs()[0]]>0):
                        A = prod.lhs()
                        B = prod.rhs()[0]
                        proba = prod.prob()*score[i][i+1][B]
                        if proba > score[i][i+1][A]:
                            score[i][i+1][A] = proba
                            nodes[i][i+1][A] = B
                            added = True

                      
                      
def parsing(parser, corrector, sentences, file_out):
    parses = []

    for i, sentence in enumerate(tqdm(sentences)):
        sentence = word_tokenize(sentence)
        s = [""] + sentence + [""]
#         print("To parse...", sentence)
        replacements = {corrector(s[i-1], s[i], s[i+1]):s[i] 
                             for i in range(1, len(sentence)+1)}
#         print("Replaced by...", list(replacements.keys()))
        t = parser.parse(list(replacements.keys()))
        if t is None:
            prediction = " ".join(sentence)
        else:
            for leaf_pos in t.treepositions('leaves'):
                t[leaf_pos] = replacements[t[leaf_pos]]
            prediction = ' '.join(str(t).split())
#         print("Replaced parse...", prediction)
        parses.append(prediction)
        
    save(parses, file_out)


if __name__ == "__main__":

    parser = ArgumentParser(description="Parse a sentence using CYK algorithm")
    parser.add_argument("--database", "-d", 
            default=Path("data/sequoia-corpus+fct.mrg_strict"),
            type=Path, help="Path to SEQUOIA database")
    parser.add_argument("--eval", "-e", 
            default=Path("results/eval/"),
            type=Path, help="Path to output evaluation sentences")
    parser.add_argument("--output", "-o", 
            default=Path("results/parse"), type=Path, 
            help="Path to output parsing")
    parser.add_argument("--split", "-s", 
           default=Path("results/split"), 
           type=Path, help="Path to splitting database")
 

    args = parser.parse_args()
    print(args)

    dataset = Dataset(args.database)
    os.makedirs(args.eval, exist_ok=True)
    os.makedirs(args.split, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    db = Dataset(args.database)
    pcfg = db.get_pcfg(TRAINING)
    test_sentences = db.get_sentences(TESTING)
    train_sentences = db.get_sentences(TRAINING)
    cyk = CYK(pcfg)
    oov = OOV(train_sentences)
#     print("Training set...")
#     print("Validation set...")
    

    print("Test set...")
    parsing(cyk, oov.correct_word, test_sentences,
            args.output / "testset.txt")
    save(test_sentences, args.eval / "testset.txt")
    ground_truth = db.datasets[TESTING]
    parses = []
    for t in ground_truth:
        prediction = ' '.join(str(t).split())
        parses.append(prediction)
    save(parses, args.split / "testset.txt")

