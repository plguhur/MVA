#!/usr/bin/env python3

"""
Pierre-Louis Guhur - TP2 SL MVA
-------------------------------

Parse a sentence using the CYK algorithm
"""


from utils import load, save, load_grammar
from argparse import ArgumentParser
from pathlib import Path
import os

from nltk.grammar import induce_pcfg, Nonterminal, is_nonterminal
              
class CYK:
    def __init__(self,grammar):
	
        self.non_terminal_dic = dict()
        self.terminals_dic = dict()
        self.grammar = load_grammar(grammar)
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
            if (begin_,end_,label_) in self.terminals_dic:
                t = Tree(str(label_),[self.terminals_dic[(begin_,end_,label_)]])
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
            self.non_terminal_dic[nonterm] = 0
        n = len(sentence)
        score = [[self.non_terminal_dic.copy() for i in range(n+1)] for j in range(n+1)]
        nodes = [[self.non_terminal_dic.copy() for i in range(n+1)] for j in range(n+1)]
        for i in range(n):
            for prod in self.leaves_rules:
                if prod.rhs()[0] == sentence[i]:
                    score[i][i+1][prod.lhs()] = prod.prob()
                    self.terminals_dic[(i,i+1,prod.lhs())] = sentence[i]
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

        t = self.backtrack(sentence,nodes,0,n,Nonterminal('SENT'))
        Tree.un_chomsky_normal_form(t)
        return(t)



def parsing(parser, file_in, file_out):
    sentences = load(filename)
    sentences = list(map(lambda x: x.split(), sentences))
    parses = []

    for sentence in sentences:
        count += 1
        print('Sentence number',count,'is in process...')
        t = parser.parse(sentence)
        if t is None:
            print('Not found in the Grammar:', sentence)
        else:
            prediction = ' '.join(str(t).split())
            parses.append(prediction)
    save(parses, file_out)


if __name__ == "__main__":

    parser = ArgumentParser(description="Parse a sentence using CYK algorithm")
    parser.add_argument("--grammar", "-g", 
            default=Path("results/grammar/trainset.txt"),
            type=Path, help="Path to grammar")
    parser.add_argument("--sentences", "-s", 
            default=Path("results/eval/"),
            type=Path, help="Path to sentences")
    parser.add_argument("--output", "-o", 
            default=Path("results/parse"), type=Path, 
            help="Path to sentences")
    args = parser.parse_args()
    print(args)

    cyk = CYK(args.grammar)
    os.makedirs(args.output, exist_ok=True)

    print("Training set...")
    parsing(cyk, args.sentences / "trainset.txt", \
		 args.output / "trainset.txt")

    print("Validation set...")
    parsing(cyk, args.sentences / "validset.txt", \
		 args.output / "validset.txt")

    print("Test set...")
    parsing(cyk, args.sentences / "testset.txt", \
		 args.output / "testset.txt")
