#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from utils import create_data_from_lines, ConvertToChomsky, count_nb_of_non_terminal, all_symbols, reverse_rules
import numpy as np
from string import punctuation
import re
punctuation_list = list(punctuation.replace("'",''))

from nltk.tag import StanfordPOSTagger

from enchant import checker


class Tree(object):
    '''
    Classe qui permet de crée un arbre (utile pour le decoding du résultat dde PCYK)
    '''
    def __init__(self):
        '''
        Attributs de base de la class
        '''
        self.left = None
        self.right = None
        self.data = None
        self.index_right = 0
        self.index_left = 0


class ProbabilisticCYK:
    '''
    Classe ProbabilisticCYK qui permet d'utiliser l'algorithme Probabilistic CYK à partir d'une 
    grammaire apprise.
    Attributs: @proba_rules: dictionnaire qui contient les probabilités associées à chaque règle
    @proba_vocab: dictionnaire qui contient les probabilités associées à chaque règle qui mènent à des ancres
    @rules: dictionnaire de set qui contient les Règles de la forme A -> BC (Chomsky normal form)
    @lexicon: Règle de la forme  \gamma -> A ou \gamma est un mot du vocabulaire et A un symbole (dictionnaire de set)
    @nb_non_terminal: Nombre de symboles
    @all_non_terminal: liste de tous les symboles
    @reverse_rules: dictionnaire qui contient les règle sous la forme inversée BC -> A (pour appliquer PYCK)
    @pos_tagger_unknow_vocab: postagger standford pour gérer les mots inconnues
    @use_autcorrection: utiliser ou non la correction automatique des phrases dans le traitement
    @spell_checker: Checker enchant
    '''
    
    def __init__(self,jar=None,model=None,use_autocorrection=False):
        '''
        Constructeur de la classe
        '''
        self.proba_rules = {}
        self.proba_vocab = {}
        self.rules = {}
        self.lexicon = {}
        self.nb_non_terminal = 0
        self.all_non_terminal = []
        self.reverse_rules = {}
        if ((jar!=None) and (model!=None)):
            self.pos_tagger_unknow_vocab = StanfordPOSTagger(model, jar, encoding='utf8' )
        else:
            self.pos_tagger_unknow_vocab = None
        self.use_autcorrection = use_autocorrection
        self.spell_checker= checker.SpellChecker("fr_FR")

        
    def fit(self,lines):
        '''
        Fonction qui permet d'apprendre la grammaires probabilisées
        Param: @lines: list de str qui contient des exemples de phrases parsées
        '''
        self.lexicon,rules,self.proba_vocab,proba_rules = create_data_from_lines(lines)
        self.rules,self.proba_rules = ConvertToChomsky(rules,proba_rules)
        self.nb_non_terminal= count_nb_of_non_terminal(self.rules)
        self.all_non_terminal = list(all_symbols(self.rules))
        self.reverse_rules = reverse_rules(self.rules)
        
    def predict_one_line(self,line):
        '''
        Applique probabilistic CYK sur une ligne (fit doit avoir été appelé avant)
        Param: @line : ligne à parser
        '''
        line_to_use = str(line)
        
        if self.use_autcorrection:
            self.spell_checker.set_text(line_to_use)
        
            for err in self.spell_checker:
                sug = err.suggest()[0]
                err.replace(sug)
            line_to_use = self.spell_checker.get_text()
        line_to_use = line_to_use.replace("\'","\' ")
        
        global punctuation_list
        for punc in punctuation_list:
            line_to_use = line_to_use.replace(punc,' '+ punc+ ' ')
            

        words = line_to_use.split()
        length_line = len(words)
        table = np.zeros((length_line+1,length_line+1,self.nb_non_terminal))
        back=  {}
        for j in range(1,length_line+1):
            word= words[j-1]
            if word in self.lexicon:
                list_of_possible_symbols = self.lexicon[word]
                all_proba = None
            else:
                if self.pos_tagger_unknow_vocab==None:
                    all_proba = 1/self.nb_non_terminal
                    list_of_possible_symbols = [1]
                else:
                    standford_symbol = self.pos_tagger_unknow_vocab.tag(word.split())[0][1]
                    if standford_symbol in self.all_non_terminal:
                        table[j-1,j,self.all_non_terminal.index(standford_symbol)] = 1 #noprobabilities in the tagger
                        list_of_possible_symbols = []
                    else:
                        all_proba = 1/self.nb_non_terminal
                        list_of_possible_symbols = [1]
            for a in list_of_possible_symbols:
                if all_proba ==None:
                    tuple_proba = tuple([word,a])
                    table[j-1,j,self.all_non_terminal.index(a)] =  self.proba_vocab[tuple_proba]
                else:
                    table[j-1,j,:] = all_proba*np.ones((self.nb_non_terminal))
                    
            
            list_to_run_through = list(range(0,j-1))
            list_to_run_through.reverse()
        
            for i in list_to_run_through:
                for k in range(i+1,j):
                    possible_B = np.array(self.all_non_terminal)[table[i,k,:]>0]
                    possible_C = np.array(self.all_non_terminal)[table[k,j,:]>0]
                    
                    for B in possible_B:
                        for C in possible_C:
                            tuple_symbol = tuple([B,C])
                            if tuple_symbol in self.reverse_rules:
                                list_of_possible_rules = self.reverse_rules[tuple_symbol]
                                all_proba = 0
                            else:
                                continue
                            
                            for rule in list_of_possible_rules:
                                A = rule
                                tuple_proba = tuple([rule,tuple_symbol])
                                proba_A_BC = self.proba_rules[tuple_proba]
                                proba_B = table[i,k,self.all_non_terminal.index(B)]
                                proba_C = table[k,j,self.all_non_terminal.index(C)]
                                
                                full_proba = proba_A_BC*proba_B*proba_C
                                
                                if (table[i,j,self.all_non_terminal.index(A)]<full_proba):
                                    table[i,j,self.all_non_terminal.index(A)] = full_proba
                                    back[tuple([i,j,A])] = tuple([k,B,C])
        return table,back
    
    def parse_level_bellow(self,back,k,A,B,C,root,right_i,left_j):
        '''
        Fonction qui permet d'interpréter le résultat renvoyé par 
        la fonction predict_one_line. On lit en backward les résultats optimaux
        des sous problèmes de la programmation dynamique grâce à l'attribut back
        Param: @back: dictionnaire résultat sortie de predict one line
        @k: int point de coupure précédent
        @A: loi quand il n'y en a qu'une suele (root seulement)
        @B: symbole de gauche dans A -> BC
        @C: symbole de droite dans A -> BC 
        @root: arbre à compléter
        @right_i: coin gauche du parsing B représente le parsing entre right_i et k
        @right_j: coin droit du parsing  C représente le parsing entre k et left_j
        '''
        if k==-1:
            i = right_i
            j = left_j
            
            if tuple([i,j,A]) in back:
                root.left = Tree()
                root.right = Tree()
            
                new_k,new_B,new_C = back[tuple([i,j,A])]
                root.left.data = new_B
                root.right.data = new_C
                self.parse_level_bellow(back,new_k,A,new_B,new_C,root,right_i,left_j)
                
        else:  
            root.left.index_left = right_i
            root.left.index_right = k
            
            
            root.right.index_left = k 
            root.right.index_right = left_j
            
            if tuple([right_i,k,B]) in back:
                B_k,B_B,B_C = back[tuple([right_i,k,B])]
                #print(B_k,B_B,B_C)
                root.left.left = Tree()
                root.left.right= Tree()
                root.left.left.data = B_B
                root.left.right.data = B_C
                self.parse_level_bellow(back,B_k,A,B_B,B_C,root.left,right_i,k)
             
            if tuple([k,left_j,C]) in back:
                C_k,C_B,C_C = back[tuple([k,left_j,C])]
                root.right.left = Tree()
                root.right.right= Tree()
                root.right.left.data = C_B
                root.right.right.data = C_C
                self.parse_level_bellow(back,C_k,A,C_B,C_C,root.right,k,left_j)
                

  
        
    
    def parsing_line(self,line):
        '''
        Fonction qui crée l'arbre associé à une décomposition
        Param: @line: ligne à parser
        '''
        line_to_use = line.replace("\'","\' ")

        global punctuation_list
        for punc in punctuation_list:
            line_to_use = line_to_use.replace(punc,' '+ punc+ ' ')
            
        words = line_to_use.split()
        len_word = len(words)
        
        
        table,back = self.predict_one_line(line)
        
        root = Tree()
        
        root.data = 'SENT'
        root.index_left = 0
        root.index_right = len_word
        self.parse_level_bellow(back,-1,'SENT',0,0,root,0,len_word)
        
        
        return root
    
    def read_tree(self,list_to_complete,root):
        '''
        Fonction qui lit un arbre pour extraire la phrase à renvoyer
        Param: @root: racine de l'arbre à parser
        '''
        i = root.index_left
        j = root.index_right
        symbol = root.data
        if (('+' not in symbol)or(symbol =='P+D')):
            list_to_complete[i] += ' (' +symbol
        
        list_to_complete[j] += ')'
        if root.left != None:
            self.read_tree(list_to_complete,root.left)
        
        
                
        if root.right != None:
            self.read_tree(list_to_complete,root.right)
        
        
    
    def parse_line(self,line):
        '''
        Fonction predict qui donne le parsing dans le même format que le format initial SEQUOIA
        Param: @line: ligne à parser
        '''
        tree_parse = self.parsing_line(line)
        line_to_use = line.replace("\'","\' ")
        global punctuation_list
        for punc in punctuation_list:
            line_to_use = line_to_use.replace(punc,' '+ punc+ ' ')

        words = line_to_use.split()
        len_word = len(words)
        
        
        list_symbols = []
        
        for i in range(len_word+1):
            list_symbols.append('')
        
        self.read_tree(list_symbols,tree_parse)
        list_word_symbol = []
        list_word_symbol.append('(')
        for i in range(len_word):
            list_word_symbol.append(list_symbols[i])
            list_word_symbol.append(' ')
            list_word_symbol.append(words[i])
        
        list_word_symbol.append(list_symbols[-1])
        list_word_symbol.append(')')
        return ''.join(list_word_symbol)         
                
