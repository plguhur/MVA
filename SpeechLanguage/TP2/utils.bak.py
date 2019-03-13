#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np

def readfile(path_file):
    '''
    Lire les lignes d'un fichier texte
    Param: @pathfile: chemin 
    '''
    with open(path_file,'r') as file:
        lines = file.readlines()
    return lines

def writefile(pathfile,lines):
    '''
    Ecrire un fichier texte contenant chaque ligne
    Param: @pathfile: chemin 
    lines: Les lignes à écrire
    '''
    lines = '\n'.join(lines)
    with open(pathfile,'w')  as f:
        f.write(lines)

def level_of_each_symbol(line):
    '''
    Fonction qui donne le "niveau" de chaque symbole grâce au décompte des parenthèses
    Param: @line: ligne qui est un exemple de parsing sous le format SEQUOIA
    '''
    list_of_level = []
    split_line = line.split()
    cur_level = 0
    for word in split_line[1:]: #doesn't consider the first parenthese
        if '(' in word:
            cur_level = cur_level +1
            list_of_level.append(cur_level)
        elif ')' in word:
            nb_parenthese = len([v for v in word if v==')'])
            cur_level=  cur_level - nb_parenthese
    
    return list_of_level

def RemoveFunctional(word):
     '''
     Retirer les tirets des symboles
     '''
     split_word_hyphen = word.split('-')
     importantpart = split_word_hyphen[0]
     return importantpart

def create_data_from_lines(lines):
    '''
    FOnctin qui crée les règles de la grammaires à partir d'exemples présents
    dans lines 
    Param: @lines: list of str - exemple de lignes parsées
    Return: @lexicon Règle de la forme  \gamma -> A ou \gamma est un mot du vocabulaire et A un symbole (dictionnaire de set)
    @rules: dictionnaire de set qui contient les Règles de la forme A -> BC (Chomsky normal form)
    @probabilities_vocabl: dictionnaire qui contient les probabilités associées à chaque règle qui mènent à des ancres
    @probabilities_rules: dictionnaire qui contient les probabilités associées à chaque règle
    '''
    lexicon = {}
    rules ={}
    probabilities_rules = {} #key = tuple(root,tuple(symbols))
    normalization_rules = {} #key = root , number of time that the root appears
    
    probabilities_vocabl = {} #key = tuple(word,non-terminal symbol)
    normalization_vocab = {} #key = word , number of time that each word appear
    for line in lines:
        split_line = line.split()
        split_line_to_proceed = list(split_line)
        words_in_the_phrase = [w for w in split_line if '(' not in w]
        levels = level_of_each_symbol(line)
        
        symbols = []
        real_index_sent = []
        for index_line,l in  enumerate(split_line):
            if l not in words_in_the_phrase:
                symbols.append(l)
                real_index_sent.append(index_line)
        
        symbols = symbols[1:]
        real_index_sent = real_index_sent[1:]
        
        for ind_level,level in enumerate(levels):
            all_values = []
            if ind_level<(len(levels)-1):
                next_level = levels[ind_level+1]
            else:
                next_level = -10
            ind_plus = 1
            
            if (next_level==(level+1)):
                while ((next_level!=(level))):
                    if (next_level==(level+1)):
                        all_values.append(symbols[ind_level+ind_plus])
                    ind_plus = ind_plus + 1
                    if (ind_level+ind_plus)<(len(levels)):
                        next_level = levels[ind_level+ind_plus]
                    else:
                        next_level = -10
                        break
                    
            if len(all_values)>0:
                if len(all_values)==1:
                    split_line_to_proceed[real_index_sent[ind_level+1]] = split_line_to_proceed[real_index_sent[ind_level]]
                    symbols[ind_level+1] = split_line_to_proceed[real_index_sent[ind_level]]
                        
                else:
                    clean_values = [RemoveFunctional(re.sub('\(|\)','',i_word)) for i_word in all_values]
                    root = RemoveFunctional(re.sub('\(|\)','',symbols[ind_level]))
                    
                    if root not in rules:
                        rules[root] = set()
                        normalization_rules[root] = 0
                    rules[root].add(tuple(clean_values))
                    normalization_rules[root] +=1
                    tuple_proba = tuple([root,tuple(clean_values)])
                    if tuple_proba not in probabilities_rules:
                        probabilities_rules[tuple_proba]=0
                    probabilities_rules[tuple_proba] +=1
            
        for w in words_in_the_phrase:
            index_word_split_line = split_line_to_proceed.index(w)
            previous_word = split_line_to_proceed[index_word_split_line-1]
            
            if w.replace(')','') not in lexicon:
                lexicon[w.replace(')','')] = set()
                normalization_vocab[w.replace(')','')] = 0
            lexicon[w.replace(')','')].add(RemoveFunctional(previous_word.replace('(','')))
            normalization_vocab[w.replace(')','')] += 1
            tuple_proba = tuple([w.replace(')',''),RemoveFunctional(previous_word.replace('(',''))])
            if tuple_proba not in probabilities_vocabl:
                probabilities_vocabl[tuple_proba] = 0
            probabilities_vocabl[tuple_proba] += 1

    
    for items in probabilities_vocabl:
        word = items[0]
        normalization_term = normalization_vocab[word]
        probabilities_vocabl[items] = probabilities_vocabl[items]/normalization_term


    for items in probabilities_rules:
        rule = items[0]
        normalization_term = normalization_rules[rule]
        probabilities_rules[items] = probabilities_rules[items]/normalization_term
    
    return lexicon,rules,probabilities_vocabl,probabilities_rules
    


def create_lexicon(path_file):
    '''
    Crée les règles présentes dans un chemin
    '''
    lines = readfile(path_file)
    lexicon,rules,probabilities_vocabl,probabilities_rules = create_data_from_lines(lines)
    return lexicon,rules,probabilities_vocabl,probabilities_rules
            

def ConvertToChomsky(rules,probabilities_rules):
    '''
    Convertit toutes les règles afin de respecte le Chomsky normal form.
    Les units productions sont traitées directement dans la lecture
    Param: @rules :dictionnaires des règles (qui ne mènent pas vers des ancres)
    @probabilities_rules: probabilitées associées au règles
    '''
    new_rule = dict(rules)
    new_probabilities_rules = dict(probabilities_rules)
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
                    tuple_proba = tuple([new_symbol[0],concatenation])
                    new_probabilities_rules[tuple_proba] = 1
                
                new_tuple_proba = tuple([rule,tuple(new_symbol)])
                tuple_proba_to_remove = tuple([rule,tuple(set_of_symbols)])
                new_probabilities_rules[new_tuple_proba] = probabilities_rules[tuple_proba_to_remove]
                del new_probabilities_rules[tuple_proba_to_remove]
                
                new_rule[rule].add(tuple(new_symbol))
    return new_rule,new_probabilities_rules

def count_nb_of_non_terminal(rules):
    '''
    Compte le nombre de symboles 
    '''
    all_symbols = set()
    for rule in rules:
        all_symbols.add(rule)
        list_of_nn_symbols = rules[rule]
        for set_of_symbols in list_of_nn_symbols:
            for symbol in set_of_symbols:
                all_symbols.add(symbol)
    return len(all_symbols)
    
def all_symbols(rules):
    '''
    Retourne un set contenant tous les symboles présents dans les règles
    '''
    all_symbols = set()
    for rule in rules:
        all_symbols.add(rule)
        list_of_nn_symbols = rules[rule]
        for set_of_symbols in list_of_nn_symbols:
            for symbol in set_of_symbols:
                all_symbols.add(symbol)
    return all_symbols

def reverse_rules(rules):
    '''
    Inverse le dictionnaire de règles qui donnent A -> BC en BC -> A
    '''
    reverse_rules= {}
    for rule in rules:
        list_of_nn_symbols = rules[rule]
        for tuple_symbol in list_of_nn_symbols:
            if tuple_symbol not in reverse_rules:
                reverse_rules[tuple_symbol] = set()
            reverse_rules[tuple_symbol].add(rule)
    return reverse_rules
    
def remove_symbols(line):
    '''
    Permet de transformer une ligne parsé en ligne sans le parsing
    '''
    split_line = line.split()
    words_in_the_phrase = [w.replace(')','') for w in split_line if '(' not in w]
    return ' '.join(words_in_the_phrase)
    
