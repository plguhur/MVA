#!/usr/bin/env python3

"""
Pierre-Louis Guhur - TP2 SL MVA
-------------------------------

Parse a sentence using the CYK algorithm
"""


from utils import *
from argparse import ArgumentParser
from pathlib import Path
import os
import pickle
import numpy
from operator import itemgetter
import re
import numpy
import itertools 
import numpy as np

from nltk import word_tokenize 
from nltk.util import ngrams


def preprocess(string):
    return word_tokenize(unidecode(string.lower()))

def levenshtein_distance(s, t, ns=-1, nt=-1):
    """ Levenshtein distance in a recursive mode
    >>> levenshtein_distance("flaw", "lawn")
    2
    """
    if ns == 0:
        return nt
    if nt == 0:
        return ns
    if ns == -1:
        ns = len(s)
    if nt == -1:
        nt = len(t)
        
    cost = 0 if s[ns-1] == t[nt-1] else 1

    return min(levenshtein_distance(s, t, ns-1, nt) + 1,
                 levenshtein_distance(s, t, ns, nt-1) + 1,
                 levenshtein_distance(s, t, ns-1, nt-1) + cost)


def eps_nn_levenshtein(s, ss, eps=1):
    """ Return the words in ss that are eps-close to s.
    This is very long to compute when len(ss) is big """
    distances = [levenshtein_distance(s, t) for t in ss]
    return [ss[i] for i, distance in enumerate(distance) if distance <= eps]
    

def generate_neighbors(word, dico, 
        alphabet="abcdefghijklmnopqrstuvwxyzàèéçâê"):
    """ Generate all possible words with 1 modification that are still in dico """
    neighbors = []
    for i in range(len(word)):
        for letter in alphabet:
            if word[:i] + letter + word[i+1:] in dico:
                neighbors.append(word[:i] + letter + word[i+1:])
            if word[:i] + letter + word[i:] in dico:
                neighbors.append(word[:i] + letter + word[i:])
        if word[:i] + word[i+1:] in dico:
            neighbors.append(word[:i] + word[i+1:])
    return neighbors


class Embedding:
    
    def __init__(self, pickle_file=Path("system") / "data" / "polyglot-fr.pkl"):
        self.load_polyglot(pickle_file)
        
        # Special tokens
        self.Token_ID = {"<UNK>": 0, "<S>": 1, "</S>":2, "<PAD>": 3}
        self.ID_Token = {v:k for k,v in self.Token_ID.items()}

        # Map words to indices and vice versa
        self.word_id = {w:i for (i, w) in enumerate(self.words)}
        self.id_word = dict(enumerate(self.words))

        # Noramlize digits by replacing them with #
        self.DIGITS = re.compile("[0-9]", re.UNICODE)

        # Number of neighbors to return.
        self.k = 5
           
        self.search_words = [w.lower() for w in self.words]
        
        
    def load_polyglot(self, pickle_file=Path("data") / "polyglot-fr.pkl"):
        if not os.path.isfile(pickle_file):
            raise ArgumentError("Polyglot file does not seem to exist... Please download it on https://doc-0c-5g-docs.googleusercontent.com/docs/securesc/222cjijgelfc0p72t59ng1jsibmf5vbc/51qmicktg8na0est4vk3ppr2q6h4tr0u/1552824000000/10341224892851088318/02980337768698978614/0B5lWReQPSvmGekFBUjZDMzM5NDA")

        with open(pickle_file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            self.words, self.embeddings = u.load()
        
    def normalize(self, word):
        """Pre process word to find it in word2idx.
        >>> e.normalize('marketing')
        'markting'
        """
        if not word in self.words:
            word = self.DIGITS.sub("#", word)
            
        if word in self.words:
            return word
        
        if word.title() in self.words:
            return word.title()
        if word.lower() in self.words:
            return word.lower()
        if word.upper() in self.words:
            return word.upper()
        
        # we find the closest word in the Levenshtein distance meaning
        neighbors = generate_neighbors(word.lower(), self.search_words)
        # Because we use lower case, we need to rectify case
        neighbors = [self.search_words.index(n) for n in neighbors] 
        neighbors = [self.words[n] for n in neighbors] 
        
        return neighbors


    def l2_nearest(self, word_index, k):
        """Sorts words according to their Euclidean distance.
           To use cosine distance, embeddings has to be normalized,
           so that their l2 norm is 1."""

        e = self.embeddings[word_index]
        distances = (((self.embeddings - e) ** 2).sum(axis=1) ** 0.5)
        sorted_distances = sorted(enumerate(distances), key=itemgetter(1))
        return zip(*sorted_distances[:k])


    def knn(self, word, k=-1):
        """
        >>> e.knn("markting")
        ['marketing', 'management', 'média', 'logistique', 'design']
        """
        if k == -1: 
            k = self.k
        word = self.normalize(word)
        if word is None or word == []: 
            return None
        if isinstance(word, (list,)):
            neighbors = word
            neighbors = [self.knn(word, k) for word in neighbors]
            neighbors = [n for nn in neighbors for n in nn]
            return neighbors
        
        word_index = self.word_id[word]
        indices, distances = self.l2_nearest(word_index, k)
        neighbors = [self.id_word[idx] for idx in indices]
        return neighbors
 
        


class OOV:
    
    def __init__(self, sentences, _lambda=0.1):
        self.build_word2idx(sentences)
        self.build_transition_matrix(sentences, _lambda=0.1)
        self.embedding = Embedding()
        self.words = self.word2idx.keys()
        self.words_by_length = {}
        for word in self.words:
            n_letters = len(re.sub("[^a-z]", "", word))
            if n_letters in self.words_by_length:
                self.words_by_length[n_letters] += [word]
            else:
                self.words_by_length[n_letters] = [word]
    
    def build_word2idx(self, sentences):
        words = [preprocess(s) for s in sentences]
        words = set([s for ss in words for s in ss])
        self.word2idx = {w:i for i, w in enumerate(words)}
        self.idx2word = dict(enumerate(self.word2idx))
        return self.word2idx
    

    def build_transition_matrix(self, sentences, _lambda=0.1):
        """ Build a transition matrix using linear interpolation 
        given a list of sentences """
        all_words = preprocess(" ".join(sentences))
        known_words = self.word2idx.keys()
        all_words = [self.word2idx[w] for w in all_words if w in known_words]
        all_words = np.asarray(all_words, dtype=int)
        n_all_words = len(all_words)

        # find all bigrams in sentences
        bigrams = np.zeros((2*n_all_words, 2), dtype=int)
        n = 0
        for line in sentences:
            token = preprocess(line)
            bigram = list(ngrams(token, 2)) 
            for i in range(len(bigram)):
                bigrams[i+n, :] = [self.word2idx[bigram[i][0]], self.word2idx[bigram[i][1]]]
            n += len(bigram)

        # case where initialisation was too large
        bigrams = bigrams[:n+1, :] 

        # build the transition matrix
        n_words = len(self.word2idx)
        Puni = np.asarray([np.sum(all_words == i) for i in range(n_words)], dtype=float)
        Puni = np.broadcast_to(Puni, (n_words, n_words))/float(n_all_words)
        Pmle = np.zeros((n_words, n_words), dtype=float)
        for a, b in bigrams:
            Pmle[a, b] += 1
        Pmle /= n
        
        self.transition_matrix = _lambda*Puni + (1 - _lambda)*Pmle
        return self.transition_matrix


    def correct_word(self, before, word, after):
        """ Given a word and its context (the word before and the word after), we try to find its best match 
        #TODO try to split word: "thisidea" -> "this idea"
        >>> oov.correct_word("la", "formtion", "intra-procedurale")
        'formation'
        >>> oov.correct_word("la", "fewrpgjoh", "intra-procedurale")
        ','
        >>> 
        """
        word = unidecode(word.lower())
        before = unidecode(before.lower())
        after = unidecode(after.lower())
        
        if word in self.words:
            return word 
        
        # First candidates are simple spelling errors
        candidates = generate_neighbors(word, self.word2idx.keys())
        
        # Then, we add suggestions from Polyglot
        neighbors = self.embedding.knn(word)
        if neighbors is not None:
            for neighbor in neighbors:
                candidates += generate_neighbors(word, self.word2idx.keys())
        
        # If we still have nothing, we just pray:
        if candidates == []:   
            candidates = self.find_best_guess2(before, word, after)
            
        return self.find_best_candidate(before, candidates, after)

    
    def find_best_guess(self, before, word, after):
        """ This function outputs only "," """
        if not before in self.words:
            before = ""
        if not after in self.words:
            after = ""
            
        if before == "" and after == "":
            idx = np.argmax(np.diag(self.transition_matrix))
        elif before == "":
            after = self.word2idx[after]
            idx = np.argmax(self.transition_matrix[:, after])
        elif after == "":
            before = self.word2idx[before]
            idx = np.argmax(self.transition_matrix[before, :])
        else:
            likelihood = self.transition_matrix[self.word2idx[before], :]   \
                  *self.transition_matrix[:, self.word2idx[after]]
            idx = np.argmax(likelihood)
        return [self.idx2word[idx]]
    
    
    def find_best_guess2(self, before, word, after):
        """ We really don't know the word so we try to find a word with the same number of letter """
        n_letters = len(re.sub("[^a-z]", "", word))
        candidates = self.words_by_length[n_letters]
        return candidates
        
            
    def find_best_candidate(self, before, candidates,  after):
        """ Return the most likely word to be between "before" and "after" 
        among a list of candidates using a transition matrix 
        >>> find_best_candidate(["formation", "qui", "propos"], "la", "intra-procedurale", 
        'formation'
        """
        if not before in self.words:
            before = ""
        if not after in self.words:
            after = ""
            
        cand_idx = [self.word2idx[c] for c in candidates]

        if before == "" and after == "":
            scores = [self.transition_matrix[c, c] for c in cand_idx]
        elif before == "":
            after = self.word2idx[after]
            scores = [self.transition_matrix[c, after] for c in cand_idx]
        elif after == "":
            before = self.word2idx[before]
            scores = [self.transition_matrix[before, c] for c in cand_idx]
        else:
            before = self.word2idx[before]
            after = self.word2idx[after]
            scores = [self.transition_matrix[before, c] * \
                  self.transition_matrix[c, after] for c in cand_idx]

        return candidates[sorted(range(len(scores)), key=scores.__getitem__)[-1]]
   


               
if __name__ == "__main__":

    db = Dataset("data/sequoia-corpus+fct.mrg_strict")
    sentences = db.get_sentences(TESTING)

    oov = OOV(sentences)
    
