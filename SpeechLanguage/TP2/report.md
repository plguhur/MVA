# Algorithms for Speech and NLP TP2: Robust Probabilistic Parser
Pierre-Louis Guhur
MVA ENS Paris-Saclay
pierre-louis.guhur@ens-paris-saclay.fr

This report relates the development of a robust probabilistic constituency parser for French based on CYK algorithm and the PCFG model. 
Dataset was extracted from the SEQUOIA database.
 
# 1. Parsing sentences with the PCFG model

In formal language theory, a grammar is a set of production rules that describe how to form strings from the language's alphabet according to the language's syntax. 
In the Chomsky hierarchy, context-free grammar (CFG) relates to a certain type of grammar, in which the left-hand side of each production rule consists of only a single nonterminal symbol.

A [robabilistic context-free grammar (PCFG) assigns to each production rule a probability, such that the probability of a derivation is the product of the probabilities of the productions used in that derivation.

The first task was to parse the SEQUOIA dataset. For example, for the sentence: 

```    ( (SENT (PP-MOD (P En) (NP (NC 1996))) (PONCT ,) (NP-SUJ (DET la) (NC municipalité)) (VN (V étudie)) (NP-OBJ (DET la) (NC possibilité) (PP (P d') (NP (DET une) (NC construction) (AP (ADJ neuve))))) (PONCT .))) ```

The functional labels are first ignored, such that the sentence becomes:

```    ( (SENT (PP (P En) (NP (NC 1996))) (PONCT ,) (NP (DET la) (NC municipalité)) (VN (V étudie)) (NP (DET la) (NC possibilité) (PP (P d') (NP (DET une) (NC construction) (AP (ADJ neuve))))) (PONCT .))) ```

Then, the sentence is recursively transformed as a tree structure. I used here the function `ntlk.fromstring`.




# 2. Learning the model 

To build a parser based on PCFG and CYK, we need first to build rules, transform them in the normal Chomsky form and finally estimate the associated probabilities to each rule. 

The estimation is done statistically by taking the frequency of each rule. An important size of the trainig set is then assumed.
In practice, this is not the case. This is all the more an issue, than some words appear in the training, but in the testing. 

# 3. Dealing with out-of-vocabulary (OOV) words

The corpus contains several challenges: some words are not frequent enough to appear both on the training set and the testing set, whereas other words have spelling errors (such as "barisienne" instead of "parisienne"). 

To deal with OOV words, I used first the "Polyglot" library, which provides an embedding for a large corpus of French words. Unfortunately, more than 40% of words in the testing set were not present in this corpus. 
That is why, I extended the search of OOV words by several manners:

- words are tested with different upper and lower cases
- numbers are transformed with a unique unicode symbol
- each combination of 1-Levenshtein distance is tested

If the word is finally found on the Polyglot's corpus, I searched its $K$ nearest neighbors with the L2-distance. I manually set $K=5$, as it produces reasonable words; however, $K$ could also be chosen as the one providing the best overall result. Because most of these neighbors are not in the training set, I had to generate of all possibilites in the training set of these neighbors.


For example, if the testing set contains the word "markting", I first generate all words distant from "markting" by one in the Levenshtein distance (such that "merkting", "marktin", "marketing"...), then I keep only those which are present in Polyglot's corpus (here it is only "marketing"). The KNN of "marketing" are: "marketing", "management", "média", "logistique", "design". Then, I search in the training set whether these words, or some similar words close by a Levenshtein distance of 1, exist or nots. This creates a pool of candidate word. 

I search
If the wor
Not found 

# 5. Further work

This work was particularly interesting to execute for the OOV words, as it allows to find new ideas. 
It also opens new questions. For example, I only corrected OOV words during the testing phase, assuming that errors appearing in the training set would not matter. Indeed, if the training set contains "markting" instead of "marketing", all words "marketing" in the testing set will be transformed as "markting". However, this assumption is not always followed. For example, the word "merketing" would not be corrected as it has a Levenshtein distance of 2 to "markting", but it would have be corected with a better training set; similarly, numbers should all be encoded as special characters during the training and the testing. 

I should also consider words where a space is missing. For example, "thatmovie" could be split into two words, by testing each splitting possibility, until Polyglot testifies the existence of the two words.

-> cut words


After that, we applu the probabilistic version of CYK algorithm (Cocke-Younger-Kasami).
The
# 3. Making the 

Je gère les règles avec plus de 2 éléments en sortie en
faisant des fusions itératives tel que : A → BCD avec
probabilité p devient A → ’B + C’D avec probabilité
p et j’ajoute la règle ’B + C’ → BC avec probabilité
1. Je traite tous les cas avec des sorties plus grandes que
2 ainsi, introduire le symbole ’+’ permet de repérer les
règles ajoutés que l’on ne voudra pas afficher à la fin.

3.2. Evaluer sur la base de test
Phrase Originale : Affaire politico-financière
Prediction : ( (SENT (NC Affaire) (AP (NC politico)
(PONCT -)) (ADJ financière))))
Solution : ( (SENT (NP (NC Affaire) (AP (PREF politico) (ADJ financière)))))

2.4. Probabilistic CYK
Puisque notre espace des règles est probabilisé, on utilise l’algorithme de CYK probabilisé basé sur la programmation dynamique pour trouver le parsing optimal
de plus grande probabilité. L’algorithme est très bien
expliqué dans l’ouvrage ([1], chap 13). J’ai créé une
classe ProbabilisticCYK qui s’appuie sur les outils précedemment décrits pour apprendre ses attributs
(une grammaire de règles probabilisées), on apprend cela
avec la méthode fit. Ensuite, la fonction predict one line
permet fait l’algorithme CYK probabilisé à proprement
parlé pour effectuer le decodage d’une phrase. Je stocke
l’étiquetage morpho-syntaxique dans un objet arbre (une
racine, une branche à gauche et une branche à droite, plus
l’étendu sur lequel l’étiquetage est valide). La fonction finale pour avoir le parsing sous le format initial est donné
par la méthode parse line.

Phrase Originale : Wikipédia : ébauche droit.
Prediction : ( (SENT (NP Wikipédia) (PONCT :)) (NP
(NC ébauche) (AP droit))) (PONCT .)))
Solution : ( (SENT (NP (NPP Wikipédia)) (PONCT :) (NP
(NC ébauche) (NC droit)) (PONCT .)))

3.3. Evaluer sur des exemples illustratifs
Phrase Originale : Je vais au cllege.
Prediction : ( (SENT (VN (CLS Je) (V vais)) (PP (P+D
au) (NP (NC cllege) (PONCT .)))))
On observe que l’algorithme fonctionne dans la majorité des cas même si l’on ne prédit pas exactement la même
chose que la vraie solution (on ne peut pas prédire les unit
productions par exemple). La correction orthographique
permet de corriger certains points (comme dans l’exemple
précédent) mais elle mène à des absurdités parfois. Il est
préférable de ne pas l’utiliser dans un premier temps. Le
tiret est séparé de politico à cause de la manière avec laquelle je fais ma tokenization. Il faudrait mieux gérer certains cas particuliers.

2.5. Améliorer le résultat grâce à des libraires
complémentaire
Afin d’améliorer le résultat final obtenu, je laisse à l’utilisateur la possibilité d’utiliser le PoS Tagger de standford
comme possible complément à mon dictionnaire qui associe un symbole terminale au vocabulaire (cet algorithme
est utilisé uniquement pour trouver un symbole associable
à une ancre pas présentes dans le lexique). J’offre aussi
la possibilité de s’appuyer sur la librairie enchant afin de
corriger les fautes d’orthographes présentes dans le texte
avant de faire le parsing (cette option fonctionne moyennement car la correction n’est pas toujours très bonne).

3.4. Amélioration envisageable
Afin d’améliorer l’algorithme on pourrait déjà commencer par utiliser une base d’entrainement plus grande
pour améliorer la qualité de notre grammaire. Utiliser une
base plus grande permettrait d’ajouter les labels fonctionnelles qui sont améliorants d’après [1]. On pourrait
également effectuer la correction orthographique dans les
cas où l’on est certain de la correction.

3. Résultats et tests
L’entrainement de la grammmaire sera faite sur 80% de
la base SEQUOIA (on prendra les 80% premier), et 10%
sera utilisé pour les tests (comme requis).

4. Conclusion
Ce TD, bien que long et assez prennant lorsque l’on
veut tout implémenter seul aura été une excellente occasion de manipuler les Parser. On voit que l’on peut obtenir des résultats très intéressants en utilisant les préceptes
du cours. On voit aussi comment l’on pourrait combiner le TD 3 avec celui là et à quoi aurait pu servir le
préprocessing.

3.1. Evaluer numériquement le modèle
Il aurait été intéressant d’obtenir une valeur numérique
sur les performances de mon algorithme. Néanmoins, il est
très difficile d’évaluer de manière automatique un modèle
puisque notre algorithme peut par exemple créer un regroupement en trop ce qui provoquera un décalage par rapport au parsing proposé dans SEQUOIA et il faudrait donc
trouver l’alignement idéal. Afin d’évaluer l’algorithme

Références
[1] D. Jurafsky and J. H. Martin. Speech and language processing, volume 3. Pearson London :, 2014.

1


