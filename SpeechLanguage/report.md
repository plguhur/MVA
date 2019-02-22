# Speech Commands MVA 2019

Student: Pierre-Louis Guhur, pierre-louis.guhur@ens-paris-saclay.fr

## Abstract

Audio speech recognition is a difficult field, due to the high variations between peoples, rhythms, or intonations.
This report aims at presenting a few method.

## Classifying a segmented voice command

Given a set of words $\left(W_i \right)_{i\leq n}$ and a set of sound features $(X_i)_{i\leq m}$, we want to learn a mapping between these two sets.

### From a sound to its features

Two transformations are proposed:

- MFCCs are the coefficients of the mel-frequency cepstrum, a representation of the short-term power spectrum of a sound;
- Mel-filterbanks are the results of the product between the spectrogram and the Mel-filters.

Mel-filterbanks are more correlated than MFCCs, but they correspond better to the nature of the speech signal and the human percpetion of such signals.

I have slightly modified the parameters to the Spectral instance. In particular, I changed the frequency bounds to $[10,8000]$ and the length of the Hamming window to 0.2. This allows me to achieve better classifying results.

### Training classifiers to find the mapping

I train several classifiers to learn the representation between the sound features and the words. Here are the obtained accuracies:

| Classifier | MFCC | Mel-filterbanks |
|---+---+---|
| Logistic regression | $34.2\%$ | $29.0\%$ |
| MLP | $68.8\%$ | $54.0\%$ |
| Random forest | $61.6\%$ | $56.8\%$ |


Hyper-parameters were tuned in order to match best results.
The multi-layer perceptron (MLP) has 700 hidden layers. The logistic regression has a tolerance of 0.001, using the LBFGS solver. The random forest has a maximum of depth of $10$, and $500$ estimators.

In this setting, MFCCs seem to outperform Mel-filterbanks. However, it depends on the choice of the hyper-parameters. One explanation is related to the correlation inside the Mel-filterbanks.


## Classifying a  non-correlated sequence of voice command

Now, the challenge is to decode a sequence of speech commands. We will compare several methods from dynamic programming to find the best decoded sequence.

To begin with, we consider the case where all words in the sequence are non-correlated. This means that the sequence is made from words picked randomly with a uniform distribution.

In this case, the decoding consists in decoding the sequence words by words independently with on the classifier trained before. This is equivalent in doing a greedy algorithm, namely an algorithm that optimizes locally without considering a global structure.

Doing so achieves a WER of:

- 0.55 on a subset of the train set;
- 0.51 on the test set.

We can observe that the train WER and the test WER are closed, or, in other words, the prediction did not over-fit.



## Classifying a correlated sequence of voice command with a beam search

Voice commands have a certain structure: the presence of a word has implication on the following words.
Another possibility consists in learning a language model prior to decoding a sequence of words.

### Building a language model

The language model can be learned with a n-gram model. Here, we focus on the bigram model, whose equations are recalled in the notebook.

The language model is then built as a transition matrix $T[i,j] = P(W_j|W_i)$. Given the limited size of the dictionary (30 words), the matrix remains small and no sparsity algorithm is required.

Because two words are missing in the ngrams ("bed" and "dog"), while other words only appear with certain words ("eight" with "right"), many zeroes appear in the transition matrix. This is a limitation, in the meaning that it excludes some paths to be visited. It might lead to systematic errors as explained in Question 2.9. Instead, I used a Laplacian smoothing with a degree of 1.

### Building the beam search


The beam search is a first possibility to improve the decoding. For each word, the $K$ best possibilities for the next word are kept.

To improve the numerical stability of the beam search, I used `-log` instead of direct probabilities.

However, such algorithm does not use the transition matrix. Instead, I suggest to do a beam search with the transition matrix. This method is referred later as \emph{BeamTransition}.

We want to find the most likely sequence of words:
$$P(\hat{W}) = P(O|W)P(W),$$
where $P(O|W)$ is the likelihood probability and $P(W)$ is the language model.

Assuming independence in the distribution of the likelihood and the bigram model, we have:
$$P(O|W) = \prod_i P(O_i|W_i),$$
$$P(W) = P(W_1)P(W_2|W_1)...P(W_{N}|W_{N-1}).$$

Therefore, we obtain that:
$$P(\hat{W}) = \left[P(O_1|W_1)P(W_1)\right] \left[P(O_2|W_2)P(W_2|W_1)\right] ...\left[P(O_N|W_N)P(W_{N}|W_{N-1})\right].$$

And using the equations:
$$P(X_i|W_i)  \propto P_{\text{discriminator single word}}(W_i|X_i),$$
and,
$P(W_i) = \text{cst}$
we finally get:
$$P(\hat{W}) \propto P_{disc}(W_1|O_1) \left[P_{disc}(W_2|O_2)P(W_2|W_1)\right] ...\left[P_{disc}(W_N|O_N)P(W_{N}|W_{N-1})\right].$$

Hence, we can do a beam search using "P_{disc}(W_i|O_i)P(W_i|W_{i-1})" instead of "P_{disc}(W_i|O_i)".

Both methods still suffer from the causality assumption: a word is only decoded with former and latter words. The results are the following with a beam size of 5:

| Set | Beam search | \emph{BeamTransition} |
|---+---+---|
| Training | $48.4\%$ | $40.7\%$ |
| Testing | $48.5\%$ | $39.2\%$ |

Therefore, the beam search improved results over the greedy algorithm, but using the transition matrix in the beam search has slightly improved further the results.

It is noticeable that results in the training and testing sets are particularly closed. This might be also explained by the similarities between both sets. In particular, they have the same transition matrices.



## Classifying a correlated sequence of voice command with a Viterbi algorithm

The Viterbi algorithm is much stronger than the \emph{BeamTransition} algorithm, because it relies on a forward-backward method.
In particular, the \emph{BeamTransition} algorithm is not able to deal with errors in the first word.

The Viterbi algorithm consists in two steps:

- updating the probabilities of the most probable state sequence:
$$V_{t,k} = \max _{x\in S}\left(\mathrm {P} {\big (}y_{t}\ |\ k{\big )}\cdot a_{x,k}\cdot V_{t-1,x}\right)$$

- updating consequently the states:
$$V_{t,k}=\arg\max _{x\in S}\left(\mathrm {P} {\big (}y_{t}\ |\ k{\big )}\cdot a_{x,k}\cdot V_{t-1,x}\right).$$


The obtained results are once again improved:

| Set | Viterbi |
|---+---|
| Training | $27.9\%$ |
| Testing | $29.4\%$ |

## Conclusion

The report shown preliminary methods for automatic speech recognition. It was assumed all along that the words can easily be segmented.

A new method based on the beam search was proposed, but it does not win against the Viterbi algorithm, as it is only forward.
