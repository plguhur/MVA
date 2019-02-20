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



## Classifying a correlated sequence of voice command
