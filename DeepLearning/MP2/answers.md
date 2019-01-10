---
title: "Deep Learning for Natural Language Processing"
author: "Pierre-Louis Guhur"
header-includes:
   - \usepackage{amsmath}
   - \DeclareMathOperator{\tr}{Tr}
   - \newcommand{\norm}[1]{\left\lVert#1\right\rVert}
output:
    pdf_document
---

# Deep Learning for Natural Language Processing

Pierre-Louis Guhur

## Question 1:
Using the orthogonality and the properties of the trace, prove that,
for X and Y two matrices:
$$W^* = \arg\min_{W\in O(\mathbb{R})} \norm{WX - Y}_F = U V^T,$$
with $U \Sigma V^T = SVD(Y X^T)$.

*Answer:* we see first that:

$$W^* = \arg\min_{W\in O(\mathbb{R})} \norm{WX - Y}_F^2$$
Developing the Frobenius norm's definition and using orthogonality of $W$:
$$W^* = \arg\min_{W\in O(\mathbb{R})} \norm{X}_F^2  + \norm{Y}_F^2  - 2 \tr{X^HWY}$$
Consequently:
$$W^* = \arg\max_{W\in O(\mathbb{R})} \tr{X^HWY} = \arg\max_{W\in O(\mathbb{R})} \tr{X^HWY}^2 $$
Using trace properties and SVD:
$$W^* = \arg\max_{W\in O(\mathbb{R})} \tr{YX^HW} = \arg\max_{W\in O(\mathbb{R})} \tr{U\Sigma V^TW}^2 =  \arg\max_{W\in O(\mathbb{R})} \tr{\Sigma V^TWU}^2$$

Using Cauchy-Schwarz inequality:
$$W^* \leq \tr{\Sigma}^2\tr{V^TWU}^2 = \tr{\Sigma}^2$$.

The cost function is strictly convex and the maximal is reached for $W=UV^T$.

Therefore, $W^*=UV^T$.
