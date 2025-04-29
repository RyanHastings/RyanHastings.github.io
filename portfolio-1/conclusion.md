
---
title: "5. Conclusion"
---

Here I have walked through the skeleton of the standard process for setting up a machine learning model. Along the way I introduced some concepts I seldom see discussed elsewhere, such as the mutual information coefficient and the use of feature selection using error contribution rather than prediction contribution. The final result was that a multilayer perceptron with recursive feature selection based on error contribution was clearly superior.

It might be tempting for the naive data scientist to just assume the more complex neural networks or deep learning systems would always automatically be better. But even within this project, we saw that the simplest classification model, logistic regression, performed significantly better than the more complex ensemble models. It can never be a foregone conclusion that the complicated model will automatically be better, and with any problem multiple models should be investigated. For example, when I was analyzing infant mortality in Indiana, the random forest performed best. In addition, simple models are easier to understand and interpret and have a lower computational cost. But the choice should always be guided by the requirements of the data more than anything else.
