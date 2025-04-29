
---
title: "Classification Model With Numerical Features"
---

The first example I would like to give is the creation of a classification model using numerical regressors. I will walk through a skeleton of the standard techniques for approaching a data set, first doing initial exploration, then going into model selection. Instead of this simply being a straightforward (and boring!) stroll, I am also going to take up a few topics I rarely find discussed elsewhere and explore them a bit.

There are three fairly novel topics herein, all of which I found on [*Towards Data Science*](https://towardsdatascience.com/) posts. One is the use of a mutual information coefficient based on relative entropy to measure associations, instead of something more common such as a correlation coefficient. Another is the use of calibration on the probabilistic predictions. Finally, there is an exploration of whether or not it is more effective to use the contribution of a feature to error rather than prediction for feature selection. All work is done in python.

1. [Construction of the Data Set](1-construction.md)
2. [Data Exploration](data-exploration.md)
3. [Modeling](modeling.md)
4. [Model Feature Selection](feature-selection.md)
5. [Conclusion](conclusion.md)
