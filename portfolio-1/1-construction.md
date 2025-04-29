
---
title: "1. Construction of the Data Set"
---

1. [[#Import Libraries]]
2. [[#Create dataset]]

## Import Libraries

As always, we begin by importing necessary functions from various libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `shap`, and `scikit-learn`.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import shap

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_regression, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    brier_score_loss, log_loss, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.frozen import FrozenEstimator
```

## Create dataset

Next we create the dataset. It will have 50,000 samples and thirty features, and it will be associated with a binary class. 0.7 of the samples which will be negative (0) and the remainder positive (1). 20 of the features will be informative and 5 redundant.

```python
# create data set
X, y = make_classification(
    n_samples = 50000,
    n_features = 30,
    n_classes = 2,
    n_informative = 20,
    n_redundant = 5,
    shuffle = True,
    random_state = 42,
    n_clusters_per_class = 1,
    class_sep = 1,
    weights = [0.7,]
)

# concantenate them into one frame and create column names that are simply X plus the number of column position
data = pd.DataFrame(X, columns = ['X' + str(n) for n in range(30)]).assign(target=y)

# take a look at the data frame
data.head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X0</th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
      <th>...</th>
      <th>X21</th>
      <th>X22</th>
      <th>X23</th>
      <th>X24</th>
      <th>X25</th>
      <th>X26</th>
      <th>X27</th>
      <th>X28</th>
      <th>X29</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.107030</td>
      <td>5.857486</td>
      <td>-0.501960</td>
      <td>-0.630735</td>
      <td>0.547076</td>
      <td>-4.274163</td>
      <td>15.300989</td>
      <td>4.762018</td>
      <td>3.399235</td>
      <td>-1.875450</td>
      <td>...</td>
      <td>-1.614864</td>
      <td>2.706797</td>
      <td>0.636519</td>
      <td>-5.180686</td>
      <td>5.917957</td>
      <td>2.218638</td>
      <td>3.705592</td>
      <td>-2.352288</td>
      <td>5.437839</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.617422</td>
      <td>-2.260785</td>
      <td>-0.920099</td>
      <td>4.851533</td>
      <td>-0.975667</td>
      <td>0.393628</td>
      <td>1.223156</td>
      <td>-0.876361</td>
      <td>-0.552878</td>
      <td>0.817603</td>
      <td>...</td>
      <td>3.005481</td>
      <td>-2.238610</td>
      <td>-0.160044</td>
      <td>-1.743732</td>
      <td>-5.538825</td>
      <td>3.567385</td>
      <td>-5.170720</td>
      <td>0.283220</td>
      <td>-0.612102</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.923485</td>
      <td>0.596967</td>
      <td>-4.973955</td>
      <td>-1.475220</td>
      <td>-5.166408</td>
      <td>2.858951</td>
      <td>9.810482</td>
      <td>3.263809</td>
      <td>-0.070150</td>
      <td>-0.952388</td>
      <td>...</td>
      <td>-3.032525</td>
      <td>-1.037887</td>
      <td>1.729179</td>
      <td>-0.675174</td>
      <td>3.160159</td>
      <td>1.899279</td>
      <td>-1.805782</td>
      <td>-1.469503</td>
      <td>1.952417</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.132601</td>
      <td>-0.817862</td>
      <td>1.830852</td>
      <td>-8.783676</td>
      <td>6.366377</td>
      <td>-1.983822</td>
      <td>4.238954</td>
      <td>-0.506094</td>
      <td>0.738517</td>
      <td>-4.967117</td>
      <td>...</td>
      <td>-5.471895</td>
      <td>-4.347002</td>
      <td>-1.605614</td>
      <td>0.099890</td>
      <td>1.513436</td>
      <td>-1.348966</td>
      <td>-1.273202</td>
      <td>-1.341054</td>
      <td>-3.541808</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.564079</td>
      <td>-2.209570</td>
      <td>0.011249</td>
      <td>-2.332179</td>
      <td>-1.424068</td>
      <td>2.019029</td>
      <td>5.746230</td>
      <td>-1.138455</td>
      <td>-3.561440</td>
      <td>-2.635148</td>
      <td>...</td>
      <td>-2.375369</td>
      <td>2.071340</td>
      <td>0.251526</td>
      <td>1.704671</td>
      <td>0.401561</td>
      <td>-2.491117</td>
      <td>-0.755343</td>
      <td>-0.779528</td>
      <td>4.030070</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>

