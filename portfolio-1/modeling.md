
---
title: "3. Modeling"
---

Now we can begin with modeling. We will first see whether predictive modeling is even a viable option. That is, we will see whether the categories have some amount of separation in a reduced dimensional space. Then, if it is an option, we will proceed first with some individual looks at models (which will include calibration of the probabilities), and conclude with feature selection to come up with a viable model.

- [[#Separability of Target Category]]
- [[#Creating the Train, Validation, and Test Sets]]
- [[#Initial Modeling Exploration]]
- [[#Calibration of Probabilities]]
## Separability of Target Category

For this step, we need some method for reducing the dimensions to see easily. I will use the simple principle component analysis (PCA) method.

```python
# do the PCA
X = PCA(n_components=2).fit_transform(data)

# make the plot
fg,ax = plt.subplots(1,1,figsize=(4,4))
negscat = ax.scatter(X[data['target']==0,0],X[data['target']==0,1],c='black',s=1,label='negative');
posscat = ax.scatter(X[data['target']==1,0],X[data['target']==1,1],c='red',s=1,label='positive');
ax.set_title("Principle Component Analysis");
ax.legend(handles=[negscat,posscat]);
```

![png](Classification%20Model%20for%20Portfolio%20d0_17_0.png)

So, there is some degree of separability; thus modeling is possible.

## Creating the Train, Validation, and Test Sets

Now we're going to split the set into three parts: the training set, a validation set, and a test set. The validation set will be used for calibrating probabilities and computing feature importance. The test set will be used to evaluate the quality of the model.

```python
# split the data frame into three parts, first by taking a random sample of 35,000 from the total
TrainSet = data.sample(35000, random_state = 42)

# next we drop those and select 10,000 for our validation set
temp = data.drop(TrainSet.index)
ValSet = temp.sample(10000, random_state = 42)

# the test set is whatever remains
TestSet = temp.drop(ValSet.index)

# a quick function for splitting each of the sets into X and y
def get_XY(df):
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]
    return X,y

X_train, y_train = get_XY(TrainSet)
X_val, y_val = get_XY(ValSet)
X_test, y_test = get_XY(TestSet)
```

## Initial Modeling Exploration

For the sake of simplicity, I will only experiment with four classifiers: logistic regression, random forest, AdaBoost, and a multilayer perceptron neural network. Prior experimentation showed that for the logistic regression, the features needed to be scaled, so I create a pipeline with `StandardScaler()`.

I create a dictionary with the models, which I then loop through first to train them, and then to create and display confusion matrices. Here I train the models with the training set, and I use the validation set for the confusion matrices.

```python
# create a dictionary of the estimators
estimators = { 'Logistic Regression':
    Pipeline(
        [
            ('scaler',StandardScaler()),
            ('clf',LogisticRegression())
        ]
    ),
    'random forest':RandomForestClassifier(random_state = 42, max_depth = 5, max_features = 1),
    'AdaBoost':AdaBoostClassifier(random_state = 42),
    'Multilayer Perceptron':MLPClassifier( alpha = 1, max_iter = 1000, random_state = 42 )
}

# create empty dictionary that will contain the confusion matrices
confusion_matrices = {}

# loop through estimators, train them, and get the confusion matrix for each
for estimator in estimators.keys():
    cloned = clone(estimators[estimator])
    cloned.fit(X_train,y_train)
    y_pred = cloned.predict(X_val)
    confusion_matrices[estimator] = confusion_matrix( y_val, y_pred, normalize = 'all')

# create plots
fg, ax = plt.subplots(2,2)

labels = ['negative','positive']

count = 0
for i in range(2):
    for j in range(2):
        temp = list(estimators.keys())[count]
        ConfusionMatrixDisplay(confusion_matrices[temp],display_labels = labels).plot(ax=ax[i,j])
        ax[i,j].set(title=temp)
        count+=1
ax[0,0].set(xlabel=None)
ax[0,1].set(xlabel=None)
ax[0,1].set(ylabel=None)
ax[1,1].set(ylabel=None)

fg.tight_layout()
plt.show()
```

![png](confusion_matrices.png)
    


Unsurprisingly, given the dataset was constructed to be fairly clean, the models perform quite well.

### Calibration of Probabilities

Next I would like to calibrate the probabilities. Many of the estimator predictions made with `predict_proba` are uncalibrated. Besides accurate probabilities being nice to have, the method of feature selection based on error contribution below uses the cross entropy or log loss, which requires calibrated probabilities.

First I will set up some utility functions.

```python

def get_scores(y_true, y_pred, y_proba ):
    ''' a function that will quickly compute the brier loss, log loss, and F1 scores for binary classes.
     
    parameters:
        y_true: ground truth labels
        y_pred: predicted labels
        y_proba: predicted probabilities

    returns:
        score: a dictionary that has the elements:
                    'textstr': a text string that can be used in a plot
                    'brier': brier loss score
                    'logloss': log loss score
                    'f1': F1 score
                    '''
    
    # dummy checks
    if len(np.unique(y_true))>2 or len(np.unique(y_pred))>2:
        raise ValueError("must be binary categories")

    if np.any( y_proba<0 ) or np.any( y_proba>1 ):
        raise ValueError("probabilities must be on the interval [0,1]")

    # compute the scores 
    Bscore = brier_score_loss(y_true, y_proba)
    LogLoss = log_loss(y_true, y_proba)
    F1 = f1_score( y_true, y_pred)

    # create a text string
    textstr = '\n'.join((
        f"Brier score loss: {Bscore:0.3f}",
        f"Log loss: {LogLoss:0.3f}",
        f"F1 score: {F1:0.3f}"#,
    ))

    # create the data dictionary to return
    scores = {'textstr':textstr,'brier':Bscore,'logloss':LogLoss,'f1':F1}

    return scores


def plot_reliability_curve(y_true, y_pred, y_proba, ax):
    '''a function to plot the reliability curve or calibration curve.

    parameters:
        y_true: ground truth labels
        y_pred: predicted labels
        y_proba: predicted probabilities
        ax: axis to plot on

    returns nothing
    '''

    # create a line for perfect calibration
    ax.plot([0,1], [0,1], 'b--', label = 'perfect calibration')

    # get the scores
    scores = get_scores(y_true, y_pred, y_proba)

    # create the text box
    props = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5)
    ax.text(0.05, 0.95, scores['textstr'], transform = ax.transAxes, fontsize = 10, verticalalignment = 'top', bbox = props)

    # get the calibration curve and plot it
    fraction_of_positives, mean_predicted_values = calibration_curve(y_true, y_proba, n_bins = 100)
    ax.plot(mean_predicted_values, fraction_of_positives)

def get_predictions( estimator, X_test ):
    '''a function to get the predictions of labels and probabilities from an estimator.

    parameters:
        estimator: an estimator to be used to make predictions
        X_test: a set to make the predictions on

    return:
        y_proba: predicted probabilities
        y_pred: predicted labels
        '''
    y_proba = estimator.predict_proba(X_test)[:,1]
    y_pred = estimator.predict(X_test)
    return y_proba, y_pred

```

Next we will do the calibrations and create a set of plots. The left plots will be uncalibrated predictions, and the right plots will be calibrated ones.

```python
# set up a dictionary that will hold all of the calibrated estimators
calibrated = {}

# create the figure
fg, ax = plt.subplots(4,2,figsize=(10,10))

# loop to create plots and calibrated estimators
for i, estimator in enumerate(estimators.keys()):

    # get predictions for the uncalibrated estimator
    cloned = clone(estimators[estimator]).fit(X_train, y_train)
    y_proba, y_pred = get_predictions(cloned, X_test)

    # plot the reliability curve for the uncalibrated estimator
    plot_reliability_curve( y_test, y_pred, y_proba, ax[i,0] )
    ax[i,0].set_ylabel("Fraction of positives")
    ax[i,0].set_title(estimator+" uncalibrated")
    
    # calibrate the estimator
    calibrated[estimator] = CalibratedClassifierCV( FrozenEstimator(cloned) ).fit(X_val, y_val)

    # get predictions for the calibrated estimator
    y_proba, y_pred = get_predictions(calibrated[estimator], X_test)

    # plot the reliability curve for the calibrated estimator
    plot_reliability_curve( y_test, y_pred, y_proba, ax[i,1] )
    ax[i,1].set_title(estimator+" calibrated")

# set axis labels and plot
ax[3,0].set_xlabel("Mean predicted value")
ax[3,1].set_xlabel("Mean predicted value")

fg.tight_layout()
plt.show()
```

![png](Classification%20Model%20for%20Portfolio%20d0_25_0.png)

Here we see dramatic changes when calibrating the ensemble methods, and minor ones for the logistic regression and neural network. We can also see that the losses are dramatically lower for the neural network than for the other methods. Based on this, we might simply drop the others and go forward with only the neural network, but for the purposes of illustration I would like to continue with all of them.
