
---
title: "2. Data Exploration"
---


- [[#Initial Examination of Data]]
- [[#Missing Values]]
- [[#Distribution of Independent Variables]]
- [[#Associations Between Features]]
- [[#The Use of the Mutual Information Coefficient]]

## Initial Examination of Data

First, I will look at the number of rows (which are by design, so I actually already know), and the distribution of the categories.

```python
# get the length of the data frame
nrows = len(data)

# number and proportion for target=1
ntarget1 = data['target'].sum()
ptarget1 = ntarget1/nrows

# number and proportion for target=0
ntarget0 = nrows-ntarget1
ptarget0 = 1-ptarget1

# present results
print(f"The total number of rows is {nrows:,d}. Of those, {ntarget1:,d} are positive, {ntarget0:,d} are negative."+ 
      f"\nThe proportion of positive cases is {ptarget1:.03} and negative cases is {ptarget0:.03}.")

# pie chart
fig,ax = plt.subplots()
ax.pie(x=[ptarget1,ptarget0],labels=['positive','negative'],autopct='%1.1f%%');
plt.show()
```

    The total number of rows is 50,000. Of those, 15,073 are positive, 34,927 are negative.
    The proportion of positive cases is 0.301 and negative cases is 0.699.



    
![png](Classification%20Model%20for%20Portfolio%20d0_5_1.png)
    


## Missing Values

I also want to see how many missing values there are.

```python
# this loop will print the name of any column with a nonzero sum of missing values along with the number of missing values
no_missing_flag = True
for colname in data.columns:
    n_nans = data[colname].isna().sum()
    if n_nans > 0:
        print(f"{colname} has {n_nans} missing values.")
        no_missing_flag = False
if no_missing_flag:
    print("There are no missing values.")
```

    There are no missing values.


Because this is a synthetic dataset, there are no missing values. Were there missing values, I would then need further analysis to determine whether to drop rows or columns with missing data or try some method of imputation (which will be demonstrated in a future addition to this portfolio).

## Distribution of Independent Variables

These data are all numerical, so to get a look at the distributions of feature values I will simply use the `describe` method. On a real dataset I would use domain knowledge to take a closer look at features I think might be relevant.

```python
data.describe()
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
      <th>count</th>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>...</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.009305</td>
      <td>0.385435</td>
      <td>0.402292</td>
      <td>0.067548</td>
      <td>-2.437292</td>
      <td>-0.374888</td>
      <td>4.402445</td>
      <td>0.984402</td>
      <td>-0.407140</td>
      <td>-0.400710</td>
      <td>...</td>
      <td>-0.992325</td>
      <td>1.010425</td>
      <td>-0.416586</td>
      <td>0.388522</td>
      <td>0.380183</td>
      <td>-0.995731</td>
      <td>0.386466</td>
      <td>-0.001563</td>
      <td>1.014605</td>
      <td>0.301460</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.001876</td>
      <td>2.877833</td>
      <td>2.704703</td>
      <td>7.284647</td>
      <td>6.166970</td>
      <td>2.724958</td>
      <td>7.715400</td>
      <td>2.475133</td>
      <td>2.543759</td>
      <td>2.350943</td>
      <td>...</td>
      <td>2.801943</td>
      <td>2.810665</td>
      <td>2.779655</td>
      <td>2.893455</td>
      <td>2.792027</td>
      <td>2.425633</td>
      <td>2.725618</td>
      <td>0.995862</td>
      <td>2.523641</td>
      <td>0.458897</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-4.309958</td>
      <td>-12.576973</td>
      <td>-10.434212</td>
      <td>-27.566633</td>
      <td>-26.646586</td>
      <td>-10.982415</td>
      <td>-25.189695</td>
      <td>-8.883930</td>
      <td>-13.818169</td>
      <td>-9.206762</td>
      <td>...</td>
      <td>-12.265536</td>
      <td>-10.008664</td>
      <td>-12.440082</td>
      <td>-12.672520</td>
      <td>-11.012225</td>
      <td>-14.302898</td>
      <td>-10.511940</td>
      <td>-4.034314</td>
      <td>-9.444280</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.685982</td>
      <td>-1.538148</td>
      <td>-1.450334</td>
      <td>-4.871029</td>
      <td>-6.573651</td>
      <td>-2.206324</td>
      <td>-0.901286</td>
      <td>-0.677715</td>
      <td>-2.133717</td>
      <td>-2.012085</td>
      <td>...</td>
      <td>-2.876197</td>
      <td>-0.866984</td>
      <td>-2.296212</td>
      <td>-1.581324</td>
      <td>-1.515726</td>
      <td>-2.625107</td>
      <td>-1.454356</td>
      <td>-0.673433</td>
      <td>-0.680536</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.010349</td>
      <td>0.395964</td>
      <td>0.360203</td>
      <td>-0.296250</td>
      <td>-2.434660</td>
      <td>-0.377266</td>
      <td>4.297483</td>
      <td>0.991522</td>
      <td>-0.391489</td>
      <td>-0.488600</td>
      <td>...</td>
      <td>-0.991356</td>
      <td>1.002126</td>
      <td>-0.406314</td>
      <td>0.390705</td>
      <td>0.378676</td>
      <td>-0.985660</td>
      <td>0.403399</td>
      <td>0.000029</td>
      <td>1.016488</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.671657</td>
      <td>2.341213</td>
      <td>2.220133</td>
      <td>4.671136</td>
      <td>1.714638</td>
      <td>1.451967</td>
      <td>9.611936</td>
      <td>2.639159</td>
      <td>1.313721</td>
      <td>1.122236</td>
      <td>...</td>
      <td>0.878624</td>
      <td>2.898819</td>
      <td>1.486461</td>
      <td>2.356106</td>
      <td>2.277618</td>
      <td>0.622938</td>
      <td>2.236307</td>
      <td>0.669239</td>
      <td>2.713905</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.014015</td>
      <td>11.043349</td>
      <td>12.134694</td>
      <td>34.839229</td>
      <td>23.487909</td>
      <td>11.338587</td>
      <td>40.424541</td>
      <td>11.305053</td>
      <td>10.613916</td>
      <td>10.470961</td>
      <td>...</td>
      <td>10.353978</td>
      <td>12.980623</td>
      <td>11.112211</td>
      <td>13.285860</td>
      <td>12.423742</td>
      <td>10.259914</td>
      <td>12.033986</td>
      <td>4.152785</td>
      <td>11.962137</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>

## Associations Between Features

Next I want to take a quick look at associations. While this might commonly be done with Pearson's correlation coefficient, I prefer to use the Spearman. Pearson's assumes a linear relationship, and thus can fail to reflect robust nonlinear associations. That is, there might be a very good nonlinear association, but Pearson's coefficient would show a weaker one. By using rank, Spearman only considers the mutual monotonicity of the variables.

I will begin by looking only at the features.


```python
# set up the figures
fg, ax = plt.subplots(1,1,figsize=(10,10))

# create a correlation matrix
corr_matrix = data.iloc[:,:-1].corr(method='spearman')

# create a heatmap
ax = sns.heatmap(np.abs(corr_matrix))

plt.show()
```


    
![png](Classification%20Model%20for%20Portfolio%20d0_11_0.png)
    


We see some higher correlations between features (e.g., X13 and X16), as well as completely uncorrelated features (which was by design). One option we could now use is to try linear combinations of those highly correlated features. However, this should be handled by our feature selection methods below.

## The Use of the Mutual Information Coefficient

I would like more information than Spearman can provide for measuring the association between the features and the target variable. For that, I like the mutual information coefficient. That particular measure I found from [An Undeservedly Forgotten Correlation Coefficient](https://towardsdatascience.com/an-undeservedly-forgotten-correlation-coefficient-86245ccb774c/) on *Towards Data Science*. In short, the mutual information is the relative entropy between the joint distributions of two variables and the product of their marginal distributions. That is, for probability $p$ defined over variables $X$ and $Y$ with values $x$ and $y$ respectively with joint probability $p(x,y)$ and marginal probabilities $p(x)$ and $p(y)$:
$$
I(X,Y) = \int_X \int_Y p(x,y) \text{log}\frac{p(x,y)}{p(x)p(y)}.
$$

This is scaled to be between on the interval $[0,1]$ through:
$$
R(X,Y) = [1 - \text{exp}(-2*I(X,Y))]^\frac{1}{2}.
$$
This is the *mutual information coefficient* (MIC), and it is significantly more informative than either Pearson or Spearman. We can regard it as telling us how much information we get about one variable when we know the other variable. An example of its utility in the above article uses a plot that is a ring in the plane. There clearly is a strong relationship between the $x$ and $y$ variables, but traditional correlation coefficients show virtually no relationship. The MIC, on the other hand, reflects the actual strong association. Note it does take some time to compute, so it is not efficient to use on a large data set. I will use it to look at relationships between the features and the target variable, to get an initial idea of the effect size from each individual feature.


```python
# compute the mutual information coefficient
MI = mutual_info_regression(data.iloc[:,:-1],data['target'])
R = np.sqrt(1-np.exp(-2*MI))

# create a quick dataframe fro ease of plotting
df_R = pd.DataFrame({'feature':data.columns[:-1],'MIC':R}).sort_values('MIC')

# make a plot
fg, ax = plt.subplots(1,1,figsize=(10,10));

ax.barh(y = df_R['feature'], width = df_R['MIC']);
ax.set_title("Mutual Information Coefficients of Features and Target");
ax.set_xlabel("Mutual Information Coefficient");
ax.set_ylabel("Feature");

```


    
![png](Classification%20Model%20for%20Portfolio%20d0_13_0.png)
    


We see a number of features that are completely uncorrelated, so we can safely drop those.


```python
data.drop(columns = df_R.loc[df_R['MIC']==0,'feature'], inplace = True)
```

