<p align="center">
  <img width="100" height="100" src="py_mob/py_mob1.jpg">
</p>

### <p align="center">  Python Implementation of </p>
### <p align="center"> Monotonic Optimal Binning (PY_MOB) </p>

#### Introduction

As an attempt to mimic the mob R package (https://CRAN.R-project.org/package=mob), the py_mob is a collection of python functions that would generate the monotonic binning and perform the WoE (Weight of Evidence) transformation used in consumer credit scorecard developments. The woe transformation is a piecewise transformation that is linear to the log odds. For a numeric variable, all of its monotonic functional transformations will converge to the same woe transformation. In addition, Information Value and KS statistic of each independent variables is also calculated to evaluate the variable predictiveness.

Different from other python packages for the same purpose, the py\_mob package is very lightweight and the underlying computation is driven by the built-in python list or the numpy array. Functions would return lists of dictionaries, which can be easily converted to other data structures, such as pandas.DataFrame or astropy.table. 

What's more, six different monotonic binning algorithms are implemented, namely qtl\_bin(), bad\_bin(), iso\_bin(), rng\_bin(), kmn\_bin(), and gbm\_bin(), that would provide different predictability and cardinality. 

People without the background knowledge in the consumer risk modeling might be wondering why the monotonic binning and thereafter the WoE transformation are important. Below are a couple reasons based on my experience. They are perfectly generalizable in other use cases of logistic regression with binary outcomes. 
1. Because the WoE is a piecewise transformation based on the data discretization, all missing values would fall into a standalone category either by itself or to be combined with the neighbor with a similar bad rate. As a result, the special treatment for missing values is not necessary.
2. After the monotonic binning of each variable, since the WoE value for each bin is a projection from the predictor into the response that is defined by the log ratio between event and non-event distributions, any raw value of the predictor doesn't matter anymore and therefore the issue related to outliers would disappear.
3. While many modelers would like to use log or power transformations to achieve a good linear relationship between the predictor and log odds of the response, which is heuristic at best with no guarantee for the good outcome, the WoE transformation is strictly linear with respect to log odds of the response with the unity correlation. It is also worth mentioning that a numeric variable and its strictly monotone functions should converge to the same monotonic WoE transformation.
4. At last, because the WoE is defined as the log ratio between event and non-event distributions, it is indicative of the separation between cases with Y = 0 and cases with Y = 1. As the weighted sum of WoE values with the weight being the difference in event and non-event distributions, the IV (Information Value) is an important statistic commonly used to measure the predictor importance.


#### Package Dependencies

```text
pandas, numpy, scipy, sklearn, lightgbm, tabulate
```

#### Installation

```python
pip3 install py_mob
```

#### Core Functions

```
py_mob
  |-- qtl_bin()  : An iterative discretization based on quantiles of X.
  |-- bad_bin()  : A revised iterative discretization for records with Y = 1.
  |-- iso_bin()  : A discretization algorthm driven by the isotonic regression between X and Y.
  |-- rng_bin()  : A revised iterative discretization based on the equal-width range of X.
  |-- kmn_bin()  : A discretization algorthm based on the kmean clustering of X.
  |-- gbm_bin()  : A discretization algorthm based on the gradient boosting machine.
  |-- summ_bin() : Generates the statistical summary for the binning outcome.
  |-- view_bin() : Displays the binning outcome in a tabular form.
  |-- cal_woe()  : Applies the WoE transformation to a numeric vector based on the binning outcome.
  |-- pd_bin()   : Discretizes each vector in a pandas DataFrame.
  `-- pd_woe()   : Applies WoE transformaton to each vector in the pandas DataFrame.
```
