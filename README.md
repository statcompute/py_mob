### <p align="center">  Python Implementation of </p>
### <p align="center"> Monotonic Optimal Binning (MOB) </p>

#### Introduction

As an attempt to mimic the mob R package (https://CRAN.R-project.org/package=mob), the py_mob is a collection of python functions that would generate the monotonic binning and perform the WoE (Weight of Evidence) transformation used in consumer credit scorecard developments. Being a piecewise constant transformation in the context of logistic regressions, the WoE has also been employed in other use cases, such as consumer credit loss estimation, prepayment, and even fraud detection models. In addition, Information Value and KS statistic of each independent variables is also calculated to evaluate the variable predictiveness.

#### Installation

```python
pip3 install py_mob
```

#### Example

```python
import sas7bdat, py_mob

df = sas7bdat.SAS7BDAT("accepts.sas7bdat").to_data_frame()

utl = df.rev_util.to_numpy()

bad = df.bad.to_numpy()

utl_bin = py_mob.qtl_bin(utl, bad)

py_mob.view_bin(utl_bin)
#|   bin |   freq |   miss |   bads |   rate |     woe |     iv | rule        |
#|-------+--------+--------+--------+--------+---------+--------+-------------|
#|     1 |   2962 |      0 |    467 | 0.1577 | -0.3198 | 0.047  | $X$ <= 30.0 |
#|     2 |   2875 |      0 |    729 | 0.2536 |  0.2763 | 0.0406 | $X$ > 30.0  |


```
