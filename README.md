<p align="center">
  <img width="100" height="100" src="py_mob/py_mob1.jpg">
</p>

### <p align="center">  Python Implementation of </p>
### <p align="center"> Monotonic Optimal Binning (PY_MOB) </p>

#### Introduction

As an attempt to mimic the mob R package (https://CRAN.R-project.org/package=mob), the py_mob is a collection of python functions that would generate the monotonic binning and perform the WoE (Weight of Evidence) transformation used in consumer credit scorecard developments. The woe transformation is a piecewise transformation that is linear to the log odds. For a numeric variable, all of its monotonic functional transformations will converge to the same woe transformation. In addition, Information Value and KS statistic of each independent variables is also calculated to evaluate the variable predictiveness.

Different from other python packages for the same purpose, the py_mob package is very lightweight and the underlying computation is driven by the built-in python list or the numpy array. Functions would return lists of dictionaries, which can be easily converted to other data structures, such as pandas.DataFrame or astropy.table. 

Currently, four different monotonic binning algorithms are implemented, namely qtl_bin(), bad_bin(), iso_bin(), and rng_bin(). For details, please see core functions below. 
https://github.com/statcompute/py\_mob
#### Package Dependencies

```text
numpy, scipy, sklearn, tabulate
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
   |-- summ_bin() : Generates the statistical summary for the binning outcome.
   |-- view_bin() : Displays the binning outcome in a tabular form.
   `-- cal_woe()  : Applies the WoE transformation to a numeric vector based on the binning outcome.
```

#### Example

```python
import sas7bdat, py_mob

df = sas7bdat.SAS7BDAT("accepts.sas7bdat").to_data_frame()

utl = df.rev_util.to_numpy()

bad = df.bad.to_numpy()

utl_bin = py_mob.qtl_bin(utl, bad)

for key in utl_bin:
  print(key + ":")
  for lst in utl_bin[key]:
    print(lst)
#cut:
#30.0
#tbl:
#{'bin': 1, 'freq': 2962, 'miss': 0, 'bads': 467.0, 'rate': 0.1577, 'woe': -0.3198, 'iv': 0.047, 
# 'rule': '$X$ <= 30.0'}
#{'bin': 2, 'freq': 2875, 'miss': 0, 'bads': 729.0, 'rate': 0.2536, 'woe': 0.2763, 'iv': 0.0406, 
# 'rule': '$X$ > 30.0'}

py_mob.view_bin(utl_bin)
#|   bin |   freq |   miss |   bads |   rate |     woe |     iv | rule        |
#|-------+--------+--------+--------+--------+---------+--------+-------------|
#|     1 |   2962 |      0 |    467 | 0.1577 | -0.3198 | 0.047  | $X$ <= 30.0 |
#|     2 |   2875 |      0 |    729 | 0.2536 |  0.2763 | 0.0406 | $X$ > 30.0  |

py_mob.summ_bin(utl_bin)
#{'bad rate': 0.2049, 'iv': 0.0876, 'ks': 14.71}

py_mob.view_bin(py_mob.bad_bin(utl, bad))
#|   bin |   freq |   miss |   bads |   rate |     woe |     iv | rule                           |
#|-------+--------+--------+--------+--------+---------+--------+--------------------------------|
#|     1 |   2495 |      0 |    399 | 0.1599 | -0.3029 | 0.0357 | $X$ <= 21.0                    |
#|     2 |   2125 |      0 |    399 | 0.1878 | -0.1087 | 0.0042 | ($X$ > 21.0) and ($X$ <= 73.0) |
#|     3 |   1217 |      0 |    398 | 0.327  |  0.6343 | 0.0991 | $X$ > 73.0                     |

py_mob.view_bin(py_mob.iso_bin(utl, bad))
#|   bin |   freq |   miss |   bads |   rate |     woe |     iv | rule                           |
#|-------|--------|--------|--------|--------|---------|--------|--------------------------------|
#|     1 |   3250 |      0 |    510 | 0.1569 | -0.3254 | 0.0533 | $X$ <= 36.0                    |
#|     2 |    182 |      0 |     32 | 0.1758 | -0.1890 | 0.0011 | ($X$ > 36.0) and ($X$ <= 40.0) |
#|     3 |    669 |      0 |    137 | 0.2048 | -0.0007 | 0.0000 | ($X$ > 40.0) and ($X$ <= 58.0) |
#|     4 |     77 |      0 |     16 | 0.2078 |  0.0177 | 0.0000 | ($X$ > 58.0) and ($X$ <= 60.0) |
#|     5 |    408 |      0 |     95 | 0.2328 |  0.1636 | 0.0020 | ($X$ > 60.0) and ($X$ <= 72.0) |
#|     6 |     34 |      0 |      8 | 0.2353 |  0.1773 | 0.0002 | ($X$ > 72.0) and ($X$ <= 73.0) |
#|     7 |     62 |      0 |     16 | 0.2581 |  0.2999 | 0.0010 | ($X$ > 73.0) and ($X$ <= 75.0) |
#|     8 |    246 |      0 |     70 | 0.2846 |  0.4340 | 0.0089 | ($X$ > 75.0) and ($X$ <= 83.0) |
#|     9 |    376 |      0 |    116 | 0.3085 |  0.5489 | 0.0225 | ($X$ > 83.0) and ($X$ <= 96.0) |
#|    10 |     50 |      0 |     17 | 0.3400 |  0.6927 | 0.0049 | ($X$ > 96.0) and ($X$ <= 98.0) |
#|    11 |    483 |      0 |    179 | 0.3706 |  0.8263 | 0.0695 | $X$ > 98.0                     |

py_mob.view_bin(py_mob.rng_bin(utl, bad))
#|   bin |   freq |   miss |   bads |   rate |     woe |     iv | rule                           |
#|-------|--------|--------|--------|--------|---------|--------|--------------------------------|
#|     1 |   2703 |      0 |    423 | 0.1565 | -0.3286 | 0.0452 | $X$ <= 25.0                    |
#|     2 |   1111 |      0 |    200 | 0.1800 | -0.1603 | 0.0047 | ($X$ > 25.0) and ($X$ <= 50.0) |
#|     3 |    868 |      0 |    191 | 0.2200 |  0.0905 | 0.0013 | ($X$ > 50.0) and ($X$ <= 75.0) |
#|     4 |   1155 |      0 |    382 | 0.3307 |  0.6511 | 0.0995 | $X$ > 75.0                     |

py_mob.iso_bin(utl, bad)['cut']
#[36.0, 40.0, 58.0, 60.0, 72.0, 75.0, 83.0, 96.0, 98.0]

for x in py_mob.cal_woe(utl[:3], py_mob.iso_bin(utl, bad)):
  print(x)
#{'x':  0.0, 'bin': 1, 'woe': -0.3254}
#{'x':  2.0, 'bin': 1, 'woe': -0.3254}
#{'x': 21.0, 'bin': 1, 'woe': -0.3254}
```
