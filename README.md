### <p align="center">  Python Implementation of </p>
### <p align="center"> Monotonic Optimal Binning (MOB) </p>

#### Introduction

As an attempt to mimic the mob R package (https://CRAN.R-project.org/package=mob), the py_mob is a collection of python functions that would generate the monotonic binning and perform the WoE (Weight of Evidence) transformation used in consumer credit scorecard developments. In addition, Information Value and KS statistic of each independent variables is also calculated to evaluate the variable predictiveness.

Different from other python packages for the same purpose, the py_mob package is very lightweight and the underlying computation is driven by the built-in python list or the numpy array. Functions would return lists of dictionaries, which can be easily converted to other data structures, such as pandas.DataFrame or astropy.table. 

Currently, three different monotonic binning algorithms are implemented, namely qtl_bin(), bad_bin(), and iso_bin(). For details, please refer to https://github.com/statcompute/mob. 

#### Package Dependencies

```text
numpy, scipy, sklearn, tabulate
```

#### Installation

```python
pip3 install py_mob
```

#### Functions

```
py_mob
   |-- bad_bin.py
   |-- iso_bin.py
   |-- manual_bin.py
   |-- miss_bin.py
   |-- qcut.py
   |-- qtl_bin.py
   |-- summ_bin.py
   `-- view_bin.py
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
#{'bin': 1, 'freq': 2962, 'miss': 0, 'bads': 467.0, 'rate': 0.1577, 'woe': -0.3198, 'iv': 0.047, 'rule': '$X$ <= 30.0'}
#{'bin': 2, 'freq': 2875, 'miss': 0, 'bads': 729.0, 'rate': 0.2536, 'woe': 0.2763, 'iv': 0.0406, 'rule': '$X$ > 30.0'}

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
#|-------+--------+--------+--------+--------+---------+--------+--------------------------------|
#|     1 |   3250 |      0 |    510 | 0.1569 | -0.3254 | 0.0533 | $X$ <= 36.0                    |
#|     2 |    182 |      0 |     32 | 0.1758 | -0.189  | 0.0011 | ($X$ > 36.0) and ($X$ <= 40.0) |
#|     3 |    669 |      0 |    137 | 0.2048 | -0.0007 | 0      | ($X$ > 40.0) and ($X$ <= 58.0) |
#|     4 |     77 |      0 |     16 | 0.2078 |  0.0177 | 0      | ($X$ > 58.0) and ($X$ <= 60.0) |
#|     5 |    408 |      0 |     95 | 0.2328 |  0.1636 | 0.002  | ($X$ > 60.0) and ($X$ <= 72.0) |
#|     6 |     96 |      0 |     24 | 0.25   |  0.2573 | 0.0012 | ($X$ > 72.0) and ($X$ <= 75.0) |
#|     7 |    246 |      0 |     70 | 0.2846 |  0.434  | 0.0089 | ($X$ > 75.0) and ($X$ <= 83.0) |
#|     8 |    376 |      0 |    116 | 0.3085 |  0.5489 | 0.0225 | ($X$ > 83.0) and ($X$ <= 96.0) |
#|     9 |     50 |      0 |     17 | 0.34   |  0.6927 | 0.0049 | ($X$ > 96.0) and ($X$ <= 98.0) |
#|    10 |    483 |      0 |    179 | 0.3706 |  0.8263 | 0.0695 | $X$ > 98.0                     |

py_mob.iso_bin(utl, bad)['cut']
#[36.0, 40.0, 58.0, 60.0, 72.0, 75.0, 83.0, 96.0, 98.0]
```
