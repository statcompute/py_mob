# py_mob/py_mob.py

import numpy, scipy.stats, sklearn.isotonic, tabulate


###################################################################################################


def cal_woe(x, bin):
  """
  The function applies the woe transformation to a numeric vector based on the binning outcome.

  Parameters:
    x   : A numeric vector, which can be a list, 1-D numpy array, or pandas series
    bin : An object containing the binning outcome.

  Returns:
    A list of dictionaries with three keys

  Example:
    ltv_bin = qtl_bin(ltv, bad)

    for x in cal_woe(ltv[:3], ltv_bin):
      print(x)

    # {'x': 109.0, 'bin': 6, 'woe': 0.2694}
    # {'x':  97.0, 'bin': 3, 'woe': 0.0045}
    # {'x': 105.0, 'bin': 5, 'woe': 0.1829}
  """

  _cut = sorted(bin['cut'] + [numpy.PINF, numpy.NINF])
  _dat = [[_1[0], _1[1], _2] for _1, _2 in zip(enumerate(x), ~numpy.isnan(x))]

  _m1 = [_[:2] for _ in _dat if _[2] == 0]
  _l1 = [_[:2] for _ in _dat if _[2] == 1]

  _l2 = [[*_1, _2] for _1, _2 in zip(_l1, numpy.searchsorted(_cut, [_[1] for _ in _l1]).tolist())]

  flatten = lambda l: [item for subl in l for item in subl]
 
  _l3 = flatten([[[*l, b['woe']] for l in _l2 if l[2] == b['bin']] for b in bin['tbl'] if b['bin'] > 0])

  if len(_m1) > 0:
    if len([_ for _ in bin['tbl'] if _['miss'] > 0]) > 0:
      _m2 = [l + [_['bin'] for _ in bin['tbl'] if _['miss'] > 0] 
               + [_['woe'] for _ in bin['tbl'] if _['miss'] > 0] for l in _m1]
    else:
      _m2 = [l + [0, 0] for l in _m1]
    _l3.extend(_m2)

  _key = ["x", "bin", "woe"]

  return(list(dict(zip(_key, _[1:])) for _ in sorted(_l3, key = lambda x: x[0])))


###################################################################################################


def summ_bin(x):
  """
  The function summarizes the binning outcome generated from a binning function, e.g. qtl_bin() or
  iso_bin().

  Parameters:
    x: An object containing the binning outcome.

  Returns:
    A dictionary with statistics derived from the binning outcome

  Example:
    summ_bin(iso_bin(ltv, bad))

    # {'bad rate': 0.2049, 'iv': 0.18, 'ks': 16.88}
  """
 
  _r1 = round(sum([_['bads'] for _ in x['tbl']]) / sum([_['freq'] for _ in x['tbl']]), 4)
  _r2 = round(sum([_['iv'] for _ in x['tbl']]), 4)

  _freq = sum([_['freq'] for _ in x['tbl']])
  _bads = sum([_['bads'] for _ in x['tbl']])

  cumsum = lambda x: [sum([_ for _ in x][0:(i+1)]) for i in range(len(x))]

  _cumb = cumsum([_['bads'] / _bads for _ in x['tbl']])
  _cumg = cumsum([(_['freq'] - _['bads']) / (_freq - _bads) for _ in x['tbl']])
  _r3 = round(max([numpy.abs(_[0] - _[1]) for _ in zip(_cumb, _cumg)]) * 100, 2)

  return({"bad rate": _r1, "iv": _r2, "ks": _r3})


###################################################################################################


def view_bin(x):
  """
  The function shows the binning outcome generated from a binning function, e.g. qtl_bin() or iso_bin().

  Parameters:
    x: An object containing the binning outcome.

  Returns:
    None

  Example:
    view_bin(iso_bin(ltv, bad))

    |   bin |   freq |   miss |   bads |   rate |     woe |     iv | rule                         |
    |-------|--------|--------|--------|--------|---------|--------|------------------------------|
    |     0 |    213 |    213 |     70 | 0.3286 |  0.6416 | 0.0178 | numpy.isnan($X$)             |
    |     1 |   2850 |      0 |    367 | 0.1288 | -0.5559 | 0.1268 | $X$ <= 0.0                   |
    |     2 |    891 |      0 |    193 | 0.2166 |  0.0704 | 0.0008 | ($X$ > 0.0) and ($X$ <= 1.0) |
    |     3 |    810 |      0 |    207 | 0.2556 |  0.2867 | 0.0124 | ($X$ > 1.0) and ($X$ <= 3.0) |
    |     4 |   1073 |      0 |    359 | 0.3346 |  0.6684 | 0.0978 | $X$ > 3.0                    |

  """
  
  print(tabulate.tabulate(x['tbl'], headers = "keys", tablefmt = "github",
                        floatfmt = (".0f", ".0f", ".0f", ".0f", ".4f", ".4f", ".4f")))


###################################################################################################


def qcut(x, n):
  """
  The function discretizes a numeric vector into n pieces based on quantiles.

  Parameters:
    x: A numeric vector.
    n: An integer indicating the number of categories to discretize.

  Returns:
    A list of numeric values to divide the vector x into n categories.

  Example:
    qcut(range(10), 3)

    # [3, 6]
  """

  _q = numpy.linspace(0, 100, n, endpoint = False)[1:]
  _x = [_ for _ in x if not numpy.isnan(_)]
  _c = numpy.unique(numpy.percentile(_x, _q, interpolation = "lower"))
  return([_ for _ in _c])


###################################################################################################


def manual_bin(x, y, cuts):
  """
  The function discretizes the x vector and then summarizes over the y vector 
  based on the discretization result.

  Parameters:
    x    : A numeric vector to discretize without missing values, e.g. numpy.nan or math.nan
    y    : A numeric vector with binary values of 0/1 and with the same length of x 
    cuts : A list of numeric values as cut points to discretize x. 

  Returns:
    A list of dictionaries for the binning outcome with following keys:
      "bin"  : the binning group
      "freq" : the number of records in each binning group
      "miss" : the placeholder for records with missing values of x
      "bads" : the number of records for y = 1 in each binning group
      "minx" : the minimum value of x for each binning group
      "maxx" : the maximum value of x for each binning group

  Example:
    for x in manual_bin(scr, bad, [650, 700, 750]):
      print(x)

    # {'bin': 1, 'freq': 1311, 'miss': 0, 'bads': 520.0, 'minx': 443.0, 'maxx': 650.0}
    # {'bin': 2, 'freq': 1688, 'miss': 0, 'bads': 372.0, 'minx': 651.0, 'maxx': 700.0}
    # {'bin': 3, 'freq': 1507, 'miss': 0, 'bads': 157.0, 'minx': 701.0, 'maxx': 750.0}
    # {'bin': 4, 'freq': 1016, 'miss': 0, 'bads':  42.0, 'minx': 751.0, 'maxx': 848.0}
  """

  _x = [_ for _ in x]
  _y = [_ for _ in y]
  _c = sorted([_ for _ in set(cuts)] + [numpy.NINF, numpy.PINF])
  _g = numpy.searchsorted(_c, _x).tolist()

  _l1 = list(dict(zip(['g', 'x', 'y'], _)) for _ in zip(_g, _x, _y))
  _l2 = zip(set(_g), [[l for l in _l1 if l["g"] == g] for g in set(_g)])

  return(sorted([dict(zip(["bin", "freq", "miss", "bads", "minx", "maxx"],
                          [_1, 
                           len(_2), 
                           0,
                           sum([_["y"] for _ in _2]),
                           min([_["x"] for _ in _2]),
                           max([_["x"] for _ in _2])])) for _1, _2 in _l2], 
                key = lambda x: x["bin"]))


###################################################################################################


def miss_bin(y):
  """
  The function summarizes the y vector with binary values of 0/1 and is not supposed to be called
  directly by users.

  Parameters:
    y : A numeric vector with binary values of 0/1.

  Returns:
    A dictionary with following keys:
      "bin"  : 0 as default 
      "freq" : the number of records 
      "miss" : same value as "freq"
      "bads" : the number of records with y = 1
      "minx" : nan as default
      "maxx" : nan as default
  """

  return({"bin": 0, "freq": len([_ for _ in y]), "miss": len([_ for _ in y]), 
          "bads": sum([_ for _ in y]), "minx": numpy.nan, "maxx": numpy.nan})


###################################################################################################


def qtl_bin(x, y):
  """
  The function discretizes the x vector based on percentiles and then summarizes over the y vector
  to derive the weight of evidence transformaton (WoE) and information values.

  Parameters:
    x : A numeric vector to discretize. It is a list, 1-D numpy array, or pandas series.
    y : A numeric vector with binary values of 0/1 and with the same length of x.

  Returns:
    A dictionary with two keys:
      "cut" : A numeric vector with cut points applied to the x vector.
      "tbl" : A list of dictionaries summarizing the binning outcome. 
              It is a list, 1-D numpy array, or pandas series.

  Example:
    qtl_bin(derog, bad)["cut"]
    #  [0.0, 1.0, 3.0]

    view_bin(qtl_bin(derog, bad))

    |   bin |   freq |   miss |   bads |   rate |     woe |     iv | rule                         |
    |-------|--------|--------|--------|--------|---------|--------|------------------------------|
    |     0 |    213 |    213 |     70 | 0.3286 |  0.6416 | 0.0178 | numpy.isnan($X$)             |
    |     1 |   2850 |      0 |    367 | 0.1288 | -0.5559 | 0.1268 | $X$ <= 0.0                   |
    |     2 |    891 |      0 |    193 | 0.2166 |  0.0704 | 0.0008 | ($X$ > 0.0) and ($X$ <= 1.0) |
    |     3 |    810 |      0 |    207 | 0.2556 |  0.2867 | 0.0124 | ($X$ > 1.0) and ($X$ <= 3.0) |
    |     4 |   1073 |      0 |    359 | 0.3346 |  0.6684 | 0.0978 | $X$ > 3.0                    |
  """

  _data = [_ for _ in zip(x, y, ~numpy.isnan(x))]
  _freq = len(_data)
  _bads = sum([_[1] for _ in _data])

  _x = [_[0] for _ in _data if _[2] == 1]
  _y = [_[1] for _ in _data if _[2] == 1]

  _n = numpy.arange(2, max(3, min(50, len(numpy.unique(_x)) - 1)))
  _p = numpy.unique([qcut(_x, _) for _ in _n])

  _l1 = [[_, manual_bin(_x, _y, _)] for _ in _p]

  _l2 = [[l[0], 
          min([_["bads"] / _["freq"] for _ in l[1]]), 
          max([_["bads"] / _["freq"] for _ in l[1]]),
          scipy.stats.spearmanr([_["bin"] for _ in l[1]], [_["bads"] / _["freq"] for _ in l[1]])[0]
         ] for l in _l1]

  _l3 = [l[0] for l in sorted(_l2, key = lambda x: -len(x[0])) 
         if numpy.abs(round(l[3], 8)) == 1 and round(l[1], 8) > 0 and round(l[2], 8) < 1][0]

  _l4 = sorted(*[l[1] for l in _l1 if l[0] == _l3], key = lambda x: x["bads"] / x["freq"])

  if len([_ for _ in _data if _[2] == 0]) > 0:
    _m1 = miss_bin([_[1] for _ in _data if _[2] == 0])
    if _m1["bads"] == 0: 
      for _ in ['freq', 'miss', 'bads']:
        _l4[0][_]  = _l4[0][_]  + _m1[_] 
    elif _m1["freq"] == _m1["bads"]: 
      for _ in ['freq', 'miss', 'bads']:
        _l4[-1][_]  = _l4[-1][_]  + _m1[_]
    else:
      _l4.append(_m1)

  _l5 = sorted([{**_, 
                 "rate": round(_["bads"] / _["freq"], 4),
                 "woe" : round(numpy.log((_["bads"] / _bads) / ((_["freq"] - _["bads"]) / (_freq - _bads))), 4),
                 "iv"  : round((_["bads"] / _bads - (_["freq"] - _["bads"]) / (_freq - _bads)) * 
                               numpy.log((_["bads"] / _bads) / ((_["freq"] - _["bads"]) / (_freq - _bads))), 4)
                } for _ in _l4], key = lambda x: x["bin"])

  for _ in _l5:
    if _["bin"] == 0:
      _["rule"] = "numpy.isnan($X$)"
    elif _["bin"] == len(_l3) + 1:
      if _["miss"] == 0:
        _["rule"] = "$X$ > " + str(_l3[-1])
      else:
        _["rule"] = "($X$ > " + str(_l3[-1]) + ") or numpy.isnan($X$)"
    elif _["bin"] == 1:
      if _["miss"] == 0:
        _["rule"] = "$X$ <= " + str(_l3[0])
      else:
        _["rule"] = "($X$ <= " + str(_l3[0]) + ") or numpy.isnan($X$)"
    else:
        _["rule"] = "($X$ > " + str(_l3[_["bin"] - 2]) + ") and ($X$ <= " + str(_l3[_["bin"] - 1]) + ")"

  _sel = ["bin", "freq", "miss", "bads", "rate", "woe", "iv", "rule"]

  return({"cut": _l3, "tbl": [{k: _[k] for k in _sel} for _ in _l5]})
  

##################################################################################################


def bad_bin(x, y):
  """
  The function discretizes the x vector based on percentiles and then summarizes over the y vector
  with y = 1 to derive the weight of evidence transformaton (WoE) and information values.

  Parameters:
    x : A numeric vector to discretize. It is a list, 1-D numpy array, or pandas series.
    y : A numeric vector with binary values of 0/1 and with the same length of x.
        It is a list, 1-D numpy array, or pandas series.

  Returns:
    A dictionary with two keys:
      "cut" : A numeric vector with cut points applied to the x vector.
      "tbl" : A list of dictionaries summarizing the binning outcome.

  Example:
    bad_bin(derog, bad)["cut"]
    # [0.0, 2.0, 4.0]

    view_bin(bad_bin(derog, bad))

    |   bin |   freq |   miss |   bads |   rate |     woe |     iv | rule                         |
    |-------|--------|--------|--------|--------|---------|--------|------------------------------|
    |     0 |    213 |    213 |     70 | 0.3286 |  0.6416 | 0.0178 | numpy.isnan($X$)             |
    |     1 |   2850 |      0 |    367 | 0.1288 | -0.5559 | 0.1268 | $X$ <= 0.0                   |
    |     2 |   1369 |      0 |    314 | 0.2294 |  0.1440 | 0.0051 | ($X$ > 0.0) and ($X$ <= 2.0) |
    |     3 |    587 |      0 |    176 | 0.2998 |  0.5078 | 0.0298 | ($X$ > 2.0) and ($X$ <= 4.0) |
    |     4 |    818 |      0 |    269 | 0.3289 |  0.6426 | 0.0685 | $X$ > 4.0                    |
  """

  _data = [_ for _ in zip(x, y, ~numpy.isnan(x))]
  _freq = len(_data)
  _bads = sum([_[1] for _ in _data])

  _x = [_[0] for _ in _data if _[2] == 1]
  _y = [_[1] for _ in _data if _[2] == 1]

  _n = numpy.arange(2, max(3, min(50, len(numpy.unique([_[0] for _ in _data if _[1] == 1 and _[2] == 1])) - 1)))
  _p = numpy.unique([qcut([_[0] for _ in _data if _[1] == 1 and _[2] == 1], _) for _ in _n])

  _l1 = [[_, manual_bin(_x, _y, _)] for _ in _p]

  _l2 = [[l[0], 
          min([_["bads"] / _["freq"] for _ in l[1]]), 
          max([_["bads"] / _["freq"] for _ in l[1]]),
          scipy.stats.spearmanr([_["bin"] for _ in l[1]], [_["bads"] / _["freq"] for _ in l[1]])[0]
         ] for l in _l1]

  _l3 = [l[0] for l in sorted(_l2, key = lambda x: -len(x[0])) 
         if numpy.abs(round(l[3], 8)) == 1 and round(l[1], 8) > 0 and round(l[2], 8) < 1][0]

  _l4 = sorted(*[l[1] for l in _l1 if l[0] == _l3], key = lambda x: x["bads"] / x["freq"])

  if len([_ for _ in _data if _[2] == 0]) > 0:
    _m1 = miss_bin([_[1] for _ in _data if _[2] == 0])
    if _m1["bads"] == 0: 
      for _ in ['freq', 'miss', 'bads']:
        _l4[0][_]  = _l4[0][_]  + _m1[_] 
    elif _m1["freq"] == _m1["bads"]: 
      for _ in ['freq', 'miss', 'bads']:
        _l4[-1][_]  = _l4[-1][_]  + _m1[_]
    else:
      _l4.append(_m1)

  _l5 = sorted([{**_, 
                 "rate": round(_["bads"] / _["freq"], 4),
                 "woe" : round(numpy.log((_["bads"] / _bads) / ((_["freq"] - _["bads"]) / (_freq - _bads))), 4),
                 "iv"  : round((_["bads"] / _bads - (_["freq"] - _["bads"]) / (_freq - _bads)) * 
                               numpy.log((_["bads"] / _bads) / ((_["freq"] - _["bads"]) / (_freq - _bads))), 4)
                } for _ in _l4], key = lambda x: x["bin"])

  for _ in _l5:
    if _["bin"] == 0:
      _["rule"] = "numpy.isnan($X$)"
    elif _["bin"] == len(_l3) + 1:
      if _["miss"] == 0:
        _["rule"] = "$X$ > " + str(_l3[-1])
      else:
        _["rule"] = "($X$ > " + str(_l3[-1]) + ") or numpy.isnan($X$)"
    elif _["bin"] == 1:
      if _["miss"] == 0:
        _["rule"] = "$X$ <= " + str(_l3[0])
      else:
        _["rule"] = "($X$ <= " + str(_l3[0]) + ") or numpy.isnan($X$)"
    else:
        _["rule"] = "($X$ > " + str(_l3[_["bin"] - 2]) + ") and ($X$ <= " + str(_l3[_["bin"] - 1]) + ")"

  _sel = ["bin", "freq", "miss", "bads", "rate", "woe", "iv", "rule"]

  return({"cut": _l3, "tbl": [{k: _[k] for k in _sel} for _ in _l5]})
  

##################################################################################################


def iso_bin(x, y):
  """
  The function discretizes the x vector based on the isotonic regression and then summarizes over 
  the y vector to derive the weight of evidence transformaton (WoE) and information values.

  Parameters:
    x : A numeric vector to discretize. It is a list, 1-D numpy array, or pandas series.
    y : A numeric vector with binary values of 0/1 and with the same length of x.
        It is a list, 1-D numpy array, or pandas series.

  Returns:
    A dictionary with two keys:
      "cut" : A numeric vector with cut points applied to the x vector.
      "tbl" : A list of dictionaries summarizing the binning outcome.

  Example:
    iso_bin(derog, bad)["cut"]
    # [1.0, 2.0, 3.0]

    view_bin(iso_bin(derog, bad))

    |   bin |   freq |   miss |   bads |   rate |     woe |     iv | rule                         |
    |-------|--------|--------|--------|--------|---------|--------|------------------------------|
    |     0 |    213 |    213 |     70 | 0.3286 |  0.6416 | 0.0178 | numpy.isnan($X$)             |
    |     1 |   3741 |      0 |    560 | 0.1497 | -0.3811 | 0.0828 | $X$ <= 1.0                   |
    |     2 |    478 |      0 |    121 | 0.2531 |  0.2740 | 0.0066 | ($X$ > 1.0) and ($X$ <= 2.0) |
    |     3 |    332 |      0 |     86 | 0.2590 |  0.3050 | 0.0058 | ($X$ > 2.0) and ($X$ <= 3.0) |
    |     4 |   1073 |      0 |    359 | 0.3346 |  0.6684 | 0.0978 | $X$ > 3.0                    |
  """

  _data = [_ for _ in zip(x, y, ~numpy.isnan(x))]
  _freq = len(_data)
  _bads = sum([_[1] for _ in _data])

  _x = [_[0] for _ in _data if _[2] == 1]
  _y = [_[1] for _ in _data if _[2] == 1]

  _cor = scipy.stats.spearmanr(_x, _y)[0]
  _reg = sklearn.isotonic.IsotonicRegression()

  _f = numpy.abs(_reg.fit_transform(_x, list(map(lambda y:  y * _cor / numpy.abs(_cor), _y))))

  _l1 = sorted(list(zip(_f, _x, _y)), key = lambda x: x[0])
  _l2 = [[l for l in _l1 if l[0] == f] for f in sorted(set(_f))]
  _l3 = [[*set(_[0] for _ in l), 
          max(_[1] for _ in l), 
          numpy.mean([_[2] for _ in l]),
          sum(_[2] for _ in l)] for l in _l2]
  
  _c = sorted([_[1] for _ in [l for l in _l3 if l[2] < 1 and l[2] > 0 and l[3] > 1]])
  _p = _c[1:-1] if len(_c) > 2 else _c[:-1]

  _l4 = sorted(manual_bin(_x, _y, _p), key = lambda x: x["bads"] / x["freq"])

  if len([_ for _ in _data if _[2] == 0]) > 0:
    _m1 = miss_bin([_[1] for _ in _data if _[2] == 0])
    if _m1["bads"] == 0: 
      for _ in ['freq', 'miss', 'bads']:
        _l4[0][_]  = _l4[0][_]  + _m1[_] 
    elif _m1["freq"] == _m1["bads"]: 
      for _ in ['freq', 'miss', 'bads']:
        _l4[len(_l4) - 1][_]  = _l4[len(_l4) - 1][_]  + _m1[_]
    else:
      _l4 = [_m1] + _l4 

  _l5 = sorted([{**_, 
                 "rate": round(_["bads"] / _["freq"], 4),
                 "woe" : round(numpy.log((_["bads"] / _bads) / ((_["freq"] - _["bads"]) / (_freq - _bads))), 4),
                 "iv"  : round((_["bads"] / _bads - (_["freq"] - _["bads"]) / (_freq - _bads)) * 
                               numpy.log((_["bads"] / _bads) / ((_["freq"] - _["bads"]) / (_freq - _bads))), 4)} 
                for _ in _l4], key = lambda x: x["bin"])

  for _ in _l5:
    if _["bin"] == 0:
      _["rule"] = "numpy.isnan($X$)"
    elif _["bin"] == len(_p) + 1:
      if _["miss"] == 0:
        _["rule"] = "$X$ > " + str(_p[-1])
      else:
        _["rule"] = "($X$ > " + str(_p[-1]) + ") or numpy.isnan($X$)"
    elif _["bin"] == 1:
      if _["miss"] == 0:
        _["rule"] = "$X$ <= " + str(_p[0])
      else:
        _["rule"] = "($X$ <= " + str(_p[0]) + ") or numpy.isnan($X$)"
    else:
        _["rule"] = "($X$ > " + str(_p[_["bin"] - 2]) + ") and ($X$ <= " + str(_p[_["bin"] - 1]) + ")"

  _sel = ["bin", "freq", "miss", "bads", "rate", "woe", "iv", "rule"]

  return({"cut": _p, "tbl": [{k: _[k] for k in _sel} for _ in _l5]})


##################################################################################################


def rng_bin(x, y):
  """
  The function discretizes the x vector based on the equal-width range and then summarizes over the
  y vector to derive the weight of evidence transformaton (WoE) and information values.

  Parameters:
    x : A numeric vector to discretize. It is a list, 1-D numpy array, or pandas series.
    y : A numeric vector with binary values of 0/1 and with the same length of x.
        It is a list, 1-D numpy array, or pandas series.

  Returns:
    A dictionary with two keys:
      "cut" : A numeric vector with cut points applied to the x vector.
      "tbl" : A list of dictionaries summarizing the binning outcome.

  Example:
    rng_bin(derog, bad)["cut"]
    # [7.0, 14.0, 21.0] 

    view_bin(rng_bin(derog, bad))

    |   bin |   freq |   miss |   bads |   rate |     woe |     iv | rule                           |
    |-------|--------|--------|--------|--------|---------|--------|--------------------------------|
    |     0 |    213 |    213 |     70 | 0.3286 |  0.6416 | 0.0178 | numpy.isnan($X$)               |
    |     1 |   5243 |      0 |   1001 | 0.1909 | -0.0881 | 0.0068 | $X$ <= 7.0                     |
    |     2 |    322 |      0 |    104 | 0.3230 |  0.6158 | 0.0246 | ($X$ > 7.0) and ($X$ <= 14.0)  |
    |     3 |     46 |      0 |     15 | 0.3261 |  0.6300 | 0.0037 | ($X$ > 14.0) and ($X$ <= 21.0) |
    |     4 |     13 |      0 |      6 | 0.4615 |  1.2018 | 0.0042 | $X$ > 21.0                     |
  """

  _data = [_ for _ in zip(x, y, ~numpy.isnan(x))]
  _freq = len(_data)
  _bads = sum([_[1] for _ in _data])

  _x = [_[0] for _ in _data if _[2] == 1]
  _y = [_[1] for _ in _data if _[2] == 1]

  _n = numpy.arange(2, max(3, min(50, len(numpy.unique(_x)) - 1)))
  _p = numpy.unique([qcut(numpy.unique(_x), _) for _ in _n])

  _l1 = [[_, manual_bin(_x, _y, _)] for _ in _p]

  _l2 = [[l[0], 
          min([_["bads"] / _["freq"] for _ in l[1]]), 
          max([_["bads"] / _["freq"] for _ in l[1]]),
          scipy.stats.spearmanr([_["bin"] for _ in l[1]], [_["bads"] / _["freq"] for _ in l[1]])[0]
         ] for l in _l1]

  _l3 = [l[0] for l in sorted(_l2, key = lambda x: -len(x[0]))
         if numpy.abs(round(l[3], 8)) == 1 and round(l[1], 8) > 0 and round(l[2], 8) < 1][0]

  _l4 = sorted(*[l[1] for l in _l1 if l[0] == _l3], key = lambda x: x["bads"] / x["freq"])

  if len([_ for _ in _data if _[2] == 0]) > 0:
    _m1 = miss_bin([_[1] for _ in _data if _[2] == 0])
    if _m1["bads"] == 0:
      for _ in ['freq', 'miss', 'bads']:
        _l4[0][_]  = _l4[0][_]  + _m1[_]
    elif _m1["freq"] == _m1["bads"]:
      for _ in ['freq', 'miss', 'bads']:
        _l4[-1][_]  = _l4[-1][_]  + _m1[_]
    else:
      _l4.append(_m1)

  _l5 = sorted([{**_, 
                 "rate": round(_["bads"] / _["freq"], 4),
                 "woe" : round(numpy.log((_["bads"] / _bads) / ((_["freq"] - _["bads"]) / (_freq - _bads))), 4),
                 "iv"  : round((_["bads"] / _bads - (_["freq"] - _["bads"]) / (_freq - _bads)) *
                               numpy.log((_["bads"] / _bads) / ((_["freq"] - _["bads"]) / (_freq - _bads))), 4)
                } for _ in _l4], key = lambda x: x["bin"])

  for _ in _l5:
    if _["bin"] == 0:
      _["rule"] = "numpy.isnan($X$)"
    elif _["bin"] == len(_l3) + 1:
      if _["miss"] == 0:
        _["rule"] = "$X$ > " + str(_l3[-1])
      else:
        _["rule"] = "($X$ > " + str(_l3[-1]) + ") or numpy.isnan($X$)"
    elif _["bin"] == 1:
      if _["miss"] == 0:
        _["rule"] = "$X$ <= " + str(_l3[0])
      else:
        _["rule"] = "($X$ <= " + str(_l3[0]) + ") or numpy.isnan($X$)"
    else:
        _["rule"] = "($X$ > " + str(_l3[_["bin"] - 2]) + ") and ($X$ <= " + str(_l3[_["bin"] - 1]) + ")"

  _sel = ["bin", "freq", "miss", "bads", "rate", "woe", "iv", "rule"]

  return({"cut": _l3, "tbl": [{k: _[k] for k in _sel} for _ in _l5]})

