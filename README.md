# Fractional dominance for $s$-concentration curves

---

### Fractional dominance is a method to find the best indirect tax reforms

In this package, we find:

  * Draw $s$-curves with a given fractional dominance order
  * Compute critical ratios of costs of funds at the poverty line 
  * Minimal fractional dominance order for $s$-curves dominance
  * Find the crossing point between $s$-curves (percentiles)
  * Find the crossing point between $s$-curves (percentiles) for fixed gamma (ratio of costs)
  * Confidence intervals for $s$-curves


### Libraries
* numpy
* pandas
* matplotlib


### Import Gini PCA

```python
from Gini_PCA import GiniPca
```

### Import data and plot utilities: example on cars data


```python
from Gini_PCA import GiniPca
import torch
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D
from outliers import smirnov_grubbs as grubbs
from iteration_utilities import deepflatten
cars = pd.read_excel("cars.xlsx", index_col=0)
type(cars)
x = torch.DoubleTensor(cars.values)
```

### Run the model by setting your own Gini parameter >=0.1 and != 1

```python
gini_param = 2
model = GiniPca(gini_param)
```


### Rank matrix

```python
model.ranks(x)
```


### Gini mean difference (GMD) matrix

```python
model.gmd(x)
```
