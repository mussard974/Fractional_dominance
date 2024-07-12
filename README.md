# Fractional dominance for $s$-concentration curves

---

### Fractional dominance is a method to find the best indirect tax reforms

In this package, we find:

  * Grid search for detecting and minimizing the influence of outliers 
  * Absolute contributions of the observations 
  * Relative contributions of the observations (equivalent to cos²)
  * Feature importance of each variable (U-statistics test)
  * Outlier detection using Grubbs test 
  * Example on cars data


### Install outlier_utils and iteration-utilities
* torch >= 1.9.0
```python
!pip install outlier_utils
!pip install iteration-utilities
```

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
