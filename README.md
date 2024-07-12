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


### Import Fractional Dominance

```python
from Fractional import FractionalDominance
import pandas as pd
```

### Import data and organize the data (X = consumption y =  incomes)

```python
base = pd.read_csv('primomaroc.csv', sep=',')
X = base[['butane', 'sugar', 'flour', 'gasoline_all', 'diesel_fuel', 'essence']]
y = base['incomes']
```

### Fit the model (the function returns an array with all $s$-curves)

```python
model = FractionalDominance()
fractional_param = 0.91
dominance_param = 1
results = model.fit(X, y, dominance_param = dominance_param, fractional_param = fractional_param)
```

### Print the $s$-curves

```python
model.graph_all()
```

### Gini mean difference (GMD) matrix

```python
model.gmd(x)
```
