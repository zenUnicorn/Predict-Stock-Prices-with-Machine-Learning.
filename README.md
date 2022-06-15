# Predict-Stock-Prices-with-Machine-Learning.
Predict Stock Prices with Machine Learning.

Although there are plenty of different machine learning techniques for predicting stock price, including Decision Trees, Linear Regression, and Long Short-Term Memory (LTSM), among others. Although LTSM appears to be the most generally used, we utilized Decision Tree for this tutorial, and I hope you like it.

## Libraries
```python
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')
```

## Dataset
Netflix stock price dataset from www.kaggle.com (It's here in this repo by the way).

## Visualizations

```python
plt.figure(figsize=(16,8))
plt.title('Netflix Stock price', fontsize = 10)
plt.xlabel('Days', fontsize= 10)
plt.ylabel('Closing Price ($)', fontsize = 10)
plt.plot(dataset['Close'])
plt.show()
```

## Machine Learning model
```python
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)

#Create the decision tree regressor model
tree = DecisionTreeRegressor().fit(x_train, y_train)
```

## Decision Tree Predictions
```python

#Show the decion treee model prediction
decision_tree_prediction = tree.predict(x_future)
print( decision_tree_prediction )
```

## The Prediction Visualization
```python
#visualizing the data
predictions = decision_tree_prediction

#we will plot the data here
valid =  dataset[X.shape[0]:]
#Create a new column called 'Predictions' that will hold the predicted prices
valid['Predictions'] = predictions 
plt.figure(figsize=(16,7))
plt.title('Decison Tree Model', fontsize=10)
plt.xlabel('Days',fontsize=10)
plt.ylabel('Close Price ($)',fontsize=10)
plt.plot(dataset['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train', 'Val', 'Prediction' ], loc='upper left')
plt.show()
```

Check the Filename: `hrhrjjr` for the full code.
Please star this repo

Happy Hacking!
