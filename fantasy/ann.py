
import tensorflow as tf, keras
from keras.models import Model
from keras.layers import Dense
from keras import layers,losses

from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filtered_series = pd.read_csv('/Users/sumesh/fantasy/final_data')
target = filtered_series['Player Points'].to_numpy()
features = filtered_series[['Team','Position','Week','Opp','Prev_PTS','Avg_PTS', 'Player Rank', 'Player Opp Rank', 'Player Injury',
 'On Bye Week','QB Injury','HSTI','HSTSPI']].to_numpy()
features = StandardScaler().fit_transform(features)


def pearson_r(y_true, y_pred):
    '''accuracy metric pearson R'''
    # use smoothing for not resulting in NaN values
    epsilon = 10e-5
    x = y_true
    y = y_pred
    mx = np.mean(x)
    my = np.mean(y)
    xm, ym = x - mx, y - my
    r_num = np.sum(xm * ym)
    x_square_sum = np.sum(xm * xm)
    y_square_sum = np.sum(ym * ym)
    r_den = np.sqrt(x_square_sum * y_square_sum)
    r = r_num / (r_den + epsilon)
    return np.mean(r)

#10,30,50,30
class SCSModel(Model):
  def __init__(self):
    super(SCSModel, self).__init__()
    self.DeepLayers = tf.keras.Sequential([
    layers.Dense(10, activation="relu"),
    layers.Dense(50, activation="relu"),
    layers.Dense(50, activation="relu")
    #layers.Dense(30, activation="relu"),
    #layers.Dense(100, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))
      ])

    self.OutputLayer = tf.keras.Sequential([
      layers.Dense(1)])

  def call(self, x):
    '''call model'''
    encoded = self.DeepLayers(x)
    output = self.OutputLayer(encoded)
    return output

def predict(model,data):
    '''predict after model has been trained'''
    inferences = model(data)
    predictions = np.ravel(inferences)
    return np.round(predictions,1)


fold = 1
epoch = 30
rmses = []
pearson_r_values = []

kf_outer = KFold(n_splits=10, shuffle=True)

'''
Using Nestex k-fold CV
'''
for train_index_outer, test_index_outer in kf_outer.split(target):
    #Outer k-fold
    print(fold)
    fold+=1

    X_train, X_test = features[train_index_outer], features[test_index_outer]
    y_train, y_test = target[train_index_outer], target[test_index_outer]


    my_adam = tf.optimizers.Adam(learning_rate=0.01)

    model = SCSModel()
    model.compile(optimizer=my_adam, loss='mean_squared_error')

    history = model.fit(X_train, y_train, epochs=epoch, shuffle=True, verbose=True, batch_size=32, validation_data = (X_test,y_test))

    error = np.sqrt((model.evaluate(X_test, y_test)))
    rmses.append(error)
    print(error)

    xc  = list(range(1,epoch+1))
    plt.plot(xc, history.history['loss'], color='red')
    plt.plot(xc, history.history['val_loss'], color='blue')
    

    predictions = predict(model, X_test)
    pearson_r_values.append(pearson_r(y_test, predictions))


plt.show()

print("Average",np.mean(rmses))
print("STD",np.std(rmses))
print(np.mean(pearson_r_values))

print(pearson_r_values)
