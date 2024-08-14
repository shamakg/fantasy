
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns


filtered_series = pd.read_csv('/Users/sumesh/fantasy/final_data')
#filtered_series = filtered_series[filtered_series['Position']==2]
target = filtered_series['Player Points'].to_numpy()
features = filtered_series[['Team','Position','Week','Opp','Prev_PTS','Avg_PTS', 'Player Rank', 'Player Opp Rank', 'Player Injury',
 'On Bye Week','QB Injury','HSTI','HSTSPI']].to_numpy()
features = StandardScaler().fit_transform(features)


#features = filtered_series[['POS']].to_numpy()


print(features)

def pearson_r(y_true, y_pred):
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

# Implement Random Forest
'''
rf = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=50, n_jobs=-1)

# Perform k-fold cross-validation
folds = 10
kf = KFold(n_splits=folds, shuffle=True, random_state=42)
mse_scores = -cross_val_score(rf, features, target, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1, verbose=True)

mean_rmse = np.mean(np.sqrt(mse_scores))
std_rmse = np.std(np.sqrt(mse_scores))

print('Mean Squared Error:', round(mean_rmse, 5))
print('Standard Deviation of MAE:', round(std_rmse, 5))
'''
# Perform k-fold cross-validation and calculate the Pearson correlation coefficient for each fold
folds = 10
kf = KFold(n_splits=folds, shuffle=True,random_state=22)
pearson_r_values = []
pearson_r_train = []
pearson_r_IBM = []
mae = []
mae_ibm = []
for train_index, test_index in kf.split(features, target):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = target[train_index], target[test_index]
    y_IBM = (filtered_series['Player Projected'].to_numpy())[test_index]
    model = RandomForestRegressor(n_estimators=50, max_depth=15, max_features=5,
                                              n_jobs=-1)
                                              
    model = model.fit(X_train, y_train)
    predictions_fold = model.predict(X_test)


    pearson_r_fold = pearson_r(y_test, predictions_fold)
    pearson_r_values.append(pearson_r_fold)
    print(mean_absolute_error(y_test, predictions_fold))
    mae.append(mean_absolute_error(y_test, predictions_fold))

    mae_ibm.append(mean_absolute_error(y_test, y_IBM))
    predictions_fold_2 = model.predict(X_train)

    pearson_r_2 = pearson_r(y_train, predictions_fold_2)
    pearson_r_train.append(pearson_r_2)
    pearson_r_IBM.append(pearson_r(y_test,y_IBM))


print(pearson_r_train)
#print("IBM model",pearson_r(filtered_series['Player Points'],filtered_series['Player Projected']))
'''
plt.figure(figsize=(8, 6))
plt.plot(range(1, folds + 1), pearson_r_values, marker='o')
plt.xlabel('Cross-Validation Folds')
plt.ylabel('Pearson R')
plt.title('Pearson R for each Fold')
plt.show()
'''
print("My Model",np.mean(pearson_r_values))
print("IBM model",np.mean(pearson_r_IBM))
print("My Model MAE", np.mean(mae))
print("IBM Model MAE",np.mean(mae_ibm))
print(pearson_r_values)

plt.figure(figsize=(8, 6))
plt.scatter(predictions_fold, y_test, marker='o')
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.title('Coor')
plt.show()

feature_importances = model.feature_importances_

# Print the feature importances
for feature_name, importance in zip(['Team','Position','Week','Opp','Prev_PTS','Avg_PTS', 'Player Rank', 'Player Opp Rank', 'Player Injury',
 'On Bye Week'], feature_importances):
    print(f'{feature_name}: {importance}')

print(predictions_fold)
print(y_test)