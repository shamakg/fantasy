from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns



#week1 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week1.csv')
#week2 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week2.csv')
#week3 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week3.csv')
#week4 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week4.csv')
week5 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week5_final2.csv')
#week6 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week6.csv')
#week7 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week7.csv')
week8 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week8_final.csv')
week9 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week9_final.csv')
#week10 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week10.csv')
week11 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week11_final.csv')
#week12 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week12.csv')
week13 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week13_final.csv')
#week14 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week14.csv')
week15 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week15_final.csv')
#week16 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week16.csv')
week17 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week17_final.csv')
#week11 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week10.csv')


#week1['Week'] = 1
#week2['Week'] = 2
#week3['Week'] = 3
#week4['Week'] = 4
week5['Week'] = 5
#week6['Week'] = 6
#week7['Week'] = 7
week8['Week'] = 8
week9['Week'] = 9
##week10['Week'] = 10
week11['Week'] = 11
#week12['Week'] = 12
week13['Week'] = 13
#week14['Week'] = 14
week15['Week'] = 15
#week16['Week'] = 16
week17['Week'] = 17

df = pd.concat(
    [week5, week8, week9, week11, week13, week15, week17],ignore_index=True)

#print(df.head(30))

all_teams = [
    'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
    'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LAR', 'OAK', 'MIA',
    'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WSH', 'None'
]
position_mapping = {
    'QB': 0,
    'RB': 1,
    'WR': 2,
    'TE': 3,
    'FLEX': 4,
    'K': 5,
    'D/ST': 6,
    'DL': 7,
    'LB': 8,
    'DB': 9,
}
injury_map = {
    '[]' : 0,
    'ACTIVE': 1,
    'QUESTIONABLE': 2,
    'DOUBTFUL': 3,
    'OUT': 4,
    'INJURY_RESERVE': 5
}

df['Position'] = df['Player Position'].map(position_mapping)
df['Player Injury'] = df['Player Injury'].map(injury_map)
df['On Bye Week'] = df['On Bye Week'].astype(float)

# Create a mapping of team abbreviations to numerical values
team_mapping = {team: idx for idx, team in enumerate(all_teams)}

# Encode 'team' and 'opp' columns
df['Team'] = df['Player Team'].map(team_mapping)
df['Opp'] = df['Player Opponent'].map(team_mapping)

columns_to_keep = ['Week','Player Name', 'Player Rank', 'Team', 'Position', 'Opp', 'Player Opp Rank', 'Player Points','Player Injury','On Bye Week']

filtered_series = df

filtered_series = filtered_series.sort_values(by=['Player Name','Week'])
filtered_series['Prev_PTS'] = filtered_series['Player Points'].shift().fillna(0)

# Set the first occurrence of 'Prev_PTS' for each player to 0
first_occurrences = filtered_series.groupby('Player Name').head(1).index
filtered_series.loc[first_occurrences, 'Prev_PTS'] = 0

expanding_mean = (
    filtered_series.groupby('Player Name')['Player Points']
    .apply(lambda x: x.shift().expanding(min_periods=1).mean())
    .astype(object)
)

# Assign the result to the 'Avg_PTS' column
filtered_series.sort_values(by=['Player Name', 'Week'], inplace=True)

# Calculate the cumulative sum of 'Player Points' for each player
filtered_series['Cumulative Points'] = filtered_series.groupby('Player Name')['Player Points'].cumsum()

# Calculate the average for all weeks preceding the current week
filtered_series['Avg_PTS'] = (filtered_series['Cumulative Points'] - filtered_series['Player Points']) / (filtered_series['Week'] - 1)
filtered_series['Avg_PTS'].fillna(0, inplace=True) 


# Replace the first occurrence of NaN with 0 for each player

#filtered_series['Avg_PTS'] = filtered_series.groupby('Player Name')['Avg_PTS'].apply(lambda x: x.fillna(0))

filtered_series['Prev_PTS'].fillna(0, inplace=True)


# Create a function to get the injury status of the QB for a given player from the same team
def get_qb_injury(player_name, week):
    qb_name = 0  # Assuming QBs are marked as 'QB' in the 'Position' column
    player_row = filtered_series[(filtered_series['Player Name'] == player_name) & (filtered_series['Week']==week)]
    if not player_row.empty:
        team = player_row.iloc[0]['Team']
        qb = filtered_series[(filtered_series['Position'] == qb_name) & (filtered_series['Team'] == team) & (filtered_series['Week']==week)].sort_values(by='Avg_PTS', ascending=False).head(1)
        #print(qb)
        if not qb.empty:
            return qb.iloc[0]['Player Injury']
    return None

# Create a function to get the injury status of the highest scoring teammate from the same team
def get_highest_scoring_teammate_injury(player_name, week):
    player_row = filtered_series[(filtered_series['Player Name'] == player_name) & (filtered_series['Week']==week)]
    if not player_row.empty:
        team = player_row.iloc[0]['Team']
        position = player_row.iloc[0]['Position']
        teammates = filtered_series[(filtered_series['Position'] != 0) & (filtered_series['Position'] != position) & (filtered_series['Team'] == team) & (filtered_series['Player Name'] != player_name) & (filtered_series['Week']==week)].sort_values(by='Avg_PTS', ascending=False).head(1)
    if not teammates.empty:
        return teammates.iloc[0]['Player Injury']
    return None

# Create a function to get the injury status of the highest scoring teammate of the same position from the same team
def get_highest_scoring_teammate_of_same_position_injury(player_name, week):
    player_row = filtered_series[(filtered_series['Player Name'] == player_name) & (filtered_series['Week']==week)]
    if not player_row.empty:
        team = player_row.iloc[0]['Team']
        position = player_row.iloc[0]['Position']
        teammates = filtered_series[(filtered_series['Position'] == position) & (filtered_series['Team'] == team) & (filtered_series['Player Name'] != player_name) & (filtered_series['Week']==week)].sort_values(by='Avg_PTS', ascending=False).head(1)
        #print(teammates)
        if not teammates.empty:
            return teammates.iloc[0]['Player Injury']
    return None

# Apply the functions to create new columns
# #filtered_series['QB Injury'] = filtered_series['Player Name'].apply(lambda x: get_qb_injury(x))
#filtered_series['Highest Scoring Teammate Injury'] = filtered_series['Player Name'].apply(lambda x: get_highest_scoring_teammate_injury(x,0))
#filtered_series['Highest Scoring Teammate of Same Position Injury'] = filtered_series['Player Name'].apply(lambda x: get_highest_scoring_teammate_of_same_position_injury(x,0))
filtered_series['QB Injury'] = filtered_series.apply(lambda x: get_qb_injury(x['Player Name'], x['Week']), axis=1)
filtered_series['HSTI'] = filtered_series.apply(lambda x: get_highest_scoring_teammate_injury(x['Player Name'], x['Week']), axis=1)
filtered_series['HSTSPI'] = filtered_series.apply(lambda x: get_highest_scoring_teammate_of_same_position_injury(x['Player Name'], x['Week']), axis=1)
# You may need to adjust the column names and conditions based on your actual DataFrame structure.
filtered_series['HSTSPI'].fillna(-1, inplace=True)
filtered_series['QB Injury'].fillna(-1, inplace=True)
filtered_series['HSTI'].fillna(-1, inplace=True)

week_7_data = filtered_series[filtered_series['Week'] == 7]
#filtered_series = filtered_series[filtered_series['Week'] != 7]
filtered_series = filtered_series.reset_index(drop=True)



numeric_columns = filtered_series.select_dtypes(include=['number'])

# Create the correlation matrix
correlation_matrix = numeric_columns.corr()

# Create a heatmap using seaborn

#plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
#plt.title("Correlation Heatmap")
#plt.show()


#filtered_series.head(50)
pd.set_option('display.max_rows', None)
#print(filtered_series[['Player Name',"QB Injury"]])
filtered_series = filtered_series.reset_index(inplace=False)


bool1 = pd.isnull(df['Player Opponent'])
#bool2 = pd.isnull(df['index'])
bool3 = pd.isnull(df['Opp'])


filtered_series = filtered_series.drop(['index'], axis=1)
print(get_qb_injury('Zach Ertz',3))
#print(df[df.isna().any(axis=1)])
#print(df.isnull().any())
pd.set_option('display.max_colwidth', None)
print(filtered_series[filtered_series['Week']==9])

filtered_series = filtered_series[filtered_series['On Bye Week']==False]
filtered_series.drop(columns=['Stats']).to_csv('final_data', header=True, index=False)
