import pandas as pd
#merge player stats read before and after a given week (so we get points after but injury status before)
df1 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week17_before.csv')
df2 = pd.read_csv('/Users/sumesh/Desktop/Fantasy Data/fandata_week17_after.csv')
df1= df1.sort_values(by=['Player Name', 'Week'], ascending=True)
df2= df2.sort_values(by=['Player Name', 'Week'], ascending=True)

player_names_df1 = df1['Player Name']


df2_filtered = df2[df2['Player Name'].isin(player_names_df1)]

player_names_df2 = df2_filtered['Player Name']
df1_filtered =  df1[df1['Player Name'].isin(player_names_df2)]

df2_filtered = df2_filtered.reset_index(inplace=False)
df1_filtered = df1_filtered.reset_index(inplace=False)

df1_filtered['Player Points'] = df2_filtered['Player Points']

df1_filtered.head(50)
df1_filtered.to_csv('fandata_week17_final.csv', header=True, index=False)
