

from espn_api.football import League, Player, BoxPlayer
from typing import List, Tuple, Union
import datetime
import time
import json
import pandas as pd
class CustomLeague(League):
    def __init__(self, league_id: int, year: int, espn_s2=None, swid=None, fetch_league=True, debug=False):
        super().__init__(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid, debug=debug)

        if fetch_league:
            self.fetch_league()

    def free_agents_to_dataframe(self, week: int = None, size: int = 50, position: str = None, position_id: int = None) -> pd.DataFrame:
        '''Returns a Pandas DataFrame of Free Agents for a given week. This is taken from an empty league where all players are Free Agents'''

        if self.year < 2019:
            raise Exception('Can\'t use free agents before 2019')
        if not week:
            week = self.current_week

        slot_filter = []
        if position and position in POSITION_MAP:
            slot_filter = [POSITION_MAP[position]]
        if position_id:
            slot_filter.append(position_id)

        params = {
            'view': 'kona_player_info',
            'scoringPeriodId': week,
        }

        #sorts so we only get a certain amount of plauers
        filters = {"players": {"filterSlotIds": {"value": slot_filter}, "limit": 200, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}, "sortDraftRanks": {"sortPriority": 100, "sortAsc": True, "value": "STANDARD"}}}
        headers = {'x-fantasy-filter': json.dumps(filters)}

        data = self.espn_request.league_get(params=params, headers=headers)

        players = data['players']

        player_data = []

        #extract features for each player
        for player in players:
            b = BoxPlayer(player, self._get_pro_schedule(week), self._get_positional_ratings(week), week, self.year)
            player_info = {
                'Week': week,
                "Player Name": b.name,
                "Player Rank": b.posRank,
                "Player Team": b.proTeam,
                "Player Position": b.position,
                "Player Projected": b.projected_points,
                "Player Points": b.points,
                "Player Opponent": b.pro_opponent,
                "Player Opp Rank": b.pro_pos_rank,
                "Player Injury": b.injuryStatus,
                "Stats": b.stats,
                'Timed Played': b.game_played,
                'On Bye Week': b.on_bye_week
            }
            player_data.append(player_info)

        df = pd.DataFrame(player_data)

        return df


POSITION_MAP = {
    # Define position mappings here if needed
}

# Usage
#league = CustomLeague(league_id=YOUR_LEAGUE_ID, year=YEAR, espn_s2=YOUR_ESPN_S2, swid=YOUR_SWID,debug=True)

free_agents_df = league.free_agents_to_dataframe(week=18)


output_csv = 'fandata_week18_after.csv'

# Save the DataFrame to a new CSV file with headers
free_agents_df.to_csv(output_csv, header=True, index=False)
