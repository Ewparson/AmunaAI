# src/data_collection.py

import pandas as pd
import json

def collect_player_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        player_df = pd.DataFrame(data)
    return player_df

def collect_game_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        game_df = pd.DataFrame(data)
    return game_df
