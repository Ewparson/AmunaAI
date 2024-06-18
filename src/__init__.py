# src/__init__.py

from .data_collection import collect_player_data, collect_game_data
from .model_training import preprocess_data, train_gpt_model
from .real_time_processing import start_server, process_real_time_data
from .elo_system import ELOSystem
