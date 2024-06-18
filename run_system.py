# run_system.py

from src.data_collection import collect_player_data, collect_game_data
from src.model_training import preprocess_data, train_gpt_model
from src.real_time_processing import start_server
from src.elo_system import ELOSystem

def main():
    # Collect data from JSON files exported by UE5
    player_df = collect_player_data('path/to/your/project/Saved/PlayerData.json')
    game_df = collect_game_data('path/to/your/project/Saved/GameData.json')

    # Preprocess and train model
    data = preprocess_data(player_df, game_df)
    model = train_gpt_model(data)

    # Start real-time processing server
    start_server()

if __name__ == '__main__':
    main()
