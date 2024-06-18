class ELOSystem:
    def __init__(self, k=32):
        self.k = k

    def calculate_elo(self, player_rating, opponent_rating, outcome):
        expected_score = 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))
        new_rating = player_rating + self.k * (outcome - expected_score)
        return new_rating
