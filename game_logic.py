import numpy as np
import time
from os import makedirs

get_card_suit = lambda num: num // 10
get_card_value = lambda num: (num % 10) + 1
icons = ['♦', '♥', '♣', '♠']
card_signs = {  1:'A', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'J', 9:'Q', 10:'K'}
print_card = lambda num: card_signs[get_card_value(num)] + icons[get_card_suit(num)]

class Game:
    def __init__(self, id=0, seed=None, cards_amount=4, ts="000000") -> None:
        self.id = np.random.randint(10,1000)
        if seed is not None:
            np.random.seed(seed)
        self.ts = ts
        self.cards_amount = cards_amount
        
        self.deck = np.arange(40, dtype=np.int32)
        self.board = []
        self.distributed_cards = None


    def start_game(self):
        self.board = []
        self.distributed_cards = None
        return self.distribute_cards()
    
    def distribute_cards(self):
        self.distributed_cards = np.random.choice(self.deck, size=self.cards_amount, replace=False)
        return self.distributed_cards
    
    def endgame_points(self):
        board = np.array(self.board)
        errors = 0.
        for i in range(len(board)-1):
            if get_card_value(board[i]) > get_card_value(board[i+1]):
                errors += 1
        return 0. - errors
    
    def write_logs(self, reward):
        folder = f"results/{self.ts}"
        makedirs(folder, exist_ok=True)

        string_to_write = f"[#{self.id}] \t[ "
        for card in self.board:
            string_to_write += str(print_card(card)) + " "
        string_to_write += f"]\t{reward:4.1f}"

        with open(f"{folder}/logs.txt", "a", encoding='utf-8') as f:
            f.write(string_to_write + "\n")

    def throw_card(self, card_position) -> (np.float32, bool) :
        card = self.distributed_cards[card_position]
        self.board.append(card)
        assert len(self.board) == len(set(self.board)), "There are duplicates in the board"
        if len(self.board) == self.cards_amount:
            reward = self.endgame_points()
            self.write_logs(reward)
            return reward, True

        return 0.0, False

    