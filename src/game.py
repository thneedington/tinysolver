from pokerkit import Automation, Mode, NoLimitTexasHoldem

class Holdem:

    def __init__(self, stacks, small_blind, big_blind, ante)
        self.stacks = stacks
        self.num_players = len(stacks)
        self.small_blind: int = small_blind
        self.big_blind: int = big_blind
        self.ante: ante = ante
        self.state = self.reset_game()
        self.hand_count = 0

    def reset_game(self):
        return reset(self.stacks, self.small_blind, self.big_blind, self.ante)

    def reset(self, stacks, small_blind=50, big_blind=100, ante=0):
        '''
        reset game state to play another hand
        '''
        # automations
        automations = (
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
            Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING,
            Automation.CHIPS_PULLING,
        )

        # TODO: rotate position
        shift = self.hand_count % self.num_players
        rotated_stacks = self.stacks[shift:] + self.stacks[:shift]

        state = NoLimitTexasHoldem.create_state(
            automations = automations,
            ante_trimming_status = False,
            raw_antes = ante,
            raw_blinds_or_straddles = (small_blind, big_blind),
            min_bet = big_blind,
            raw_starting_stacks = stacks,
            player_count = len(stacks),
            mode = Mode.CASH_GAME
        )

        # deal out hole cards
        for _ in range(len(stacks)):
            state.deal_hole()

        self.hand_count += 1
        return state


    def play_hand():
        state = reset()

        while not self.state.status.is_finished():
            self.play_street()


    def play_street():
        # get action from each player until street is finished
        while state.actor_index is not None:
            action_type, amount = get_action_from_current_player()
            
            if action_type == "fold":
                state.fold()
            if action_type == "call":
                state.check_or_call()
            if action_type == "raise":
                state.complete_bet_or_raise_to(amount)

        # if the hand is over, stop
        if self.state.status.is_finished():
            return
        
        # otherwise, deal the next street
        try:
            self.state.burn_card()
            self.state.deal_board()
        except ValueError:
            # on the river
            pass
            

    def get_action_from_current_player():
        '''
         send encoded state to current actor, get action
        '''
        current_actor = self.state.actor_index
        encoded_state = encode_state(state)
        player_action, amount = self.model.get_action(current_actor, encoded_state)
        
        return player_action, amount


    def encode_state(self):
        state = self.state
        actor = state.actor_index

        if actor is None:
            return None

        hole_cards = self.encode_cards(state.hole_cards[state.actor_index], max_cards=2)
        board_cards = self.encode_cards(state.board_cards, max_cards = 5)
        
        # street
        street = np.zeros(4)
        street[state.street_index] = 1

        # position
        position = np.zeros(self.num_players)
        position[actor] = 1

        total_chips = sum(self.stacks)
        # stack size
        norm_stack = state.stacks[actor] / total_chips

        # pot size
        norm_pot = state.total_pot / total_chips

        # call amount
        call_amount = state.amount_to_call(actor)
        norm_call_amount = call_amount / state.stacks[actor] if state.stacks[actor] > 0 else 0

        # current street contributions
        street_contribs = np.array(state.street_amounts / max(self.stacks))

        return np.concatenate([
            street,
            position,
            np.array([norm_stack, norm_pot, norm_call_amount]),
            street_contribs
        ])

    
    def encode_cards(cards, max_cards = 5):
        encoding = np.zeros(max_cards * 52)
        for i, card in enumerate(cards):
            encoding[i * 52 + card.index] = 1
        
        return encoding
