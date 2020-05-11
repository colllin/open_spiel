# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python implementation for Monte Carlo Counterfactual Regret Minimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import pyspiel
import os

# Indices in the information sets for the regrets and average policy sums.
_REGRET_IDX = 0
_AVG_POLICY_IDX = 1

CHECKPOINTS_PATH = './checkpoints/'

def get_poker_betting_round_idx(state):
    # TODO
    return 0
    
class OutcomeSamplingSolver(object):
    """An implementation of the "MCCFR with Negative-Regret Pruning",
    the blueprint algorithm from Pluribus.
    
    The algorithm is summarized as "external-sampling MCCFR with two important improvements":
    1 - linear weighting for both the regret and average strategies for the first 400
        minutes of iteration
    2 - after the first 200 minutes of iteration, for 95% of game states (except states
        which are terminal or in the final betting round), current-player actions with 
        very negative regret (below -300,000,000) were not explored. Note that a regret
        floor of -310,000,000 was imposed for easier un-pruning of actions which later
        improved.
        
    See:
    Noam Brown and Tuomas Sandholm, 2019. "Superhuman AI for multiplayer poker"
    https://www.cs.cmu.edu/~noamb/papers/19-Science-Superhuman.pdf
    Noam Brown and Tuomas Sandholm, 2019. "Supplementary Materials for Superhuman 
    AI for multiplayer poker"
    https://www.cs.cmu.edu/~noamb/papers/19-Science-Superhuman_Supp.pdf
    """

    def __init__(
        self,
        game,
        update_strategy_interval_iters=10_000,
        prune_after_iter=20_000, # 200 minutes = ?? iters
        prune_regret_below=-300_000_000,
        prune_prob=0.95,
        regret_floor=-310_000_000,
        should_prune_state_fn=lambda state: 3 != get_poker_betting_round_idx(state),
        discount_until_iter=40_000, # 400 minutes = ?? iters
        discount_interval_iters=1_000, # 10 minutes = ?? iters
        should_track_state_avg_strategy_fn=lambda state: 0 == get_poker_betting_round_idx(state),
        regret_dtype=np.int32,
        phi_dtype=np.uint32,
        checkpoint_avg_strategy_at_iter=800,
        should_checkpoint_current_strategy=lambda t: (t >= 800) and ((t % 200) == 0),
    ):
        self._game = game
        self._infostates = {}  # infostate keys -> [regrets, avg strat]
        self._num_players = game.num_players()
        self._update_strategy_interval_iters = update_strategy_interval_iters
        self._prune_after_iter = prune_after_iter
        self._prune_regret_below = prune_regret_below
        self._prune_prob = prune_prob
        self._regret_floor = regret_floor
        self._should_prune_state_fn = should_prune_state_fn
        self._discount_until_iter = discount_until_iter
        self._discount_interval_iters = discount_interval_iters
        self._should_track_state_avg_strategy_fn = should_track_state_avg_strategy_fn
        self._regret_dtype = regret_dtype
        self._phi_dtype = phi_dtype
        self._checkpoint_avg_strategy_at_iter = checkpoint_avg_strategy_at_iter
        self._is_avg_strategy_checkpointed = False
        self._should_checkpoint_current_strategy = should_checkpoint_current_strategy
        self._avg_strategy_checkpoint_name = 'avg_strategy.csv'
        self._current_strategy_checkpoint_name = 'strategy_ckpt_t{t}.csv'
    
        assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, (
            "This solver requires sequential games. If you're trying to run it " +
            "on a simultaneous (or normal-form) game, please first transform it " +
            "using turn_based_simultaneous_game.")

    def iteration(self, t):
        """Performs one iteration of outcome sampling.
    
        An iteration consists of one episode for each player as the update player.
        """
        for update_player in range(self._num_players):
            state = self._game.new_initial_state()
            
            if (t % self._update_strategy_interval_iters) == 0:
                self._update_strategy(state, update_player)
                
            # self._episode(state, update_player, my_reach=1.0, opp_reach=1.0, sample_reach=1.0)
            prune_low_regrets = False
            if t > self._prune_after_iter:
                q = np.random.uniform()
                if q <= self._prune_prob:
                    prune_low_regrets = True
            self._traverse_mccfr(state, update_player, prune_low_regrets=prune_low_regrets)
                
        if (t < self._discount_until_iter) and ((t % self._discount_interval_iters) == 0):
            d = (t/self._discount_interval_iters) / (t/self._discount_interval_iters + 1)
            self._apply_discounting(d)
            
        if (t >= self._checkpoint_avg_strategy_at_iter) and not self._is_avg_strategy_checkpointed:
            self.checkpoint_and_clear_avg_strategy(self._avg_strategy_checkpoint_name)
            self._is_avg_strategy_checkpointed = True
                    
        if callable(self._should_checkpoint_current_strategy) and self._should_checkpoint_current_strategy(t):
            self.checkpoint_current_strategy(self._current_strategy_checkpoint_name.format(str(t)))
            
    def _calculate_strategy(self, cum_action_regrets):
        """Applies regret matching to get a policy.
    
        Args:
          regrets: numpy array of regrets for each action.
          num_legal_actions: number of legal actions at this state.
    
        Returns:
          numpy array of the policy indexed by the index of legal action in the
          list.
        """
        positive_regrets = cum_action_regrets.clip(0)
        regret_sum = positive_regrets.sum()
        if regret_sum > 0:
            return positive_regrets / regret_sum
        else:
            return np.ones(cum_action_regrets.size) / cum_action_regrets.size

    def _update_strategy(self, state, player):
        if state.is_terminal():
            return
        else if callable(self._is_player_in_game) and not self._is_player_in_game(state, update_player):
            return
        else if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            outcome = np.random.choice(outcomes, p=probs)
            state.apply_action(outcome)
            self._update_strategy(state, player)
        else if state.current_player() == update_player:
            infostate_info = self._lookup_infostate_info(state)

            # Skip infostates (and sub-trees?) that aren't tracking an avg policy
            # This also skips infostates after we checkpointed and cleared the avg policy.
            if infostate_info[_AVG_POLICY_IDX] is None:
                return
            
            cum_action_regrets = infostate_info[_REGRET_IDX]
            action_probs = self._calculate_strategy(cum_action_regrets)
            num_legal_actions = len(state.legal_actions())
            sampled_action_idx = np.random.choice(num_legal_actions, p=action_probs)
            
            # Increment the action counter
            infostate_info[_AVG_POLICY_IDX][sampled_action_idx] += 1
            
            sampled_action = state.legal_actions()[sampled_action_idx]
            state.apply_action(sampled_action)
            self._update_strategy(state, player)
        else:
            for action in state.legal_actions():
                self._update_strategy(state.child(action), player)

    def _apply_discounting(self, d):
        for infostate_key, infostate_info in self._infostates.items():
            # regret *= d
            if len(infostate_info) > _REGRET_IDX:
                infostate_info[_REGRET_IDX] *= d
            # cum strategy *= d
            if len(infostate_info) > _AVG_POLICY_IDX:
                infostate_info[_AVG_POLICY_IDX] *= d

    def _lookup_infostate_info(self, state):
        """Looks up an information set table for the given key.
    
        Args:
          state: the state object
    
        Returns:
          A list of:
            - regret, the cumulative action regrets as a numpy array of shape [num_legal_actions]
            - maybe: phi, the cumulative action counts as a numpy array of shape [num_legal_actions].
          Phi is only included
        """
        info_state_key = state.information_state_string(state.current_player())
        retrieved_infostate = self._infostates.get(info_state_key, None)
        if retrieved_infostate is not None:
            return retrieved_infostate
    
        num_legal_actions = len(state.legal_actions())
        initial_cum_regrets = np.zeros(num_legal_actions, dtype=self._regret_dtype))
        # If you don't pass a function to filter states which track an avg strategy, then we default to tracking the avg strategy.
        initial_cum_strategy = None
        include_cum_strategy = self._should_track_state_avg_strategy_fn(state) if callable(self._should_track_state_avg_strategy_fn) else True
        if include_cum_strategy:
            initial_cum_strategy = np.zeros(num_legal_actions, dtype=self._phi_dtype)

        infostate_info = []
        _REGRET_IDX = 0
        infostate_info.append(initial_cum_regrets)
        _AVG_POLICY_IDX = 1
        infostate_info.append(initial_cum_strategy)        
        self._infostates[info_state_key] = infostate_info
        return infostate_info

    def _traverse_mccfr(self, state, update_player, prune_low_regrets):
        if state.is_terminal():
            return state.player_return(update_player) / sample_reach, 1.0
        else if callable(self._is_player_in_game) and not self._is_player_in_game(state, update_player):
            state.apply_action(0)
            return self._traverse_mccfr(state, update_player, prune_low_regrets=prune_low_regrets)
        else if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            outcome = np.random.choice(outcomes, p=probs)
            state.apply_action(outcome)
            return self._traverse_mccfr(state, update_player, prune_low_regrets=prune_low_regrets)
        else if state.current_player() == update_player:
            infostate_info = self._lookup_infostate_info(state)
            cum_action_regrets = infostate_info[_REGRET_IDX]
            action_probs = self._calculate_strategy(cum_action_regrets)
            
            action_values = np.zeros_like(cum_action_regrets)
            prune_threshold = self._prune_regret_below if prune_low_regrets else -np.inf
            if callable(self._should_prune_state_fn) and self._should_prune_state_fn(state):
                actions_above_threshold = cum_action_regrets > prune_threshold
                # Don't prune terminal actions (Supp p.4)
                actions_terminal = np.array([state.child(action).is_terminal() for action in state.legal_actions()])
                action_explored = actions_above_threshold | actions_terminal
            else:
                # Don't prune actions on the last betting round (Supp p.4)
                action_explored = np.ones(len(cum_action_regrets), dtype=bool)
            for action_idx, action in enumerate(state.legal_actions()):
                if action_explored[action_idx]:
                    action_values[action_idx] = self._traverse_mccfr(state.child(action), update_player, prune_low_regrets=prune_low_regrets)
            state_expected_value = (action_explored * action_probs * action_values).sum()

            cum_action_regrets += action_explored * (action_values - state_expected_value)
            
            # Apply regret floor (Supp p.16)
            cum_action_regrets = cum_action_regrets.clip(self._regret_floor)
            
            infostate_info[_REGRET_IDX] = cum_action_regrets
            
            return state_expected_value
        else:
            infostate_info = self._lookup_infostate_info(state)
            cum_action_regrets = infostate_info[_REGRET_IDX]
            action_probs = self._calculate_strategy(cum_action_regrets)
            sampled_action = np.random.choice(state.legal_actions(), p=action_probs)
            state.apply_action(sampled_action)
            return self._traverse_mccfr(state, update_player, prune_low_regrets=prune_low_regrets)
            
    def checkpoint_and_clear_avg_policy(self, checkpoint_name):
        avg_strategies = []
        for infostate_key, infostate_info in self._infostates.items():
            if infostate_info[_AVG_POLICY_IDX] is not None:
                avg_strategies.append([infostate_key, infostate_info[_AVG_POLICY_IDX]])
                infostate_info[_AVG_POLICY_IDX] = None
        df = pd.Dataframe(avg_strategies, columns=['infostate_key','avg_strategy'])
        os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
        df.to_csv(os.path.join(CHECKPOINTS_PATH, checkpoint_name))
        
    def checkpoint_current_strategy(self, checkpoint_name):
        strategies = []
        for infostate_key, infostate_info in self._infostates.items():
            strategies.append([
                infostate_key,
                self._calculate_strategy(infostate_info[_REGRET_IDX]),
            ])
        df = pd.Dataframe(strategies, columns=['infostate_key','current_strategy'])
        os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
        df.to_csv(os.path.join(CHECKPOINTS_PATH, checkpoint_name))

    def callable_avg_policy(self):
        """Returns the average joint policy as a callable.
    
        The callable has a signature of the form string (information
        state key) -> list of (action, prob).
        """
        def wrap(state):
            info_state_key = state.information_state_string(state.current_player())
            legal_actions = state.legal_actions()
            infostate_info = self._lookup_infostate_info(state)
            avstrat = (
                infostate_info[_AVG_POLICY_IDX] /
                infostate_info[_AVG_POLICY_IDX].sum())
            return [(legal_actions[i], avstrat[i]) for i in range(len(legal_actions))]

        return wrap
