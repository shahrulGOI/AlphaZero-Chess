from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock

import chess
import numpy as np

from AlphaBeta.config import Config
from AlphaBeta.env.chess_env import ChessEnv, Winner

logger = getLogger(__name__)

#Encapsulates all functionality related to playing game itself.
# these are from AGZ nature paper
class VisitStats:

    #holds data of PGN to be processed
    #defaultdict stats for all actions to take form
    #sum_n is sum of the n value for each of the actions in self.a
    def __init__(self):
        self.a = defaultdict(ActionStats)
        self.sum_n = 0


class ActionStats: #starts action

    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0


class ChessPlayer:
    #used to actually play the game of chess, choosing moves based on policy and value predictions.
    #from learned model on other side of a pipe

    # dot = False
    def __init__(self, config: Config, pipes=None, play_config=None, dummy=False):
        self.moves = []

        self.tree = defaultdict(VisitStats)
        self.config = config
        self.play_config = play_config or self.config.play
        self.labels_n = config.n_labels
        self.labels = config.labels
        self.move_lookup = {chess.Move.from_uci(move): i for move, i in zip(self.labels, range(self.labels_n))}
        if dummy:
            return

        self.pipe_pool = pipes
        self.node_lock = defaultdict(Lock)

    def reset(self):
        #resets the search tree and begin a new exploration
        self.tree = defaultdict(VisitStats)

    def deboog(self, env):
        print(env.testeval())

        state = state_key(env)
        my_visit_stats = self.tree[state]
        stats = []
        for action, a_s in my_visit_stats.a.items():
            moi = self.move_lookup[action]
            stats.append(np.asarray([a_s.n, a_s.w, a_s.q, a_s.p, moi]))
        stats = np.asarray(stats)
        a = stats[stats[:,0].argsort()[::-1]]

        for s in a:
            print(f'{self.labels[int(s[4])]:5}: '
                  f'n: {s[0]:3.0f} '
                  f'w: {s[1]:7.3f} '
                  f'q: {s[2]:7.3f} '
                  f'p: {s[3]:7.5f}')

    def action(self, env, can_stop = True) -> str:
        #finds what is the next best move within the current environment then returns a string of action to take

        #param chessEnv env: environment to figure out what is the action to take
        #param boolean can_stop: checks if it is allowed to take action (return home)
        #return: none if no action should be take (resignation). Else, return a string indicating the action to make within the UCI format

        self.reset()

        # for tl in range(self.play_config.thinking_loop):
        root_value, naked_value = self.search_moves(env)
        policy = self.calc_policy(env)
        my_action = int(np.random.choice(range(self.labels_n), p = self.apply_temperature(policy, env.num_halfmoves)))

        if can_stop and self.play_config.resign_threshold is not None and \
                        root_value <= self.play_config.resign_threshold \
                        and env.num_halfmoves > self.play_config.min_resign_turn:
            # noinspection PyTypeChecker
            return None
        else:
            self.moves.append([env.observation, list(policy)])
            return self.config.labels[my_action]

    def search_moves(self, env) -> (float, float):
        #looks for all possible moves using Alphazero algorithm (holds PGN data)
        #finds the highest value possible move, does so using multiple thread to get mutliple estimate for the algorithm.

        #param chessEnv env: env search for moves
        #return (float,float): maximum value predicted by thread

        futures = []
        with ThreadPoolExecutor(max_workers=self.play_config.search_threads) as executor:
            for _ in range(self.play_config.simulation_num_per_move):
                futures.append(executor.submit(self.search_my_move,env=env.copy(),is_root_node=True))

        vals = [f.result() for f in futures]

        return np.max(vals), vals[0] # vals[0] is kind of racy

    def search_my_move(self, env: ChessEnv, is_root_node=False) -> float:
        #Q and V values are for player(always white)
        #P is value for player of next_player (black or white)

        #searches possible moves, add to tree, returns best move in search

        #param ChessEnv env: env to search for move
        #param boolean is_root_node: checks if its the root node of the tree
        #return float: value of the move. calculated by getting prediction from value network

        if env.done:
            if env.winner == Winner.draw:
                return 0
            # assert env.whitewon != env.white_to_move # side to move can't be winner!
            return -1

        state = state_key(env)

        with self.node_lock[state]:
            if state not in self.tree:
                leaf_p, leaf_v = self.expand_and_evaluate(env)
                self.tree[state].p = leaf_p
                return leaf_v # I'm returning everything from the POV of side to move

            # SELECT STEP
            action_t = self.select_action_q_and_u(env, is_root_node)

            virtual_loss = self.play_config.virtual_loss

            my_visit_stats = self.tree[state]
            my_stats = my_visit_stats.a[action_t]

            my_visit_stats.sum_n += virtual_loss
            my_stats.n += virtual_loss
            my_stats.w += -virtual_loss
            my_stats.q = my_stats.w / my_stats.n

        env.step(action_t.uci())
        leaf_v = self.search_my_move(env)  # next move from enemy POV
        leaf_v = -leaf_v

        # BACKUP STEP
        # on returning search path
        # update: N, W, Q
        with self.node_lock[state]:
            my_visit_stats.sum_n += -virtual_loss + 1
            my_stats.n += -virtual_loss + 1
            my_stats.w += virtual_loss + leaf_v
            my_stats.q = my_stats.w / my_stats.n

        return leaf_v

    def expand_and_evaluate(self, env) -> (np.ndarray, float):

        #exapands a new leaf, (only once per state). refered as statelock
        #insert P(a|s), return leaf_v

        #This gets a prediction policy and value of state given by the env
        #:return (float, float): policy and value prediction for this state
        state_planes = env.canonical_input_planes()

        leaf_p, leaf_v = self.predict(state_planes)
        # these are canonical policy and value (i.e. side to move is "white")

        if not env.white_to_move:
            leaf_p = Config.flip_policy(leaf_p) # get it back to python-chess form

        return leaf_p, leaf_v

    def predict(self, state_planes):
        #receives prediction from policy and value network
        #param state_planes: observe state shown as planes
        #return (float, float): policy (prior probability of it happening leading to current state)
        #                           and value network (value of state) predicts the state.
        pipe = self.pipe_pool.pop()
        pipe.send(state_planes)
        ret = pipe.recv()
        self.pipe_pool.append(pipe)
        return ret

    #@profile
    def select_action_q_and_u(self, env, is_root_node) -> chess.Move:

        #picks the next action to explore using the alogrithm

        #pick based on action which maximizes and minimizes action value

        #param Environment env: check for next moves
        #param is_root_node: check for root node of the algorithm
        #return chess.Move: exlore potential moves
        # this method is called with state locked
        state = state_key(env)

        my_visitstats = self.tree[state]

        if my_visitstats.p is not None: #push p to edges
            tot_p = 1e-8
            for mov in env.board.legal_moves:
                mov_p = my_visitstats.p[self.move_lookup[mov]]
                my_visitstats.a[mov].p = mov_p
                tot_p += mov_p
            for a_s in my_visitstats.a.values():
                a_s.p /= tot_p
            my_visitstats.p = None

        xx_ = np.sqrt(my_visitstats.sum_n + 1)  # sqrt of sum(N(s, b); for all b)

        e = self.play_config.noise_eps
        c_puct = self.play_config.c_puct
        dir_alpha = self.play_config.dirichlet_alpha

        best_s = -999
        best_a = None
        if is_root_node:
            noise = np.random.dirichlet([dir_alpha] * len(my_visitstats.a))
        
        i = 0
        for action, a_s in my_visitstats.a.items():
            p_ = a_s.p
            if is_root_node:
                p_ = (1-e) * p_ + e * noise[i]
                i += 1
            b = a_s.q + c_puct * p_ * xx_ / (1 + a_s.n)
            if b > best_s:
                best_s = b
                best_a = action

        return best_a

    def apply_temperature(self, policy, turn):

        #applies a random fluctation to probability of choosing various actions
        #policy: list of probabilities of each action
        #turn: number of turns in the game 
        #return: policy, randomly perrubed based on temperature. High temp = more perturbation, low = less.

        tau = np.power(self.play_config.tau_decay_rate, turn + 1)
        if tau < 0.1:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(self.labels_n)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1/tau)
            ret /= np.sum(ret)
            return ret

    def calc_policy(self, env):
        #calculate probability actions based on counts
        state = state_key(env)
        my_visitstats = self.tree[state]
        policy = np.zeros(self.labels_n)
        for action, a_s in my_visitstats.a.items():
            policy[self.move_lookup[action]] = a_s.n

        policy /= np.sum(policy)
        return policy

    def sl_action(self, observation, my_action, weight=1):
        #logs the action of self.move

        #observation: FEN format observation indicating game state
        #my_action: uci format to take action
        #weight: assigned to take action when logged into self.moves
        #return str: actions, unmodified
        policy = np.zeros(self.labels_n)

        k = self.move_lookup[chess.Move.from_uci(my_action)]
        policy[k] = weight

        self.moves.append([observation, list(policy)])
        return my_action

    def finish_game(self, z):
        #update values of past moves when game has ended
        #param z: win=1, lose=-0, draw=0
        for move in self.moves:  # add this game winner result to all past moves.
            move += [z]


def state_key(env: ChessEnv) -> str:
    #param chessEnv env: env to encode
    #return str: str representation of game state
    fen = env.board.fen().rsplit(' ', 1) # drop the move clock
    return fen[0]
