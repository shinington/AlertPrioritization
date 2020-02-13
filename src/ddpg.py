#!/usr/bin/env python3

"""Reinforcement-learning based best-response policies."""
import logging
from numpy import array, float32
from time import time

from config import config
from model_simple import Model
from test import *
from listutils import *

import multiprocessing
import itertools
import numpy as np   
import tensorflow as tf
import time
import random
import scipy.stats as ss
import sys
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
#####################  hyper parameters  ##############################

MAX_EPISODES = config.getint('parameter', 'max_episodes')
MAX_EP_STEPS = config.getint('parameter', 'max_ep_steps')
LR_A = config.getfloat('parameter', 'lr_a')
LR_C = config.getfloat('parameter', 'lr_c')
GAMMA = config.getfloat('parameter', 'gamma')
TAU = config.getfloat('parameter', 'tau')
H_DEF = config.getint('parameter', 'h_def')
H_ADV = config.getint('parameter', 'h_adv')

TRAINING_MODE = config.get('parameter', 'training_mode')
TEST_MODE = config.get('parameter', 'test_mode')
MEMORY_CAPACITY = config.getint('parameter', 'memory_capacity')
MEMORY_INIT = config.getint('parameter', 'memory_init')
BATCH_SIZE = config.getint('parameter', 'batch_size')
LEARNING_STEP = config.getint('parameter', 'learning_step')
LEARNING_COUNT = config.getint('parameter', 'learning_count')
EPSILON_DISCOUNT = config.getfloat('parameter', 'epsilon_discount')
EPSILON_MAX = config.getfloat('parameter', 'epsilon_max')

TEST_EPISODES = config.getint('parameter', 'test_episodes')
TEST_EPISODES_NS = config.getint('parameter', 'test_episodes_ns')
MAX_TEST_STEPS = config.getint('parameter', 'max_test_steps')
PRINT_STEP = config.getint('parameter', 'print_step')

DATA = config.get('dataset', 'data')
EXPLORATION = config.get('parameter', 'exploration')
#########################################################################

#config = tf.ConfigProto(allow_soft_placement=True)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
#config.gpu_options.allow_growth = True

class DDPGbase(object):
    def __init__(self, a_dim, s_dim):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.is_training = tf.placeholder(tf.bool)
        #self.saver = tf.train.Saver()

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        self.Qvalue = q
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def Q_value(self, s, a):
        return self.sess.run(self.Qvalue, {self.S: s[np.newaxis, :], self.a: a[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(min(MEMORY_CAPACITY, self.pointer), size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        raise NotImplementedError

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        raise NotImplementedError

class DDPGdefend(DDPGbase):
    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            h1 = tf.layers.dense(s, H_DEF, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer(),name='h1', trainable=trainable)
            a = tf.layers.dense(h1, self.a_dim, activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='a', trainable=trainable)
            return a


    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = H_DEF
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=tf.keras.initializers.he_normal(), trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=tf.keras.initializers.he_normal(), trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            h1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            Q = tf.layers.dense(h1, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)  # Q(s,a)
            return Q

class DDPGattack(DDPGbase):
    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            h1 = tf.layers.dense(s, H_ADV, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer(),name='h1', trainable=trainable)
            a = tf.layers.dense(h1, self.a_dim, activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='a', trainable=trainable)
            return a

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = H_ADV
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=tf.keras.initializers.he_normal(), trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=tf.keras.initializers.he_normal(), trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            h1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            Q = tf.layers.dense(h1, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)  # Q(s,a)
            return Q

class DDPGlearning:
    """
    Learning algorithm inspired by Q-learning. 
    States are represented as lists of arbitrary floats, while actions are represented as normalized lists of floats (i.e., floats are greater than or equal to zero and sum up to one).
    Differences compared to Q-learning are described in the documentation of the relevant functions.
    """

    def __init__(self, mode, state_size, action_size):
        """
        Construct a learning algorithm object.
        :param state_size: Length of lists representing states.
        :param action_size: Length of lists representing actions.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.mode = mode
        self.utility = 0.0
        if self.mode == "defend":
            self.ddpg = DDPGdefend(action_size, state_size)
        else:
            self.ddpg = DDPGattack(action_size, state_size)

    def learn_from_mix(self, model, initial_state, state_observe, state_update, op_profile, op_strategy):
        """
        Q-learning based algorithm for learning the best actions in every state against mixed strategy of the opponent.
        :param model: Model of the alert prioritization problem (i.e., Model object).
        :param initial_state: Initial state, represented as an arbitrary object (note that this can be of a different format than the states used in other functions of QLearning).
        :param state_observe: Observes the state. Function, takes either initial_state or a state output by state_update, returns a list of floats (of length state_size).
        :param state_update: Updates the state based on an action. Function, takes a state (see state_observe), an action (normalized list of floats) and oppoent's action sampled from its mixed strategy, return the next state (may be arbitrary object).
        :param op_profile: List, action profile of the opponent.
        :param op_strategy: List, mixed strategy of the opponent.
        """
        #logging.info("DDPG training starts.")

        # DDPG training process
        epsilon = EPSILON_MAX
        op_actions = np.random.choice(op_profile, MAX_EPISODES, p=op_strategy)

        for i in range(MAX_EPISODES):
            global_state = initial_state
            state = np.array(state_observe(global_state),dtype=np.float32)            
            op_action = op_actions[i]
            episode_reward = 0.0

            # Training
            for j in range(MAX_EP_STEPS):
                # epsilon-greedy
                if np.random.random() >= epsilon:
                    action = self.ddpg.choose_action(state) #action has been normalized by choos_action()
                else:
                    if self.mode == 'defend':
                        if EXPLORATION == 'discrete':
                            action = normalized([np.random.randint(0,5) for k in range(self.action_size)])
                        elif EXPLORATION == 'continuous':
                            action = normalized([np.random.random() for k in range(self.action_size)])
                    else:
                        if EXPLORATION == 'discrete':                           
                            action = [np.random.randint(0,5)*0.25 for k in range(self.action_size)]
                        elif EXPLORATION == 'continuous':
                            action = [np.random.random() for k in range(self.action_size)]
                    action = np.array(action)
                                
                # action plus a noise
                #action = self.ddpg.choose_action(state)
                #action = np.clip(np.random.normal(action, 1*epsilon), 0, 1)
                
                (next_global_state, loss) = state_update(TRAINING_MODE, global_state, list(action), op_action)
                next_state = np.array(state_observe(next_global_state), dtype=np.float32)
                reward = -1.0*loss
                self.ddpg.store_transition(state, action, reward, next_state)

                global_state = next_global_state
                state = next_state
                episode_reward += reward

                if self.ddpg.pointer > MEMORY_INIT and (j+1) % LEARNING_STEP == 0:
                    for k in range(LEARNING_COUNT):
                        self.ddpg.learn()
                    if j == MAX_EP_STEPS-1:
                        if (i+1) % PRINT_STEP == 0:
                            logging.info("Episode {}, Ave step reward {}".format(i+1, episode_reward/MAX_EP_STEPS))
                        epsilon = epsilon*EPSILON_DISCOUNT
                        break

    def evaluate(self, model, initial_state, state_observe, state_update, op_profile, op_strategy):
        """
        Evaluate the agent obtained by using Q-learning.
        :param model: Model of the alert prioritization problem (i.e., Model object).
        :param initial_state: Initial state, represented as an arbitrary object (note that this can be of a different format than the states used in other functions of QLearning).
        :param state_observe: Observes the state. Function, takes either initial_state or a state output by state_update, returns a list of floats (of length state_size).
        :param state_update: Updates the state based on an action. Function, takes a state (see state_observe), an action (normalized list of floats) and oppoent's action sampled from its mixed strategy, return the next state (may be arbitrary object).
        :param op_profile: List, action profile of the opponent.
        :param op_strategy: List, mixed strategy of the opponent.
        """
        #self.ddpg.store_model()
        # DDPG test
        #logging.info("DDPG test starts.")
        total_reward = 0
        op_actions = np.random.choice(op_profile, TEST_EPISODES, p=op_strategy)            
        for i in range(TEST_EPISODES):
            global_state = initial_state
            state = np.array(state_observe(global_state),dtype=np.float32)
            episode_reward = 0.0
            op_action = op_actions[i]
            for j in range(MAX_TEST_STEPS):
                # Choose the best action by the actor network
                action = self.ddpg.choose_action(state)
                #action = np.array(normalized(test_defense_proportion(model, global_state)[0]), dtype=np.float32)

                (next_global_state, loss) = state_update(TEST_MODE, global_state, list(action), op_action)
                next_state = np.array(state_observe(next_global_state), dtype=np.float32)
                global_state = next_global_state

                state = next_state
                step_reward = -1.0*loss
                episode_reward += GAMMA**j*step_reward
            #logging.info("Episode {}, Average reward in each step {}".format(i, episode_reward))
            total_reward += episode_reward
        if TEST_EPISODES != 0:
            ave_reward = total_reward/TEST_EPISODES
            self.utility = ave_reward
            logging.info("RL utililty: {}".format(ave_reward))

        # DDPG test
        #logging.info("Non-strategic agents test starts.")
        total_reward = 0
        op_actions = np.random.choice(op_profile, TEST_EPISODES_NS, p=op_strategy)            
        for i in range(TEST_EPISODES_NS):
            global_state = initial_state
            state = np.array(state_observe(global_state),dtype=np.float32)
            episode_reward = 0.0
            op_action = op_actions[i]
            for j in range(MAX_TEST_STEPS):
                if self.mode == "defend":
                    action = np.array(normalized(test_defense_newest(model, global_state)[0]), dtype=np.float32)
                else:
                    action = np.array(test_attack_action(model, state), dtype=np.float32)
                (next_global_state, loss) = state_update(TEST_MODE, global_state, list(action), op_action)
                next_state = np.array(state_observe(next_global_state), dtype=np.float32)
                global_state = next_global_state
                state = next_state
                step_reward = -1.0*loss
                episode_reward += GAMMA**j*step_reward
            #logging.info("Episode {}, Average reward in each step {}".format(i, episode_reward))
            total_reward += episode_reward
        if TEST_EPISODES_NS != 0:
            ave_reward = total_reward/TEST_EPISODES_NS
            logging.info("Non-strategic (Uniform) utility: {}".format(ave_reward))

    def policy(self, model, state):
        """
        Get the action given by the state
        :param model: Model of the alert prioritization problem (i.e., Model object).
        :param state: State of the alert prioritization problem (i.e., Model.State object).(one-dimensional list) given a model and a state.
        :return: the action based on the policy
        """        
        feasible_action = None
        if self.mode == "defend":
            state_array = np.array(flatten_lists(state.N),dtype=np.float32)
            action = list(self.ddpg.choose_action(state_array))
            delta = model.make_investigation_feasible(state.N, unflatten_list(action, len(model.alert_types)))
            feasible_action = delta 
        else:
            state_array = np.array(flatten_state(state),dtype=np.float32)
            action = list(self.ddpg.choose_action(state_array))
            alpha = model.make_attack_feasible(action)            
            feasible_action = alpha
        return feasible_action

class DefenderOracle:
    """Best-response investigation policy for the defender against mix strategy of the attacker."""
    def __init__(self, model, attack_profile, attack_strategy, exper_index, iteration_index, trial_index):
        """
        Construct a best-response object using QLearning.
        :param model: Model of the alert prioritization problem (i.e., Model object).
        :param attack_profile: List of attack policies.
        :param attack_strategy: List of probablities of choosing policy from the attack profile.
        :param exper_index: The index of the experiment. 
        """
        self.mode = "defend"
        self.agent = DDPGlearning(self.mode, len(model.alert_types) * model.horizon, len(model.alert_types) * model.horizon)
        saver = tf.train.Saver()
        def state_update(mode, state, action, alpha):
            """
            State update function for QLearning.learn.
            :param state: State of the alert prioritization problem (i.e., Model.State object).
            :param action: Action represented as a normalized list of floats.
            :param alpha: Attack policy sampled from the attack_profile.
            :return: Next state (i.e., Model.State object).
            """
            delta = model.make_investigation_feasible(state.N, unflatten_list(action, len(model.alert_types))) # make_investigation_feasible ``unnormalizes'' the action
            next_state = model.next_state(mode, state, delta, alpha)
            loss = next_state.U - state.U
            return (next_state, loss)
        if not os.path.exists('../model/defender-{}-{}-{}/ddpg.ckpt.meta'.format(exper_index, iteration_index, trial_index)):
            print("defend-{}-{}_{}".format(exper_index, iteration_index, trial_index))
            cmd = "mkdir ../model/defender-{}-{}-{}".format(exper_index, iteration_index, trial_index)
            os.system(cmd)
            self.agent.learn_from_mix(model,
                                     Model.State(model),
                                     lambda state: flatten_lists(state.N),
                                     state_update,
                                     attack_profile,
                                     attack_strategy)
            saver.save(self.agent.ddpg.sess, "../model/defender-{}-{}-{}/ddpg.ckpt".format(exper_index, iteration_index, trial_index))        
        #else:
        #    saver.restore(self.agent.ddpg.sess, "../model/defender-{}/ddpg.ckpt".format(exper_index))
        self.agent.evaluate(model,
                           Model.State(model),
                           lambda state: flatten_lists(state.N),
                           state_update,
                           attack_profile,
                           attack_strategy)
        tf.reset_default_graph()

class AttackerOracle:
    """Best-response attack policy for the attacker against mixed strategy of defender."""
    def __init__(self, model, defense_profile, defense_strategy, exper_index, iteration_index, trial_index):
        """
        Construct a best-response object using QLearning.
        :param model: Model of the alert prioritization problem (i.e., Model object).
        :param defense_profile: List of defense policies.
        :param defense_strategy: List of probablities of choosing policy from the defense profile 
        """       
        self.mode = "attack"
        state_size = model.horizon*(len(model.alert_types) + len(model.attack_types) + len(model.alert_types) * len(model.attack_types))
        action_size = len(model.attack_types)
        self.agent = DDPGlearning(self.mode, state_size, action_size)
        saver = tf.train.Saver()
        def state_update(mode, state, action, delta):
            """
            State update function for QLearning.learn.
            :param state: State of the alert prioritization problem (i.e., Model.State object).
            :param action: Action represented as a normalized list of floats.
            :param delta: defense policy sampled from the defense_profile.
            :return: Next state (i.e., Model.State object).
            """
            alpha = model.make_attack_feasible(action)      
            next_state = model.next_state(mode, state, delta, alpha)
            loss = -1.0 * (next_state.U - state.U)
            return (next_state, loss)                        
        if not os.path.exists('../model/attacker-{}-{}-{}/ddpg.ckpt.meta'.format(exper_index, iteration_index, trial_index)):
            print("attack-{}-{}_{}".format(exper_index, iteration_index, trial_index))
            cmd = "mkdir ../model/attacker-{}-{}-{}".format(exper_index, iteration_index, trial_index)
            os.system(cmd)
            self.agent.learn_from_mix(model,
                            Model.State(model),
                            lambda state: flatten_state(state),                        
                            state_update,
                            defense_profile,
                            defense_strategy)
            saver.save(self.agent.ddpg.sess, "../model/attacker-{}-{}-{}/ddpg.ckpt".format(exper_index, iteration_index, trial_index)) 
        self.agent.evaluate(model,
                           Model.State(model),
                           lambda state: flatten_state(state),
                           state_update,
                           defense_profile,
                           defense_strategy)        
        tf.reset_default_graph()            

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s / %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info("Experiment starts.")
    if len(sys.argv) < 7:
        print(len(sys.argv))
        print("python ddpg.py [dataset] [mode] [def_budget] [adv_budget] [n_experiment] [opponent's policy]")
        sys.exit(1)
    model_name = sys.argv[1]
    mode = sys.argv[2]
    def_budget = float(sys.argv[3])
    adv_budget = float(sys.argv[4])
    n_experiment = int(sys.argv[5])
    opponent = sys.argv[6]

    if model_name == 'suricata':
        model = test_model_suricata(def_budget, adv_budget)
    elif model_name == 'fraud':
        model = test_model_fraud(def_budget, adv_budget)

    def evaluation(exper_index):
        random_seed = exper_index
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        if mode == 'defend':
            if opponent == 'uniform':
                oracle = DefenderOracle(model, [test_attack_action], [1.0], exper_index, 0, 0)
            elif opponent == 'aics':
                oracle = DefenderOracle(model, [test_attack_aics], [1.0], exper_index, 0, 0)
        elif mode == 'attack':
            if opponent == 'uniform':
                oracle = AttackerOracle(model, [test_defense_newest], [1.0], exper_index, 0, 0)
            elif opponent == 'aics':
                oracle = AttackerOracle(model, [test_defense_aics], [1.0], exper_index, 0, 0)
            elif opponent == 'icde':
                oracle = AttackerOracle(model, [test_defense_icde], [1.0], exper_index, 0, 0)
            elif opponent == 'proportion':
                oracle = AttackerOracle(model, [test_defense_proportion], [1.0], exper_index, 0, 0)
        return oracle.agent.utility

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)

    utilities = []
    for utility in pool.map(evaluation, range(n_experiment)):
        utilities.append(utility)
    logging.info("The utility of the agent:")
    print(utilities)
    print(np.mean(np.array(utilities)))
    