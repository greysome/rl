from collections import deque
import gym
import tensorflow as tf

from learners import *
from estimators import *
from experience import *
from explorers import *
from sessions import *

from kurve_env import KurveSinglePlayer

if __name__ == '__main__':
    agent = QLearning(env=KurveSinglePlayer, checkpoint='good-kurve/model-160',
                      render=False,
                      estimator_params={'layer_spec': (100,100), 'lr':
                                        0.01, 'activation': tf.nn.relu})
    run_session(agent, TrainSession, plot=True)
