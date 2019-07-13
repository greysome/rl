import os
from time import time
from datetime import datetime
from collections import namedtuple
from itertools import accumulate
import numpy as np
import tensorflow as tf
import gym

from estimators import *
from experience import *
from explorers import *

TrainInfo = namedtuple('TrainInfo', 'episodes transition total_reward epsilon')
Transition = namedtuple('Transition', 's a r s_ done')

def create_save_dir(prefix):
    now = datetime.now()
    dir = prefix + '-' + now.strftime('%Y%m%d-%H%M%S')
    os.makedirs(dir)
    return dir

class BaseLearner(object):
    def __init__(self, env, estimator_cls, explorer_cls,
                 render=True, env_params={}, estimator_params={},
                 explorer_params={}, gamma=0.99, checkpoint='',
                 save_interval=20):
        '''
        `env`: `str` containing Gym environment name or `gym.core.Env`
        `estimator`: `Estimator` class
        `render`: whether to render the environment
        `gamma`: reward discount factor
        `checkpoint`: optional checkpoint file to load model from
        `save_interval`: number of episodes before saving model
        '''
        if isinstance(env, str):
            self.env = gym.make(env)
        elif gym.core.Env in env.__bases__:
            self.env = env(**env_params)
        self.render = render

        self.sess = tf.Session()

        self.est = estimator_cls(self.env.observation_space,
                                 self.env.action_space, self.sess,
                                 **estimator_params)
        self.sess.run(tf.global_variables_initializer())
        self.explorer = explorer_cls(**explorer_params)

        # Saver can only be initialized after variables in estimator
        # are initialized
        self.saver = tf.train.Saver()
        self.save_interval = save_interval

        if checkpoint:
            self.save_dir = checkpoint.split('/')[0]
            self.saver.restore(self.sess, checkpoint)
            print('='*5, f'restored checkpoint {checkpoint}', '='*5)
        else:
            self.save_dir = create_save_dir((env if isinstance(env, str)
                                            else env.__name__))

        self.s = self.env.reset()  # current state
        self.fetch = 0  # total discounted reward
        self.gamma = gamma
        self.episodes = 1
        self.steps = 0

    def sample_transition(self):
        a = self.get_action(self.s)
        s_, r, done, info = self.env.step(a)
        if self.render:
            self.env.render()
        return a, s_, r, done, info

    def post_iteration(self, r, s_, done):
        self.steps += 1
        if done:
            self.fetch = 0
            self.episodes += 1
            if self.episodes % self.save_interval == 0:
                self.saver.save(self.sess, self.save_dir+'/model',
                                global_step=self.episodes)
            self.s = self.env.reset()
        else:
            self.s = s_
            self.fetch += r

    def close(self):
        self.env.close()

    def build_train_info(self, s, a, r, s_, done):
        return TrainInfo(episodes=self.episodes,
                         transition=Transition(s, a, r, s_, done),
                         total_reward=self.fetch,
                         epsilon=(self.explorer.e
                                  if isinstance(self.explorer, EpsilonGreedyExplorer)
                                  else None))

    def get_action(self, s):
        pass

class MonteCarloLearner(BaseLearner):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.s_t = []
        self.r_t = []
        self.a_t = []
        self.fetches = []
        self.t = 0

    def _accumulate_left(self, iterable, func):
        acc = list(accumulate(reversed(iterable), func))
        return list(reversed(acc))

    def iteration(self):
        a, s_, r, done, info = self.sample_transition()
        self.s_t.append(self.s)
        self.r_t.append(r)
        self.a_t.append(a)

        if done:
            self.fetches = self._accumulate_left(self.r_t,
                                                 lambda x,y: x + self.gamma*y)
            self.update_policy()
            self.t = 0
            self.s_t = []
            self.r_t = []
            self.a_t = []
            self.fetches = []

        self.t += 1

        ret_val = self.build_train_info(self.s, a, r, s_, done)
        self.post_iteration(r, s_, done)
        return ret_val
    
    def get_action(self, s):
        pass

    def update_policy(self):
        pass

class QLearning(BaseLearner):
    MAX_ERROR = 1000

    def __init__(self, env, estimator_cls=DiscreteQEstimator,
                 explorer_cls=EpsilonGreedyExplorer,
                 experience_cls=PrioritisedExperience,
                 experience_params={}, replay_batch_size=64,
                 target_update_interval=100, **kwargs):
        '''
        `experience`: `BaseExperience` class
        `e_initial`: starting value of epsilon (for epsilon-greedy
        policy)
        `e_final`: final value of epsilon
        `e_decay`: decay rate of epsilon
        `replay_batch_size`: number of experiences to replay each step
        `target_update_interval`: number of steps before updating target
        network
        '''
        super().__init__(env, estimator_cls=estimator_cls,
                         explorer_cls=explorer_cls, **kwargs)
        self.sess.run(self.est.update_target_net)
        self.experience = experience_cls(**experience_params)
        self.replay_batch_size = replay_batch_size
        self.target_update_interval = target_update_interval

    def iteration(self):
        a, s_, r, done, info = self.sample_transition()

        if isinstance(self.experience, PrioritisedExperience):
            td_error = QLearning.MAX_ERROR
            self.experience.add(self.s, a, r, s_, done, td_error)
        else:
            self.experience.add(self.s, a, r, s_, done)

        self.replay_experience()

        if self.steps % self.target_update_interval == 0:
            self.sess.run(self.est.update_target_net)

        ret_val = self.build_train_info(self.s, a, r, s_, done)
        self.post_iteration(r, s_, done)
        return ret_val
 
    def get_action(self, s):
        q_cur = self.sess.run(self.est.q_all, {self.est.s: [s]})[0]
        self.explorer.update()
        return self.explorer.choose_action(q_cur)

    def replay_experience(self):
        replay_batch = self.experience.sample(self.replay_batch_size)
        if len(replay_batch) == 0:
            return
        
        idx_list = [x[0] for x in replay_batch]
        s_list = [x[1] for x in replay_batch]
        a_list = [x[2] for x in replay_batch]
        r_list = [x[3] for x in replay_batch]
        s__list = [x[4] for x in replay_batch]
        done_list = [x[5] for x in replay_batch]

        q_target = self.sess.run(self.est.q_target, {self.est.s: s__list,
                                                     self.est.r: r_list,
                                                     self.est.done: done_list,
                                                     self.est.gamma: self.gamma})
        td_error, _ = self.sess.run([self.est.td_error, self.est.update],
                                    {self.est.s: s_list, self.est.a: a_list,
                                     self.est.q_target_ph: q_target})

        if isinstance(self.experience, PrioritisedExperience):
            for i, idx in enumerate(idx_list):
                self.experience.set_error(idx, td_error[i])

class Reinforce(MonteCarloLearner):
    def __init__(self, env, explorer_cls=BoltzmannExplorer, **kwargs):
        super().__init__(env, estimator_cls=DiscretePolicyEstimator,
                         explorer_cls=explorer_cls, **kwargs)

    def get_action(self, s):
        action_probs = self.sess.run(self.est.action_probs, {self.est.s: [s]})[0]
        self.explorer.update()
        return self.explorer.choose_action(action_probs, normalise=False)

    def update_policy(self):
        self.sess.run(self.est.update, {self.est.s: self.s_t,
                                        self.est.a: self.a_t,
                                        self.est.weight: self.fetches})

class BaseActorCritic(BaseLearner):
    def __init__(self, env, actor_cls, critic_cls,
                 explorer_cls=BoltzmannExplorer, **kwargs):
        try:
            estimator_params = kwargs['estimator_params']
        except KeyError:
            kwargs.update({'estimator_params': {'actor_cls': actor_cls,
                                                'critic_cls': critic_cls}})
        else:
            kwargs['estimator_params'].update({'actor_cls': actor_cls,
                                               'critic_cls': critic_cls})

        super().__init__(env, estimator_cls=ActorCriticEstimator,
                         explorer_cls=explorer_cls, **kwargs)

        # convenience aliases
        self.actor = self.est.actor
        self.critic = self.est.critic

    def get_action(self, s):
        action_probs = self.sess.run(self.actor.action_probs,
                                     {self.actor.s: [s]})[0]
        self.explorer.update()
        return self.explorer.choose_action(action_probs, normalise=False)

    def update_critic(self, s, *args):
        pass

    def update_actor(self, s, *args):
        pass

    def iteration(self):
        pass

class QActorCritic(BaseActorCritic):
    def __init__(self, env, **kwargs):
        super().__init__(env, DiscretePolicyEstimator, DiscreteQEstimator, **kwargs)

    def iteration(self):
        a, s_, r, done, info = self.sample_transition()

        self.update_critic(self.s, a, r, s_, done)
        self.update_actor(self.s, a)

        ret_val = self.build_train_info(self.s, a, r, s_, done)
        self.post_iteration(r, s_, done)
        return ret_val

    def update_critic(self, s, a, r, s_, done):
        a_ = self.get_action(s_)
        q__next = self.sess.run(self.critic.q_all, {self.critic.s: [s_]})[0]
        q_target = [r + (1-done) * (self.gamma * q__next[a_])]
        self.sess.run(self.critic.update, {self.critic.s: [s],
                                           self.critic.a: [a],
                                           self.critic.q_target_ph: q_target})

    def update_actor(self, s, a):
        q_cur = self.sess.run(self.critic.q_all, {self.critic.s: [s]})[0]
        self.sess.run(self.actor.update, {self.actor.s: [s],
                                          self.actor.a: [a],
                                          self.actor.weight: [q_cur[a]]})

class A2C(BaseActorCritic):
    def __init__(self, env, **kwargs):
        super().__init__(env, DiscretePolicyEstimator, DiscreteVEstimator, **kwargs)

    def iteration(self):
        a, s_, r, done, info = self.sample_transition()

        v_target = self.sess.run(self.critic.v_target, {self.critic.s: [s_],
                                                        self.critic.r: [r],
                                                        self.critic.done: [done],
                                                        self.critic.gamma: self.gamma})[0]
        self.update_critic(self.s, v_target)
        self.update_actor(self.s, a, v_target)

        ret_val = self.build_train_info(self.s, a, r, s_, done)
        self.post_iteration(r, s_, done)
        return ret_val

    def update_critic(self, s, v_target):
        self.sess.run(self.critic.update, {self.critic.s: [s],
                                           self.critic.v_target_ph: [v_target]})

    def update_actor(self, s, a, v_target):
        v_cur = self.sess.run(self.critic.v_all, {self.critic.s: [s]})[0]
        self.sess.run(self.actor.update, {self.actor.s: [s],
                                          self.actor.a: [a],
                                          self.actor.weight: [v_target-v_cur]})
