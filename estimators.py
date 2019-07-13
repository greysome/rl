from functools import reduce
import numpy as np
import tensorflow as tf
from gym import spaces

class BaseEstimator(object):
    def __init__(self, observation_space, sess):
        self.sess = sess
        if isinstance(observation_space, spaces.Box):
            n_features = reduce(lambda x,y: x*y, observation_space.shape, 1)
            self.feature_shape = (None, n_features)
        elif isinstance(observation_space, spaces.Discrete):
            self.feature_shape = (None, 1)

class NNEstimator(BaseEstimator):
    def __init__(self, observation_space, sess, layer_spec=[24,24],
                 optimizer=tf.train.AdamOptimizer,
                 activation=None,
                 initializer=tf.contrib.layers.xavier_initializer,
                 lr=0.001):
        '''
        `observation_space`: as per `gym.core.Env`
        `sess`: `tf.Session` instance
        `layer_spec`: a list of neuron counts for each hidden layer
        e.g. [4, 4] creates 2 hidden layers with 4 neurons each
        `optimizer`: `tf.train.Optimizer` class
        `lr`: learning rate
        '''
        super().__init__(observation_space, sess)
        self.layer_spec = layer_spec
        self.optimizer_cls = optimizer
        self.activation = activation
        self.initializer = initializer
        self.lr = lr

    def build_network(self, input, layer_spec, output_n):
        '''
        `input`: input tensor
        `layer_spec`: a list of integers, representing the number of neurons
        in each hidden layer
        `output_n`: number of neurons in output layer
        '''
        layer = input
        for i, neurons in enumerate(layer_spec):
            layer = tf.layers.dense(layer, neurons,
                                    activation=self.activation,
                                    kernel_initializer=self.initializer())
        out_all = tf.layers.dense(layer, output_n,
                                  kernel_initializer=self.initializer())
        return out_all

    def index_with_vector(self, x, indices):
        range = tf.cast(tf.range(tf.shape(indices)[0]), indices.dtype)
        gather_indices = tf.stack((range, indices), axis=1)
        return tf.gather_nd(x, gather_indices)

class DiscreteQEstimator(NNEstimator):
    def __init__(self, observation_space, action_space, sess,
                 loss=tf.losses.mean_squared_error, **kwargs):
        '''
        `action_space`: as per `gym.core.Env`
        `loss`: function to calculate loss between q prediction and q target
        '''
        super().__init__(observation_space, sess, **kwargs)

        if isinstance(action_space, spaces.Box):
            raise ValueError('only discrete action spaces allowed')
        elif isinstance(action_space, spaces.Discrete):
            self.actions = action_space.n

        self.s = tf.placeholder('float64', shape=self.feature_shape)
        self.a = tf.placeholder('int64', shape=(None,))
        self.r = tf.placeholder('float64', shape=(None,))
        self.done = tf.placeholder('float64', shape=(None,))
        self.gamma = tf.placeholder('float64', shape=())

        with tf.variable_scope('q_pred_net'):
            self.q_all = self.build_network(self.s, self.layer_spec, self.actions)
        with tf.variable_scope('q_target_net'):
            self.q__all = self.build_network(self.s, self.layer_spec, self.actions)
        self.update_target_net = [tf.assign(new, old) for (new, old) in
                                  zip(tf.trainable_variables('q_target_net'),
                                      tf.trainable_variables('q_pred_net'))]

        # calculate the q target to update towards
        # after evaluating `self.q_target`, its value will be passed into
        # `self.q_target_ph`
        self.best_action = tf.argmax(self.q_all, axis=1)
        self.q__next = self.index_with_vector(self.q__all, self.best_action)
        self.q_target = self.r + (1-self.done) * (self.gamma * self.q__next)
        self.q_cur = self.index_with_vector(self.q_all, self.a)
        self.q_target_ph = tf.placeholder('float64', shape=(None,))

        self.td_error = tf.abs(self.q_cur - self.q_target_ph)
        self.loss = loss(self.q_cur, self.q_target_ph)
        self.optimizer = self.optimizer_cls(learning_rate=self.lr)
        self.update = self.optimizer.minimize(self.loss)

class DiscreteVEstimator(NNEstimator):
    def __init__(self, observation_space, action_space, sess,
                 loss=tf.losses.mean_squared_error, **kwargs):
        '''
        `action_space`: as per `gym.core.Env`
        `loss`: function to calculate loss between Q estimate and Q target
        '''
        super().__init__(observation_space, sess, **kwargs)

        self.s = tf.placeholder('float64', shape=self.feature_shape)
        self.r = tf.placeholder('float64', shape=(None,))
        self.done = tf.placeholder('float64', shape=(None,))
        self.gamma = tf.placeholder('float64', shape=())

        self.v_all = self.build_network(self.s, self.layer_spec, 1)[:,0]
        self.v_target = self.r + (1-self.done) * (self.gamma * self.v_all)
        self.v_target_ph = tf.placeholder('float64', shape=(None,))

        self.td_error = tf.abs(self.v_all - self.v_target_ph)
        self.loss = loss(self.v_all, self.v_target_ph)
        self.optimizer = self.optimizer_cls(learning_rate=self.lr)
        self.update = self.optimizer.minimize(self.loss)

class DiscretePolicyEstimator(NNEstimator):
    def __init__(self, observation_space, action_space, sess, **kwargs):
        '''
        `action_space`: as per `gym.core.Env`
        '''
        super().__init__(observation_space, sess, **kwargs)

        if isinstance(action_space, spaces.Box):
            raise ValueError('only discrete action spaces allowed')
        elif isinstance(action_space, spaces.Discrete):
            self.actions = action_space.n

        self.s = tf.placeholder('float64', shape=self.feature_shape)
        self.weight = tf.placeholder('float64', shape=(None,))
        self.a = tf.placeholder('int32', shape=(None,))

        self.action_vals = self.build_network(self.s, self.layer_spec, self.actions)
        self.action_probs = tf.nn.softmax(self.action_vals)

        self.taken_action_prob = self.index_with_vector(self.action_probs, self.a)
        self.loss = tf.reduce_mean(self.weight * tf.log(self.taken_action_prob))
        self.optimizer = self.optimizer_cls(learning_rate=self.lr)
        # minus because we're doing gradient ascent!
        self.update = self.optimizer.minimize(-self.loss)

class ActorCriticEstimator(BaseEstimator):
    def __init__(self, observation_space, action_space, sess,
                 actor_cls, critic_cls, actor_params={}, critic_params={}):
        '''
        A compound estimator consisting of estimators for both Q-value and policy.
        `action_space`: as per `gym.core.Env`
        '''
        super().__init__(observation_space, sess)

        if isinstance(action_space, spaces.Box):
            raise ValueError('only discrete action spaces allowed')
        elif isinstance(action_space, spaces.Discrete):
            self.actions = action_space.n

        self.actor = actor_cls(observation_space, action_space, sess,
                               **actor_params)
        self.critic = critic_cls(observation_space, action_space,
                                 sess, **critic_params)
