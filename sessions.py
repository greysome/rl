from plot import LivePlotter

def run_session(agent, session_cls, **session_params):
    sess = session_cls(agent, **session_params)
    sess.run()

class BaseSession(object):
    '''
    represents one training session; not to be confused with
    Tensorflow sessions
    '''
    def __init__(self, agent, plot=True):
        self.agent = agent
        self.plot = plot
        self.running = True
        if plot:
            self.plotter = LivePlotter()

    def run(self):
        while self.running:
            try:
                info = self.agent.iteration()
                if info.transition.done:
                    if self.plot:
                        self.plotter.update(info.total_reward)
                    self.post_episode(info)

            except KeyboardInterrupt:
                self.running = False
                self.agent.close()
                print('training stopped.')

    def post_episode(self, info):
        pass

class TrainSession(BaseSession):
    def __init__(self, agent, **kwargs):
        super().__init__(agent, **kwargs)

    def post_episode(self, info):
        print(f'{info.episodes} - fetch: {info.total_reward}')
        if info.episodes % self.agent.save_interval == 0:
            print('='*5, 'saving model', '='*5)

class SolveSession(BaseSession):
    def __init__(self, agent, consecutive_episodes, target_score, **kwargs):
        super().__init__(agent, **kwargs)
        self.scores = deque(maxlen=consecutive_episodes)
        self.target_score = target_score

    def post_episode(self, agent):
        scores.append(info.total_reward)
        average = sum(scores)/len(scores)
        print(f'{info.episodes} - fetch: {info.total_reward}, ' +
                f'average: {average}')
        if info.episodes >= consecutive_episodes and average >= target_score:
            print(f'solved in {info.episodes} episodes')
            self.running = False

