class Agent:
    def log_init(self, summary_writer):
        self._log_writer = summary_writer
        
    def train_init(self, env_definition):
        raise NotImplementedError()

    def train_step(self):
        raise NotImplementedError()

    def train_num_steps(self):
        raise NotImplementedError()

    def train_is_done(self):
        raise NotImplementedError()

    def act(self, observation):
        raise NotImplementedError()

    def parameters(self):
        raise NotImplementedError()

    def name(self):
        raise NotImplementedError()