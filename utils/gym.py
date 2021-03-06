from collections import deque

import cv2
import gym
import numpy as np

from gym import spaces
from multiprocessing import Process, Pipe
from copy import deepcopy


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = deepcopy(env_fn_wrapper.x)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob, reward, done, info = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == "custom_msg":
            if hasattr(env, "handle_message"):
                env.handle_message(data)
            else:
                raise NotImplementedError("Original environment does not handle custom messages.")
        else:
            raise NotImplementedError


class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def send_msg(self, msg):
        """
        Sends a message to all environments
        """
        pass


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))

        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset_at(self, at):
        self.remotes[at].send(("reset", None))
        obs, rew, done, info = self.remotes[at].recv()
        return obs, rew, done, info

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def send_msg(self, msg, at=None):
        if self.closed:
            return
        if at == None:
            for remote in self.remotes:
                remote.send(("custom_msg", msg))
        else:
            self.remotes[at].send(("custom_msg", msg))

    def __len__(self):
        return self.nenvs


class ObservationStack(gym.Wrapper):
    def __init__(self, env, stack_size):
        gym.Wrapper.__init__(self, env)

        self.env = env
        self.stack_size = stack_size
        self.observations = deque([], maxlen=stack_size)
        self.observation_space = spaces.Dict({
            i:env.observation_space for i in range(stack_size)
        })

    def reset(self):
        ob, reward, done, info = self.env.reset()
        for _ in range(self.stack_size):
            self.observations.append(ob)
        return self._get_ob(), reward, done, info

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.observations.append(ob)
        return self._get_ob(), reward, done, info

    def handle_message(self, msg):
        if hasattr(self.env, "handle_message"):
            self.env.handle_message(msg)
        else:
            raise NotImplementedError("Wrapped environment does not handle custom messages.")

    def _get_ob(self):
        assert len(self.observations) == self.stack_size

        return {i:self.observations[i] for i in range(len(self.observations))}


class LimitedSteps(gym.Wrapper):
    def __init__(self, env, max_steps=100):
        gym.Wrapper.__init__(self, env)

        self.env = env
        self.max_steps = max_steps
        self._num_steps = 0
        self.observation_space = env.observation_space

    def reset(self):
        self._num_steps = 0
        return self.env.reset()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        if self._num_steps >= self.max_steps:
            done = True

        self._num_steps += 1
        return ob, reward, done, info

    def handle_message(self, msg):
        if hasattr(self.env, "handle_message"):
            self.env.handle_message(msg)
        else:
            raise NotImplementedError("Wrapped environment does not handle custom messages.")


class RewardMinMaxScaler(gym.Wrapper):
    def __init__(self, env, min_reward, max_reward):
        gym.Wrapper.__init__(self, env)

        self.env = env
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.observation_space = env.observation_space

    def _scale(self, reward):
        return np.clip((reward - self.min_reward) / (self.max_reward - self.min_reward), 0.0, 1.0)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        reward = self._scale(reward)
        return ob, reward, done, info

    def handle_message(self, msg):
        if hasattr(self.env, "handle_message"):
            self.env.handle_message(msg)
        else:
            raise NotImplementedError("Wrapped environment does not handle custom messages.")


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=1., shape=(shp[0], shp[1], shp[2] * k))

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the models.
        You'd not belive how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=np.float32):
        out = np.concatenate(self._frames, axis=0)
        out = out.astype(dtype)
        return out


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(img):
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        img = img / 255.
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [1, 84, 84])
        return x_t.astype(np.float32)
