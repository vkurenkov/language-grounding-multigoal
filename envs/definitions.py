class EnvironmentDefinition:
    def __init__(self, env_constructor, **kwargs):
        self._env_constructor = env_constructor
        self._kwargs = kwargs
        self._name = self._get_name()

    def _get_name(self):
        return self._env_constructor(**self._kwargs).name()

    def build_env(self):
        return self._env_constructor(**self._kwargs)

    def name(self):
        return self._name