class VisdomConfiguration(object):
    def __init__(self, port, server, env):
        self._port = port
        self._server = server
        self._env = env

    @property
    def port(self):
        return self._port

    @property
    def server(self):
        return self._server

    @property
    def env(self):
        return self._env

    @classmethod
    def from_dict(cls, config_dict):
        return cls(config_dict['port'], config_dict['server'], config_dict['env'])
