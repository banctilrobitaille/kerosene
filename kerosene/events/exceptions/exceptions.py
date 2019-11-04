class UnsupportedEventException(Exception):
    def __init__(self, supported_events, unsupported_event):
        self._supported_events = supported_events
        self._unsupported_event = unsupported_event

    @property
    def supported_events(self):
        return self._supported_events

    @property
    def unsupported_event(self):
        return self._unsupported_event

    def __str__(self):
        return "Unsupported event provided ({}). Only {} are permitted".format(str(self.unsupported_event),
                                                                               [str(event) for event in
                                                                                self._supported_events])
