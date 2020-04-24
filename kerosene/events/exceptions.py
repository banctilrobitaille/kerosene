class UnsupportedEventException(Exception):

    def __init__(self, supported_events, unsupported_event):
        super(UnsupportedEventException, self).__init__(
            "Unsupported event provided ({}). Only {} are permitted".format(str(unsupported_event),
                                                                            [str(event) for event in
                                                                             supported_events]))
