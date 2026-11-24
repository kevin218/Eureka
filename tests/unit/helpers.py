"""Small helpers shared by the unit tests."""


class RecordingLog:
    """Collect messages written through Eureka's logging interface."""

    def __init__(self):
        self.messages = []

    def writelog(self, message, **kwargs):
        self.messages.append(message)
