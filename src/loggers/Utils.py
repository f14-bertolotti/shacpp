import logging

class RecordCounter:
    def __init__(self):
        self._count = 0

    def count(self):
        self._count += 1
        return self._count

class ContextFilter(logging.Filter):
    def filter(self, record):
        record.record_number = RecordCounter().count()
        return True
