from abc import ABC
from typing import Dict, List, Union

from kerosene.events import MonitorMode, BaseEvent, TemporalEvent
from kerosene.events.handlers.base_handler import EventHandler


class MonitorPatienceExceeded(Exception):
    pass


class MonitorInspection(object):
    def __init__(self, value=0, inspection_num=0):
        self._value = value
        self._inspection_num = inspection_num

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

    @property
    def inspection_num(self):
        return self._inspection_num

    def add_inspection(self):
        self._inspection_num = self._inspection_num + 1

    def reset_inspection_num(self):
        self._inspection_num = 0
        return self

    def with_value(self, value):
        self._value = value
        return self


class MonitorWatcher(EventHandler, ABC):
    def __init__(self, monitor_fn, mode: MonitorMode, min_delta, patience, every=1,
                 supported_events: List[Union[BaseEvent, TemporalEvent]] = None):
        super().__init__(supported_events, every)
        self._monitor_fn = monitor_fn
        self._mode = mode
        self._min_delta = min_delta
        self._patience = patience
        self._monitor_values: Dict[str, MonitorInspection] = {}

    @property
    def monitor_fn(self):
        return self._monitor_fn

    @property
    def mode(self):
        return self._mode

    @property
    def min_delta(self):
        return self._min_delta

    @property
    def patience(self):
        return self._patience

    @property
    def monitor_values(self):
        return self._monitor_values

    def watch(self, source_name, current_monitor_value):
        if source_name not in self._monitor_values.keys():
            self._monitor_values[source_name] = MonitorInspection(value=current_monitor_value)
        else:
            if self._mode is MonitorMode.MIN:
                delta = self._monitor_values[source_name].value - current_monitor_value
            else:
                delta = current_monitor_value - self._monitor_values[source_name].value

            if delta <= self._min_delta:
                self._monitor_values[source_name].with_value(current_monitor_value).reset_inspection_num()
            else:
                self._monitor_values[source_name].add_inspection()
                if self._monitor_values[source_name].inspection_num >= self._patience:
                    raise MonitorPatienceExceeded()
