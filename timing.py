import bisect
import math
from typing import NamedTuple, Iterable


class BpsInfo(NamedTuple):
    seconds: float
    beats: float
    bps: float


class TempoMap:
    def __init__(self, bpms: Iterable[tuple[float, float]]) -> None:
        # 同一拍的 BPM 以最后一条为准，社区转换器经常会重复导出它们。
        values = {}
        for beat, bpm in bpms:
            if not math.isfinite(beat) or not math.isfinite(bpm) or bpm <= 0:
                raise ValueError(f'invalid BPM event: beat={beat}, bpm={bpm}')
            values[beat] = bpm / 60
        if not values:
            raise ValueError('chart has no BPM events')
        self.events: list[BpsInfo] = []
        seconds = 0.0
        for beat, bps in sorted(values.items()):
            if self.events:
                previous = self.events[-1]
                seconds += (beat - previous.beats) / previous.bps
            self.events.append(BpsInfo(seconds, beat, bps))
        origin = self.seconds(0.0)
        self.events = [event._replace(seconds=event.seconds - origin) for event in self.events]

    def seconds(self, beats: float) -> float:
        index = max(0, bisect.bisect_right(self.events, beats, key=lambda e: e.beats) - 1)
        event = self.events[index]
        return event.seconds + (beats - event.beats) / event.bps

    def beats(self, seconds: float) -> float:
        index = max(0, bisect.bisect_right(self.events, seconds, key=lambda e: e.seconds) - 1)
        event = self.events[index]
        return event.beats + (seconds - event.seconds) * event.bps

    def beat_duration(self, seconds: float) -> float:
        index = max(0, bisect.bisect_right(self.events, seconds, key=lambda e: e.seconds) - 1)
        return 1 / self.events[index].bps
