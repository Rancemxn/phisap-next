from typing import NamedTuple, TypeVar, Generic, Protocol, runtime_checkable, Callable
from abc import ABCMeta, abstractmethod
import bisect
import math

from easing import EasingFunction, LINEAR, LVALUE, RVALUE
from functools import lru_cache

# 泛型约束：可以插值（跟自己相加减，跟浮点数相乘除可以得到同类型结果）

S = TypeVar('S', bound='_Interpable')


@runtime_checkable
class _Interpable(Protocol):
    @abstractmethod
    def __add__(self: S, other: S, /) -> S: ...

    @abstractmethod
    def __sub__(self: S, other: S, /) -> S: ...

    @abstractmethod
    def __mul__(self: S, other: float | int, /) -> S: ...


T = TypeVar('T', bound=_Interpable)


class Bamboo(Generic[T], metaclass=ABCMeta):
    @abstractmethod
    def __matmul__(self, time: float) -> T: ...

    @abstractmethod
    def __repr__(self) -> str: ...


def equal(a: float, b: float) -> bool:
    return math.isclose(a, b)


class Segment(NamedTuple, Generic[T]):
    start: float
    end: float
    start_value: T
    end_value: T


class BrokenBamboo(Bamboo[T]):
    segments: list[Segment[T]]

    def __init__(self) -> None:
        super().__init__()
        self.segments = []

    def cut(self, start: float, end: float, start_value: T, end_value: T) -> None:
        bisect.insort_left(self.segments, Segment(start, end, start_value, end_value), key=lambda s: s.start)

    def __matmul__(self, time: float) -> T:
        if not self.segments:
            raise ValueError('cannot sample an empty BrokenBamboo')
        right = bisect.bisect_right(self.segments, time, key=lambda s: s.start)
        if right == 0:
            return self.segments[0].start_value
        seg = self.segments[right - 1]
        if time >= seg.end:
            return seg.end_value
        t = (time - seg.start) / (seg.end - seg.start)
        return seg.start_value + (seg.end_value - seg.start_value) * t

    def __repr__(self) -> str:
        if self.segments:
            return f'''BrokenBamboo(segments={len(self.segments)})'''
        return 'BrokenBamboo(empty)'


class Joint(NamedTuple, Generic[T]):
    timestamp: float
    value: T
    easing: EasingFunction


# TODO timestamp全换成整数
# TODO 修复overlap的问题
class LivingBamboo(Bamboo[T]):
    joints: list[Joint[T]]

    def __init__(self) -> None:
        self.joints = []

    def cut(self, timestamp: float, value: T, easing: EasingFunction | None = None) -> None:
        easing = easing or LVALUE
        insert_point = bisect.bisect_left(self.joints, timestamp, key=lambda j: j.timestamp)
        if not self.joints:
            self.joints.append(Joint(timestamp, value, easing))
            return

        # 处理浮点数精度问题
        # 相近的timestamp合并成一个
        if insert_point == len(self.joints):
            if equal(self.joints[insert_point - 1].timestamp, timestamp):
                self.joints[insert_point - 1] = self.joints[insert_point - 1]._replace(value=value, easing=easing)
                return
        else:
            if equal(self.joints[insert_point].timestamp, timestamp):
                self.joints[insert_point] = self.joints[insert_point]._replace(value=value, easing=easing)
                return
            elif insert_point > 0 and equal(self.joints[insert_point - 1].timestamp, timestamp):
                self.joints[insert_point - 1] = self.joints[insert_point - 1]._replace(value=value, easing=easing)
                return

        self.joints.insert(insert_point, Joint(timestamp, value, easing))

    def embed(self, start: float, end: float, end_value: T, easing: EasingFunction) -> None:
        # assert start < end
        insert_point = bisect.bisect_left(self.joints, start, key=lambda j: j.timestamp)
        if insert_point < len(self.joints) and equal(self.joints[insert_point].timestamp, start):
            # 更新起点记录，插入终点记录
            left_easing = self.joints[insert_point].easing
            self.joints[insert_point] = self.joints[insert_point]._replace(easing=easing)
            # assert (
            #     insert_point >= len(self.joints) - 1
            #     or self.joints[insert_point + 1].timestamp >= end
            #     or equal(self.joints[insert_point + 1].timestamp, end)
            # )
            if insert_point >= len(self.joints) - 1 or not equal(self.joints[insert_point + 1].timestamp, end):
                self.joints.insert(insert_point + 1, Joint(end, end_value, left_easing))
        elif insert_point == len(self.joints):
            # 在尾部插入起点记录和终点记录
            value = self.joints[-1].value  # 继承上个值
            self.joints.append(Joint(start, value, easing))
            self.joints.append(Joint(end, end_value, self.joints[-2].easing))
        else:
            # 在中间插入起点记录，视情况更新现有终点记录/插入终点记录
            # assert self.joints[insert_point].timestamp >= end or equal(self.joints[insert_point].timestamp, end)
            if equal(self.joints[insert_point].timestamp, end):
                self.joints[insert_point] = self.joints[insert_point]._replace(value=end_value)
                self.joints.insert(insert_point, Joint(start, self.joints[insert_point - 1].value, easing))
            else:
                left_easing = self.joints[insert_point - 1].easing
                self.joints.insert(insert_point, Joint(end, end_value, left_easing))
                self.joints.insert(insert_point, Joint(start, self.joints[insert_point - 1].value, easing))

    def __matmul__(self, time: float) -> T:
        right = bisect.bisect_left(self.joints, time, key=lambda j: j.timestamp)
        left = right - 1
        if right == len(self.joints):
            return self.joints[left].value
        if self.joints[right].timestamp == time or right == 0:
            return self.joints[right].value
        start = self.joints[left]
        end = self.joints[right]
        t = start.easing((time - start.timestamp) / (end.timestamp - start.timestamp))
        return start.value + (end.value - start.value) * t

    def __repr__(self) -> str:
        if self.joints:
            return f'''LivingBamboo(min={self.joints[0].timestamp}, max={self.joints[-1].timestamp}, total_joints={len(self.joints)})'''
        return 'LivingBamboo(empty)'


class TwinBamboo(Bamboo[complex]):
    xs: Bamboo[float]
    ys: Bamboo[float]
    convert: Callable[[complex], complex] | None

    def __init__(
        self, xs: Bamboo[float], ys: Bamboo[float], convert: Callable[[complex], complex] | None = None
    ) -> None:
        super().__init__()
        self.xs = xs
        self.ys = ys
        self.convert = convert

    def __matmul__(self, time: float) -> complex:
        if self.convert:
            return self.convert(complex(self.xs @ time, self.ys @ time))
        return complex(self.xs @ time, self.ys @ time)

    def __repr__(self) -> str:
        return f'TwinBamboo(xs={self.xs}, ys={self.ys})'


class BambooGrove(Bamboo[T]):
    bamboos: list[Bamboo[T]]
    zero: T

    def __init__(self, bamboos: list[Bamboo[T]], zero: T) -> None:
        super().__init__()
        self.bamboos = bamboos
        self.zero = zero

    def __matmul__(self, time: float) -> T:
        return sum((b @ time for b in self.bamboos), self.zero)

    def __repr__(self) -> str:
        return f'BambooGrove(of {len(self.bamboos)} bamboos)'


class BambooShoot(Bamboo[T]):
    const: T

    def __init__(self, const: T) -> None:
        super().__init__()
        self.const = const

    def __matmul__(self, time: float) -> T:
        return self.const

    def __repr__(self) -> str:
        return f'BambooShoot({self.const})'


class BambooFunc(Bamboo[T]):
    def __init__(self, sample: Callable[[float], T]) -> None:
        self.sample = sample

    def __matmul__(self, time: float) -> T:
        return self.sample(time)

    def __repr__(self) -> str:
        return 'BambooFunc()'


class Event(NamedTuple, Generic[T]):
    start: float
    end: float
    start_value: T
    end_value: T
    easing: EasingFunction = LINEAR

# 保留事件自己的区间，避免插入下一事件时改变上一段的缓动

class EventBamboo(Bamboo[T]):

    def __init__(self, default: T, interpolate: Callable[[T, T, float], T] | None = None) -> None:
        self.events: list[Event[T]] = []
        self.default = default
        self.interpolate = interpolate or (lambda a, b, t: a + (b - a) * t)
        self._cache: tuple[float, T] | None = None

    def cut(self, start: float, end: float, start_value: T, end_value: T, easing: EasingFunction = LINEAR) -> None:
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError(f'invalid event interval: [{start}, {end}]')
        self._cache = None
        event = Event(start, end, start_value, end_value, easing)
        if not self.events or start >= self.events[-1].start:
            self.events.append(event)
        else:
            bisect.insort_right(self.events, event, key=lambda e: e.start)

    def event_at(self, time: float) -> Event[T] | None:
        index = bisect.bisect_right(self.events, time, key=lambda e: e.start) - 1
        return self.events[index] if index >= 0 else None

    def __matmul__(self, time: float) -> T:
        # 优化下，一帧中的音符会反复查询同一条线的位置、角度和透明度
        cache = self._cache
        if cache is not None and cache[0] == time:
            return cache[1]
        event = self.event_at(time)
        if event is None:
            return self.default
        if time >= event.end:
            value = self.interpolate(event.end_value, event.end_value, 1.0)
        elif event.start_value == event.end_value:
            value = self.interpolate(event.start_value, event.end_value, 0.0)
        else:
            t = event.easing((time - event.start) / (event.end - event.start))
            value = self.interpolate(event.start_value, event.end_value, t)
        self._cache = time, value
        return value

    def __repr__(self) -> str:
        return f'EventBamboo(events={len(self.events)}, default={self.default})'


@lru_cache(maxsize=4096)
def easing_integral(easing: EasingFunction, start: float, end: float) -> float:
    if easing is LINEAR:
        return (end * end - start * start) / 2
    if easing is LVALUE:
        return 0.0
    if easing is RVALUE:
        return end - start
    if integral := getattr(easing, 'integral', None):
        return integral(start, end)

    # 非线性速度只积分缓动的 [0, 1] 区间
    def simpson(a, b, fa, fm, fb, area, tolerance, depth):
        m = (a + b) / 2
        fl, fr = easing((a + m) / 2), easing((m + b) / 2)
        left = (m - a) * (fa + 4 * fl + fm) / 6
        right = (b - m) * (fm + 4 * fr + fb) / 6
        error = left + right - area
        if depth == 0 or abs(error) <= 15 * tolerance:
            return left + right + error / 15
        return simpson(a, m, fa, fl, fm, left, tolerance / 2, depth - 1) + simpson(
            m, b, fm, fr, fb, right, tolerance / 2, depth - 1
        )

    fa, fm, fb = easing(start), easing((start + end) / 2), easing(end)
    area = (end - start) * (fa + 4 * fm + fb) / 6
    return simpson(start, end, fa, fm, fb, area, 1e-8, 14)

# 积分速度，以 0 秒为原点

class IntegratedBamboo(Bamboo[float]):

    def __init__(self, source: EventBamboo[float]) -> None:
        self.source = source
        self.times = sorted({0.0, *(e.start for e in source.events), *(e.end for e in source.events)})
        self.floors = [0.0]
        for start, end in zip(self.times, self.times[1:]):
            self.floors.append(self.floors[-1] + self._integrate(start, end))
        self.origin = self._value(0.0)

    def _integrate(self, start: float, end: float) -> float:
        event = self.source.event_at(start)
        if event is None:
            return self.source.default * (end - start)
        if start >= event.end:
            return event.end_value * (end - start)
        if event.start_value == event.end_value:
            return event.start_value * (end - start)
        duration = event.end - event.start
        a, b = (start - event.start) / duration, (end - event.start) / duration
        return event.start_value * (end - start) + (event.end_value - event.start_value) * duration * easing_integral(
            event.easing, a, b
        )

    def _value(self, time: float) -> float:
        index = bisect.bisect_right(self.times, time) - 1
        if index < 0:
            return (time - self.times[0]) * self.source.default
        return self.floors[index] + self._integrate(self.times[index], time)

    def __matmul__(self, time: float) -> float:
        return self._value(time) - self.origin

    def __repr__(self) -> str:
        return f'IntegratedBamboo(events={len(self.source.events)})'
