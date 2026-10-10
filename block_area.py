from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
import math
import struct

from shapely.geometry import GeometryCollection, Polygon
from shapely.ops import unary_union


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


@lru_cache(maxsize=15)
def _ease_table(kind):
    if kind not in range(15):
        raise ValueError(f'Unsupported block-area easeType {kind}')
    values = []
    for i in range(101):
        t = f32(i / 100)
        if kind == 0:
            value = t
        elif kind == 13:
            value = 0.0
        elif kind == 14:
            value = 1.0
        else:
            power = (kind - 1) // 3 + 2
            direction = (kind - 1) % 3
            if direction == 0:
                value = f32(t ** power)
            elif direction == 1:
                value = f32(1 - f32(f32(1 - t) ** power))
            elif i < 50:
                value = f32(f32(f32(2 * i / 100) ** power) * 0.5)
            else:
                value = f32(f32(f32(1 - f32(f32(1 - f32(2 * (i - 50) / 100)) ** power)) * 0.5) + 0.5)
        values.append(value)
    return tuple(values)


def ease(progress, kind):
    table = _ease_table(kind)
    position = f32(f32(progress) * 100)
    index = int(position)
    if index < 0:
        return table[0]
    if index >= 100:
        return table[100]
    return f32(table[index] + f32(f32(position - index) * f32(table[index + 1] - table[index])))


def _vec(value):
    return f32(float(value['x'])), f32(float(value['y']))


def _safe_div(a, b):
    return 1.0 if abs(b) < max(1e-6 * abs(b), 8 * 1.401298464324817e-45) else a / b


@dataclass(frozen=True)
class Rectangle:
    center: tuple[float, float]
    size: tuple[float, float]
    angle: float
    subtract: bool
    index: int

    def polygon(self, inset=0.0):
        width, height = self.size
        if min(width, height) < 1e-4:
            return GeometryCollection()
        sign = 1 if self.subtract else -1
        hx = width * (0.5 + sign * min(0.25, inset / width))
        hy = height * (0.5 + sign * min(0.25, inset / height))
        c, s = math.cos(self.angle), math.sin(self.angle)
        x, y = self.center
        return Polygon([
            (x + dx * c - dy * s, y + dx * s + dy * c)
            for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
        ])

    def contains(self, x, y, inset=0.0):
        width, height = self.size
        if min(width, height) < 1e-4:
            return False
        dx, dy = x - self.center[0], y - self.center[1]
        c, s = math.cos(self.angle), math.sin(self.angle)
        sign = 1 if self.subtract else -1
        return (
            abs(dx * c + dy * s) <= width * (0.5 + sign * min(0.25, inset / width))
            and abs(-dx * s + dy * c) <= height * (0.5 + sign * min(0.25, inset / height))
        )


class BlockArea:
    def __init__(self, data, width, height, index=0):
        self.width, self.height, self.index = width, height, index
        self.bottom_left, self.top_right = _vec(data['bottomLeftPercentage']), _vec(data['topRightPercentage'])
        self.appear, self.enable, self.disable, self.disappear = (
            f32(float(data[k])) for k in ('appearTime', 'enableTime', 'disableTime', 'disappearTime')
        )
        self.subtract = bool(data.get('isSubtract', False))
        self.events = {}
        self.times = {}
        for key in ('scaleEvents', 'rotateEvents', 'moveEvents'):
            items = []
            for raw in data.get(key, []):
                event = dict(raw)
                for k, value in event.items():
                    if k.startswith('easeType'):
                        _ease_table(value)
                    elif isinstance(value, dict):
                        event[k] = _vec(value)
                    else:
                        event[k] = f32(float(value))
                items.append(event)
            times = tuple(e['time'] for e in items)
            if tuple(sorted(times)) != times:
                raise ValueError(f'Block {index}: unsorted {key}')
            self.events[key], self.times[key] = tuple(items), times
        numbers = [*self.bottom_left, *self.top_right, self.appear, self.enable, self.disable, self.disappear]
        for items in self.events.values():
            for event in items:
                for value in event.values():
                    numbers.extend(value if isinstance(value, tuple) else [value])
        if not all(math.isfinite(n) for n in numbers):
            raise ValueError(f'Block {index}: non-finite value')

    def _world(self, point):
        return ((point[0] - 0.5) * self.width, (point[1] - 0.5) * self.height)

    def _event(self, key, seconds):
        items = self.events[key]
        index = bisect_right(self.times[key], seconds) - 1
        if index < 0:
            return index, None
        current = items[index]
        if index == len(items) - 1:
            return index, current
        nxt = items[index + 1]
        duration = nxt['time'] - current['time']
        t = f32(f32(seconds - current['time']) / duration)
        result = dict(current)
        if key == 'rotateEvents':
            u = max(0.0, min(1.0, ease(t, current['easeType'])))
            result['rotation'] = current['rotation'] + (nxt['rotation'] - current['rotation']) * u
        else:
            k = 'scale' if key == 'scaleEvents' else 'endPosition'
            result[k] = tuple(
                current[k][axis] + (nxt[k][axis] - current[k][axis])
                * max(0.0, min(1.0, ease(t, current['easeType' + 'XY'[axis]])))
                for axis in range(2)
            )
        return index, result

    def rectangle(self, seconds):
        seconds = f32(seconds)
        lo, hi = self._world(self.bottom_left), self._world(self.top_right)
        original = ((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5)
        center = original
        size = (hi[0] - lo[0], hi[1] - lo[1])
        index, current = self._event('scaleEvents', seconds)
        if current is not None:
            items = self.events['scaleEvents']
            steps = [(items[i], items[i + 1]['scale']) for i in range(index)]
            if index < len(items) - 1:
                steps.append((items[index], current['scale']))
            for event, value in steps:
                anchor = self._world(event['anchor'])
                center = tuple(
                    anchor[i] + _safe_div(value[i], event['scale'][i]) * (center[i] - anchor[i])
                    for i in range(2)
                )
            size = tuple(abs(size[i] * current['scale'][i]) for i in range(2))
        index, current = self._event('rotateEvents', seconds)
        angle = 0.0
        if current is not None:
            items = self.events['rotateEvents']
            steps = [(items[i], items[i + 1]['rotation']) for i in range(index)]
            if index < len(items) - 1:
                steps.append((items[index], current['rotation']))
            for event, value in steps:
                anchor = self._world(event['anchor'])
                delta = math.radians(value - event['rotation'])
                c, s = math.cos(delta), math.sin(delta)
                dx, dy = center[0] - anchor[0], center[1] - anchor[1]
                center = (anchor[0] + c * dx - s * dy, anchor[1] + s * dx + c * dy)
            angle = current['rotation']
        _, current = self._event('moveEvents', seconds)
        if current is not None:
            position = self._world(current['endPosition'])
            center = tuple(center[i] + position[i] - original[i] for i in range(2))
        return Rectangle(
            (center[0] + self.width * 0.5, self.height * 0.5 - center[1]),
            tuple(abs(s) for s in size), -math.radians(angle), self.subtract, self.index
        )

    def active(self, seconds):
        return self.enable <= f32(seconds) < self.disable

    def phase(self, seconds):
        seconds = f32(seconds)
        if not self.appear <= seconds < self.disappear:
            return 'hidden'
        if self.active(seconds):
            return 'active'
        if self.enable - 0.5 <= seconds < self.enable:
            return 'ready'
        return 'disabled'


def compose(rectangles, inset=0.0):
    normal, subtract = [], GeometryCollection()
    for r in rectangles:
        polygon = r.polygon(inset)
        if r.subtract:
            subtract = subtract.symmetric_difference(polygon)
        else:
            normal.append(polygon)
    return unary_union(normal).symmetric_difference(subtract)


class BlockAreas:
    def __init__(self, entries=(), width=16.0, height=9.0):
        self.width, self.height = width, height
        self.areas = tuple(BlockArea(d, width, height, i) for i, d in enumerate(entries))
        self.key_times = tuple(sorted({
            t for b in self.areas for t in (
                b.appear, b.enable, b.disable, b.disappear, *(t for times in b.times.values() for t in times)
            )
        }))
        self._activation_index = self._index(tuple(b for b in self.areas if b.enable < b.disable))
        self._visibility_index = self._index(tuple(b for b in self.areas if b.appear < b.disappear), 'appear', 'disappear')

    @classmethod
    def _index(cls, areas, start='enable', end='disable'):
        if not areas:
            return None
        centers = sorted((getattr(b, start) + getattr(b, end)) / 2 for b in areas)
        center = centers[len(centers) // 2]
        left, right, crossing = [], [], []
        for b in areas:
            if getattr(b, end) <= center:
                left.append(b)
            elif getattr(b, start) > center:
                right.append(b)
            else:
                crossing.append(b)
        starts = tuple(sorted(crossing, key=lambda b: getattr(b, start)))
        ends = tuple(sorted(crossing, key=lambda b: getattr(b, end)))
        return (
            center, tuple(getattr(b, start) for b in starts), starts,
            tuple(getattr(b, end) for b in ends), ends,
            cls._index(left, start, end), cls._index(right, start, end)
        )

    def active_areas(self, seconds):
        return self._at(self._activation_index, seconds)

    def visible_areas(self, seconds):
        return self._at(self._visibility_index, seconds)

    @staticmethod
    def _at(node, seconds):
        seconds = f32(seconds)
        found = []
        while node is not None:
            center, starts, by_start, ends, by_end, left, right = node
            if seconds < center:
                found.extend(by_start[:bisect_right(starts, seconds)])
                node = left
            else:
                found.extend(by_end[bisect_right(ends, seconds):])
                node = right
        return tuple(sorted(found, key=lambda b: b.index))

    def __bool__(self):
        return bool(self.areas)

    @lru_cache(maxsize=8192)
    def active(self, seconds):
        return tuple(b.rectangle(seconds) for b in self.active_areas(seconds))

    @lru_cache(maxsize=4096)
    def forbidden_rectangles(self, rectangles):
        if not rectangles:
            return GeometryCollection()
        return compose(rectangles).intersection(compose(rectangles, self.height * 0.03))

    def contains(self, seconds, x, y):
        original_normal = adjusted_normal = False
        original_subtract = adjusted_subtract = 0
        for r in self.active(seconds):
            original = r.contains(x, y)
            adjusted = r.contains(x, y, self.height * 0.03)
            if r.subtract:
                original_subtract ^= original
                adjusted_subtract ^= adjusted
            else:
                original_normal |= original
                adjusted_normal |= adjusted
        return (original_normal ^ bool(original_subtract)) and (adjusted_normal ^ bool(adjusted_subtract))
