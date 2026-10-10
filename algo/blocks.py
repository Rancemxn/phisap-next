from bisect import bisect_right
from collections import defaultdict
from functools import lru_cache

from shapely.geometry import Point, box
from shapely.ops import nearest_points

from .base import TouchAction, VirtualTouchEvent


class BlockPlanningError(RuntimeError):
    pass


def compatible_config(config, algorithm):
    result = dict(config)
    for name, default in (('flick_start', -25), ('flick_end', 25), ('flick_direction', 0), ('sample_delay', 1)):
        result['algo4_' + name] = config.get(f'algo{algorithm}_{name}', config.get('algo4_' + name, default))
    result['algo4_continue_when_failed'] = False
    return result


class BlockGuard:
    def __init__(self, chart):
        self.blocks = chart.block_areas
        self.width, self.height = chart.width, chart.height
        self.margin = chart.width * 0.001
        self.screen = box(chart.width * 0.01, chart.height * 0.01, chart.width * 0.99, chart.height * 0.99)
        self.screen = self.screen.difference(
            box(0, 0, chart.width * 0.1, chart.height * 0.1).union(
                box(chart.width * 0.9, 0, chart.width, chart.height * 0.1)
            )
        )
        self.protected = defaultdict(list)
        self.masks = {}

    def mask(self, seconds):
        key = round(seconds * 1000000)
        if key not in self.masks:
            until = seconds + 0.001
            times = [seconds, (seconds + until) / 2, until]
            i = bisect_right(self.blocks.key_times, seconds)
            while i < len(self.blocks.key_times) and self.blocks.key_times[i] < until:
                t = self.blocks.key_times[i]
                times.extend((t, max(seconds, t - 1e-6)))
                i += 1
            rectangles = tuple(dict.fromkeys(self.blocks.active(t) for t in times))
            self.masks[key] = self._mask_rectangles(rectangles)
            if len(self.masks) > 8192:
                self.masks.pop(next(iter(self.masks)))
        return self.masks[key]

    @lru_cache(maxsize=4096)
    def _mask_rectangles(self, samples):
        mask = self.blocks.forbidden_rectangles(samples[0])
        for rectangles in samples[1:]:
            other = self.blocks.forbidden_rectangles(rectangles)
            if not other.is_empty:
                mask = other if mask.is_empty else mask.union(other)
        return mask.buffer(self.margin) if not mask.is_empty else mask

    def clip(self, zone, seconds):
        mask = self.mask(seconds)
        return self._clip_geometry(zone, mask) if not mask.is_empty else zone

    @lru_cache(maxsize=8192)
    def _clip_geometry(self, zone, mask):
        return zone.difference(mask)

    def protect(self, pid, start, end):
        self.protected[pid].append((start, end))

    def finish(self, result, console):
        if not result:
            return []
        protected = {}
        for pid, ranges in self.protected.items():
            merged = []
            for start, end in sorted(ranges):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            protected[pid] = ([a for a, _ in merged], merged)

        def busy(pid, ms):
            starts, ranges = protected.get(pid, ((), ()))
            i = bisect_right(starts, ms) - 1
            return i >= 0 and ms < ranges[i][1]

        output = defaultdict(list)
        active = {}
        moved = released = 0
        first, last = min(result), max(result)
        event_times = set(result)
        ms = first
        while ms <= last:
            for event in result.get(ms, ()):
                pid = event.pointer_id
                if event.action in (TouchAction.UP, TouchAction.CANCEL):
                    if pid in active:
                        output[ms].append(event._replace(pos=active.pop(pid)))
                    continue
                if event.action == TouchAction.MOVE and pid not in active:
                    event = event._replace(action=TouchAction.DOWN)
                if event.action == TouchAction.DOWN and pid in active:
                    raise BlockPlanningError(f'Duplicate DOWN for pointer {pid} at {ms / 1000:.3f}s')
                active[pid] = event.pos
                output[ms].append(event)
            if len(active) > 10:
                raise BlockPlanningError(f'More than ten active pointers at {ms / 1000:.3f}s')
            if active:
                nearby_times = [ms / 1000, (ms + 0.5) / 1000, (ms + 1) / 1000]
                i = bisect_right(self.blocks.key_times, ms / 1000)
                while i < len(self.blocks.key_times) and self.blocks.key_times[i] < (ms + 1) / 1000:
                    nearby_times.extend((self.blocks.key_times[i], self.blocks.key_times[i] - 1e-6))
                    i += 1
                for pid, pos in list(active.items()):
                    if not any(self.blocks.contains(t, pos.real, pos.imag) for t in nearby_times):
                        continue
                    if busy(pid, ms):
                        raise BlockPlanningError(f'Blocked active contact at {ms / 1000:.3f}s (pointer {pid})')
                    free = self.clip(self.screen, ms / 1000)
                    if free.is_empty:
                        output[ms].append(VirtualTouchEvent(pos, TouchAction.UP, pid))
                        del active[pid]
                        released += 1
                    else:
                        point = nearest_points(free, Point(pos.real, pos.imag))[0]
                        target = complex(point.x, point.y)
                        output[ms].append(VirtualTouchEvent(target, TouchAction.MOVE, pid))
                        active[pid] = target
                        moved += 1
            if not active and ms not in event_times:
                ms = min((t for t in event_times if t > ms), default=last + 1)
            else:
                ms += 1
        if active:
            raise BlockPlanningError('Touch plan ends with unreleased pointers')
        console.print('噪域避让完成')
        console.print(f'调整{moved}次触点，释放{released}次指针')
        return sorted(output.items())
