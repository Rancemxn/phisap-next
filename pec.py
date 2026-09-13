from typing import Self, NamedTuple
from collections import defaultdict
from dataclasses import dataclass
import re
import cmath
import math

from basis import Position, Chart, JudgeLine, NoteType, Note, VisualNote
from bamboo import EventBamboo, IntegratedBamboo
from easing import EasingFunction, LINEAR
from rpe import RPE_EASING_FUNCS
from timing import TempoMap

PEC_NOTE_TYPES = [NoteType.UNKNOWN, NoteType.TAP, NoteType.HOLD, NoteType.FLICK, NoteType.DRAG]
PEC_COMMAND_ARGS = {
    'bp': 2,
    'cv': 3,
    'cp': 4,
    'cd': 3,
    'ca': 3,
    'cm': 6,
    'cr': 5,
    'cf': 4,
    'n1': 5,
    'n2': 6,
    'n3': 5,
    'n4': 5,
    '#': 1,
    '&': 1,
}


@dataclass
class PecNote:
    type: NoteType
    time: float
    position_x: float
    speed: float = 1.0
    scale: float = 1.0
    above: bool = True
    end_time: float | None = None
    is_fake: bool = False

    def sp(self, speed: float) -> Self:
        self.speed = speed
        return self

    def sc(self, scale: float) -> Self:
        self.scale = scale
        return self

    def to_note(self) -> Note:
        end = self.time if self.end_time is None else self.end_time
        return Note(self.type, self.time, end - self.time, complex(self.position_x))


class PecEvent(NamedTuple):
    start: float
    end: float
    value: float | complex
    easing: EasingFunction = LINEAR


class PecJudgeLine(JudgeLine):
    def __init__(self, chart: 'PecChart', index: int) -> None:
        super().__init__()
        self.chart = chart
        self.index = index
        self.pec_notes: list[PecNote] = []
        self.events: defaultdict[str, list[PecEvent]] = defaultdict(list)
        self.speed = EventBamboo(0.0)

    def beat_duration(self, seconds: float) -> float:
        return self.chart.tempo.beat_duration(seconds)

    def pos(self, seconds: float, offset: Position) -> Position:
        return (self.position @ seconds) + cmath.exp((self.angle @ seconds) * 1j) * offset

    def convert_events(self, name: str, default):
        result = EventBamboo(default)
        value, last_end, overlaps = default, -math.inf, 0
        # 对齐 Phira：先按结束时间、开始时间排序，再裁剪后一个事件的起点。
        for event in sorted(self.events[name], key=lambda e: (e.end, e.start)):
            start = max(event.start, last_end)
            if start > event.start:
                overlaps += 1
            if start == event.end:
                result.cut(start, start, event.value, event.value)
            else:
                result.cut(start, event.end, value, event.value, event.easing)
            value, last_end = event.value, event.end
        if result.events:
            result.default = result.events[0].start_value
        if overlaps:
            self.chart.warnings.append(f'Clipped {overlaps} overlapping PEC {name} events on line {self.index}.')
        return result

    def finish(self) -> None:
        self.position = self.convert_events('position', complex(self.chart.width, self.chart.height) / 2)
        self.angle = self.convert_events('angle', 0.0)
        self.opacity = self.convert_events('opacity', 1.0)
        self.floor = IntegratedBamboo(self.speed)
        for pec in self.pec_notes:
            note = pec.to_note()
            visual = VisualNote(
                note,
                position_x=pec.position_x,
                speed=pec.speed,
                size=pec.scale,
                above=pec.above,
                is_fake=pec.is_fake,
                floor=self.floor @ note.seconds,
                end_floor=self.floor @ (note.seconds + note.hold),
            )
            self.visual_notes.append(visual)
            if not pec.is_fake:
                self.notes.append(note)
        self.notes.sort(key=lambda n: n.seconds)
        self.visual_notes.sort(key=lambda n: n.note.seconds)
        self.pec_notes.clear()
        self.events.clear()

    def notes_visible(self, seconds: float, visual: VisualNote) -> bool:
        alpha = self.opacity @ seconds
        if alpha >= 0:
            return True
        code = math.floor(-alpha)
        if code == 1:
            return False
        if 100 <= code < 1000:
            appear_beats = (code - 100) / 10
            start_beats = self.chart.tempo.beats(visual.note.seconds)
            return seconds >= self.chart.tempo.seconds(start_beats - appear_beats)
        return True


class PecChart(Chart):
    _CHART_WIDTH = 2048
    _CHART_HEIGHT = 1400

    def __init__(self, content: str, ratio: tuple[int, int]):
        super().__init__()
        self.width, self.height = ratio
        self.format = 'pec'
        self.judge_lines: dict[int, PecJudgeLine] = {}
        commands = self.parse(content)
        self.tempo = TempoMap(args for _, command, args in commands if command == 'bp')
        self.bpss = self.tempo.events

        last_note = None
        for line_number, command, args in commands:
            try:
                if command == '#':
                    if last_note is None:
                        raise ValueError('speed modifier has no preceding note')
                    last_note.sp(args[0])
                    continue
                if command == '&':
                    if last_note is None:
                        raise ValueError('size modifier has no preceding note')
                    last_note.sc(args[0])
                    continue
                if command == 'bp':
                    continue
                index = self.integer(args[0], 'line index')
                if index < 0:
                    raise ValueError('line index cannot be negative')
                if index not in self.judge_lines:
                    self.judge_lines[index] = PecJudgeLine(self, index)
                line = self.judge_lines[index]
                if command.startswith('n'):
                    kind = PEC_NOTE_TYPES[int(command[1])]
                    if kind == NoteType.HOLD:
                        start, end, x, above, fake = args[1:]
                        end = self._beats_to_seconds(end)
                    else:
                        start, x, above, fake = args[1:]
                        end = None
                    start = self._beats_to_seconds(start)
                    if end is not None and end < start:
                        raise ValueError('hold ends before it starts')
                    self.integer(above, 'note side')
                    self.integer(fake, 'fake flag')
                    last_note = PecNote(
                        kind,
                        start,
                        x / self._CHART_WIDTH * self.width,
                        above=above == 1,
                        end_time=end,
                        is_fake=fake != 0,
                    )
                    line.pec_notes.append(last_note)
                    continue

                start = self._beats_to_seconds(args[1])
                is_motion = command in ('cm', 'cr', 'cf')
                end = self._beats_to_seconds(args[2]) if is_motion else start
                if end < start:
                    raise ValueError('event ends before it starts')
                values = args[3:] if is_motion else args[2:]
                easing = LINEAR
                if command in ('cm', 'cr'):
                    kind = self.integer(values[-1], 'easing type')
                    if 0 <= kind < len(RPE_EASING_FUNCS):
                        easing = RPE_EASING_FUNCS[kind]
                    else:
                        warning = f'Unknown PEC easing type {kind}; using linear interpolation.'
                        if warning not in self.warnings:
                            self.warnings.append(warning)
                if command == 'cv':
                    speed = values[0] * self.height / 11.7
                    line.speed.cut(start, start, speed, speed)
                elif command in ('cp', 'cm'):
                    x, y = values[:2]
                    position = complex(x / self._CHART_WIDTH * self.width, (1 - y / self._CHART_HEIGHT) * self.height)
                    line.events['position'].append(PecEvent(start, end, position, easing))
                elif command in ('cd', 'cr'):
                    line.events['angle'].append(PecEvent(start, end, math.radians(values[0]), easing))
                elif command in ('ca', 'cf'):
                    alpha = values[0] / 255 if values[0] >= 0 else values[0]
                    line.events['opacity'].append(PecEvent(start, end, alpha))
            except (ValueError, IndexError) as error:
                raise ValueError(f'PEC line {line_number} ({command}): {error}') from error

        self.lines = [line for _, line in sorted(self.judge_lines.items())]
        for line in self.lines:
            line.finish()

    @staticmethod
    def integer(value: float, name: str) -> int:
        if not value.is_integer():
            raise ValueError(f'{name} must be an integer, got {value}')
        return int(value)

    def parse(self, content: str) -> list[tuple[int, str, list[float]]]:
        tokens = [
            (token, line_number)
            for line_number, line in enumerate(content.lstrip('\ufeff').splitlines(), 1)
            for token in re.findall(r'\S+', line.split('//', 1)[0])
        ]
        if not tokens:
            raise ValueError('empty PEC chart')
        offset, line_number = tokens[0]
        try:
            offset = float(offset)
            if not math.isfinite(offset):
                raise ValueError('offset must be finite')
        except ValueError as error:
            raise ValueError(f'PEC line {line_number}: invalid offset {offset!r}') from error
        self.offset = offset / 1000 - 0.15
        commands = []
        index = 1
        while index < len(tokens):
            command, line_number = tokens[index]
            count = PEC_COMMAND_ARGS.get(command)
            if count is None:
                raise ValueError(f'PEC line {line_number}: unknown command {command!r}')
            args = []
            for _ in range(count):
                index += 1
                if index >= len(tokens):
                    raise ValueError(f'PEC line {line_number} ({command}): expected {count} arguments')
                token, argument_line = tokens[index]
                try:
                    number = float(token)
                    if not math.isfinite(number):
                        raise ValueError('number must be finite')
                except ValueError as error:
                    raise ValueError(f'PEC line {argument_line} ({command}): invalid number {token!r}') from error
                args.append(number)
            commands.append((line_number, command, args))
            index += 1
        return commands

    def _beats_to_seconds(self, beats: float) -> float:
        return self.tempo.seconds(beats)


__all__ = ['PecChart']
