from typing import TypedDict, Required
import math
from basis import NoteType, Note, JudgeLine, Position, Chart, VisualNote
from bamboo import Bamboo, EventBamboo, IntegratedBamboo
import cmath


class PgrNoteDict(TypedDict):
    type: int
    time: int
    positionX: float
    holdTime: float
    speed: float
    floorPosition: float


class PgrNormalEventDict(TypedDict, total=False):
    startTime: Required[float]
    endTime: Required[float]
    start: Required[float]
    end: Required[float]
    start2: float
    end2: float


class PgrJudgeLineDict(TypedDict):
    bpm: float
    notesAbove: list[PgrNoteDict]
    notesBelow: list[PgrNoteDict]
    judgeLineMoveEvents: list[PgrNormalEventDict]
    judgeLineRotateEvents: list[PgrNormalEventDict]


class PgrChartDict(TypedDict):
    formatVersion: int
    offset: float
    judgeLineList: list[PgrJudgeLineDict]


PGR_NOTE_TYPES: list[NoteType] = [NoteType.UNKNOWN, NoteType.TAP, NoteType.DRAG, NoteType.HOLD, NoteType.FLICK]


class PgrJudgeLine(JudgeLine):
    bpm: float
    notes: list[Note]
    position: Bamboo[Position]
    angle: Bamboo[float]

    def __init__(self, dic: PgrJudgeLineDict, format_version: int, ratio: tuple[int, int]) -> None:
        super().__init__()
        self.bpm = dic['bpm']
        if not math.isfinite(self.bpm) or self.bpm <= 0:
            raise ValueError(f'invalid PGR BPM: {self.bpm}')
        beats_length = 1.875 / self.bpm
        w, h = ratio
        self.speed_scale = 0.6 * h
        self.warnings = []

        def events(name):
            invalid = 0
            for event in dic.get(name) or []:
                if event['endTime'] < event['startTime']:
                    invalid += 1
                    continue
                yield event
            if invalid:
                self.warnings.append(f'Ignored {invalid} reversed {name} events.')

        self.speed = EventBamboo(0.0)
        for event in events('speedEvents'):
            self.speed.cut(
                event['startTime'] * beats_length,
                event['endTime'] * beats_length,
                event['value'] * self.speed_scale,
                event['value'] * self.speed_scale,
            )
        self.floor = IntegratedBamboo(self.speed)
        self.opacity = EventBamboo(1.0)
        for event in events('judgeLineDisappearEvents'):
            self.opacity.cut(
                event['startTime'] * beats_length, event['endTime'] * beats_length, event['start'], event['end']
            )
        if self.opacity.events:
            self.opacity.default = self.opacity.events[0].start_value
        self.angle = EventBamboo(0.0)
        for event in events('judgeLineRotateEvents'):
            self.angle.cut(
                event['startTime'] * beats_length,
                event['endTime'] * beats_length,
                -math.radians(event['start']),
                -math.radians(event['end']),
            )
        if self.angle.events:
            self.angle.default = self.angle.events[0].start_value
        self.position = EventBamboo(complex(w, h) / 2)
        if format_version == 1:
            for event in events('judgeLineMoveEvents'):
                sv = event['start']
                ev = event['end']
                self.position.cut(
                    event['startTime'] * beats_length,
                    event['endTime'] * beats_length,
                    complex((sv // 1000) / 880 * w, h - (sv % 1000) / 520 * h),
                    complex((ev // 1000) / 880 * w, h - (ev % 1000) / 520 * h),
                )
        else:
            for event in events('judgeLineMoveEvents'):
                self.position.cut(
                    event['startTime'] * beats_length,
                    event['endTime'] * beats_length,
                    complex(event['start'] * w, h * (1 - event.get('start2', 0))),
                    complex(event['end'] * w, h * (1 - event.get('end2', 0))),
                )
        if self.position.events:
            self.position.default = self.position.events[0].start_value
        for above, key in ((True, 'notesAbove'), (False, 'notesBelow')):
            for item in dic.get(key, []):
                seconds = item['time'] * beats_length
                hold = item.get('holdTime', 0) * beats_length
                x = item['positionX'] * 0.05625 * w
                note = Note(PGR_NOTE_TYPES[item['type']], seconds, hold, complex(x))
                self.notes.append(note)
                self.visual_notes.append(
                    VisualNote(
                        note,
                        position_x=x,
                        above=above,
                        speed=item.get('speed', 1.0),
                        floor=self.floor @ seconds,
                        end_floor=self.floor @ (seconds + hold),
                    )
                )

    def notes_visible(self, seconds: float, visual: VisualNote) -> bool:
        return True

    def cover_distance(self, seconds: float, visual: VisualNote, floor: float) -> float:
        speed = 1.0 if visual.note.type == NoteType.HOLD else visual.speed
        return (visual.floor - floor) * speed

    def note_distances(self, seconds: float, visual: VisualNote, floor: float, y_control: float) -> tuple[float, float]:
        if visual.note.type != NoteType.HOLD:
            return super().note_distances(seconds, visual, floor, y_control)
        # 官谱 Hold 使用的是独立速度
        note = visual.note
        head = visual.floor - floor if seconds < note.seconds else 0.0
        remaining = max(0.0, note.seconds + note.hold - max(seconds, note.seconds))
        return head, head + remaining * visual.speed * self.speed_scale

    def pos(self, seconds: float, offset: Position) -> Position:
        angle = self.angle @ seconds
        pos = self.position @ seconds
        return pos + cmath.exp(angle * 1j) * offset

    def beat_duration(self, _: float) -> float:
        return 1.875 / self.bpm


class PgrChart(Chart):
    offset: float
    lines: list[PgrJudgeLine]
    _CHART_SIZE_V1 = (880, 520)
    _CHART_SIZE_V3 = (16, 9)

    def __init__(self, dic: PgrChartDict, ratio: tuple[int, int]) -> None:
        super().__init__()
        self.width, self.height = ratio
        self.format = 'pgr'
        version = dic['formatVersion']
        self.offset = dic.get('offset', 0.0)
        if not math.isfinite(self.offset):
            raise ValueError(f'invalid PGR offset: {self.offset}')
        self.lines = []
        for index, item in enumerate(dic['judgeLineList']):
            try:
                line = PgrJudgeLine(item, version, ratio)
            except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
                raise ValueError(f'PGR line {index}: {error}') from error
            self.lines.append(line)
            self.warnings.extend(f'PGR line {index}: {warning}' for warning in line.warnings)
