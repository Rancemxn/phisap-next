from typing import TypedDict, NamedTuple
import math
import cmath

from basis import Note, NoteType, JudgeLine, Chart, Position, VisualNote
from bamboo import TwinBamboo, LivingBamboo, BambooGrove, Bamboo, BambooShoot, BambooFunc, EventBamboo, IntegratedBamboo
from easing import EasingFunction, EASING_FUNCTIONS, easing_with_range, cubic_rev_bezier, LINEAR, LVALUE
from timing import TempoMap

RPE_EASING_FUNCS: list[EasingFunction] = [
    LINEAR,  # 0
    LINEAR,  # 1
    EASING_FUNCTIONS['sine_out'],  # 2
    EASING_FUNCTIONS['sine_in'],  # 3
    EASING_FUNCTIONS['quad_out'],  # 4
    EASING_FUNCTIONS['quad_in'],  # 5
    EASING_FUNCTIONS['sine_inout'],  # 6
    EASING_FUNCTIONS['quad_inout'],  # 7
    EASING_FUNCTIONS['cubic_out'],  # 8
    EASING_FUNCTIONS['cubic_in'],  # 9
    EASING_FUNCTIONS['quart_out'],  # 10
    EASING_FUNCTIONS['quart_in'],  # 11
    EASING_FUNCTIONS['cubic_inout'],  # 12
    EASING_FUNCTIONS['quart_inout'],  # 13
    EASING_FUNCTIONS['quint_out'],  # 14
    EASING_FUNCTIONS['quint_in'],  # 15
    EASING_FUNCTIONS['expo_out'],  # 16
    EASING_FUNCTIONS['expo_in'],  # 17
    EASING_FUNCTIONS['circ_out'],  # 18
    EASING_FUNCTIONS['circ_in'],  # 19
    EASING_FUNCTIONS['back_out'],  # 20
    EASING_FUNCTIONS['back_in'],  # 21
    EASING_FUNCTIONS['circ_inout'],  # 22
    EASING_FUNCTIONS['back_inout'],  # 23
    EASING_FUNCTIONS['elastic_out'],  # 24
    EASING_FUNCTIONS['elastic_in'],  # 25
    EASING_FUNCTIONS['bounce_out'],  # 26
    EASING_FUNCTIONS['bounce_in'],  # 27
    EASING_FUNCTIONS['bounce_inout'],  # 28
    EASING_FUNCTIONS['elastic_inout'],  # 29
]


class RpeBeats(NamedTuple):
    add: float
    num: float
    deno: float

    def beats(self) -> float:
        if self.deno == 0:
            raise ValueError('RPE beat denominator cannot be zero')
        return self.add + self.num / self.deno


class RpeBPMInfoDict(TypedDict):
    bpm: float
    startTime: RpeBeats


class RpeMetaInfoDict(TypedDict):
    RPEVersion: int
    offset: int


class RpeBezierControl(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float


class RpeEventDict(TypedDict):
    bezier: int
    bezierPoints: RpeBezierControl
    easingLeft: float
    easingRight: float
    easingType: int
    start: float
    end: float
    startTime: RpeBeats
    endTime: RpeBeats
    linkgroup: int


class RpeEventLayerDict(TypedDict, total=False):
    moveXEvents: list[RpeEventDict]
    moveYEvents: list[RpeEventDict]
    rotateEvents: list[RpeEventDict]


class RpeExtendedEventsDict(TypedDict, total=False):
    scaleXEvents: list[RpeEventDict]
    scaleYEvents: list[RpeEventDict]
    inclineEvents: list[RpeEventDict]


RPE_NOTE_TYPES = [NoteType.UNKNOWN, NoteType.TAP, NoteType.HOLD, NoteType.FLICK, NoteType.DRAG]


class RpeNoteDict(TypedDict):
    type: int
    above: int
    startTime: RpeBeats
    endTime: RpeBeats
    positionX: float
    yOffset: float
    alpha: int
    size: float
    speed: float
    isFake: int
    visibleTime: float


class RpeYControlDict(TypedDict):
    easing: int
    x: float
    y: float


class RpeSkewControlDict(TypedDict):
    easing: int
    x: float
    skew: float


class RpePosControlDict(TypedDict):
    easing: int
    pos: float
    x: float


class RpeJudgeLineDict(TypedDict):
    Group: int
    Name: str
    bpmfactor: float
    eventLayers: list[RpeEventLayerDict]
    extended: RpeExtendedEventsDict
    father: int
    notes: list[RpeNoteDict]
    skewControl: list[RpeSkewControlDict]
    posControl: list[RpePosControlDict]
    yControl: list[RpeYControlDict]


class RpeChartDict(TypedDict):
    META: RpeMetaInfoDict
    BPMList: list[RpeBPMInfoDict]
    judgeLineGroup: list[str]
    judgeLineList: list[RpeJudgeLineDict]


def beats(value) -> float:
    result = RpeBeats(*value).beats() if isinstance(value, (list, tuple)) else float(value)
    if not math.isfinite(result):
        raise ValueError(f'invalid RPE beat: {value}')
    return result


def get_easing(event: dict) -> EasingFunction:
    if event.get('bezier', 0):
        points = event.get('bezierPoints', (0, 0, 1, 1))
        if len(points) != 4 or not all(math.isfinite(p) for p in points):
            raise ValueError(f'invalid bezier control points: {points}')
        x1, y1, x2, y2 = points
        return cubic_rev_bezier(max(0, min(1, x1)), y1, max(0, min(1, x2)), y2)
    kind = event.get('easingType', 1)
    easing = RPE_EASING_FUNCS[kind] if isinstance(kind, int) and 0 <= kind < len(RPE_EASING_FUNCS) else LINEAR
    left, right = event.get('easingLeft', 0.0), event.get('easingRight', 1.0)
    return easing if left == 0 and right == 1 else easing_with_range(easing, left, right)


def speed_easing(event: dict, version: int) -> EasingFunction:
    kind = event.get('easingType', 1)
    if version < 162:
        return LINEAR
    if kind == 0:
        return LVALUE
    if kind == 1:
        return LINEAR
    easing = get_easing(event)
    if version >= 170:
        return easing

    # RPE 1.6.2~1.6.9 使用缓动的导函数控制速度，1.7.0 恢复对速度直接缓动
    def derivative(t):
        # 在端点内侧取样
        left, right = max(1e-7, t - 1e-6), min(1 - 1e-7, t + 1e-6)
        return (easing(right) - easing(left)) / (right - left) if right > left else 0.0

    first, last = derivative(0.0), derivative(1.0)
    delta = last - first
    if not math.isfinite(delta) or abs(delta) < 1e-8:
        return LINEAR

    def result(t):
        return (derivative(t) - first) / delta

    result.integral = lambda a, b: (easing(b) - easing(a) - first * (b - a)) / delta
    return result


def interpolate_text(start: str, end: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    if '%P%' in start and '%P%' in end:
        first, last = start.replace('%P%', ''), end.replace('%P%', '')
        if t == 0:
            return first
        if t == 1:
            return last
        try:
            a, b = float(first), float(last)
        except ValueError:
            return first
        value = a + (b - a) * t
        return f'{value:.0f}' if a.is_integer() and b.is_integer() else f'{value:.3f}'
    start, end = start.replace('%P%', ''), end.replace('%P%', '')
    if not start:
        return end[: math.floor(len(end) * t + 0.5)]
    if not end:
        return start[: math.floor(len(start) * (1 - t) + 0.5)]
    if end.startswith(start):
        return end[: len(start) + math.floor((len(end) - len(start)) * t)]
    if start.startswith(end):
        return start[: len(end) + math.floor((len(start) - len(end)) * (1 - t) + 0.5)]
    return end if t == 1 else start


class RpeJudgeLine(JudgeLine):
    chart: 'RpeChart'

    def __init__(self, dic: RpeJudgeLineDict, chart: 'RpeChart') -> None:
        super().__init__()
        self.chart = chart
        self.bpm_factor = float(dic.get('bpmfactor', 1.0))
        if not math.isfinite(self.bpm_factor) or self.bpm_factor <= 0:
            raise ValueError(f'invalid bpmfactor: {self.bpm_factor}')
        self.father: RpeJudgeLine | None = None
        self.rotate_with_father = bool(dic.get('rotateWithFather', False))
        self._cached_time = None
        self._cached_transform = (0j, 0.0)
        self.control_scale = chart._CHART_HEIGHT / chart.height
        self.is_cover = dic.get('isCover', 1) == 1
        self.z_order = dic.get('zOrder', 0)
        self.texture = dic.get('Texture', 'line.png')
        self.attach_ui = dic.get('attachUI')
        self.anchor = tuple(dic.get('anchor', (0.5, 0.5)))
        if self.texture != 'line.png':
            self.color = BambooShoot((255, 255, 255))
        if self.attach_ui is not None:
            chart.warn('attachUI lines are not displayed in the simplified preview.')

        def control(name: str, key: str, default: float) -> Bamboo[float]:
            points = sorted(dic.get(name) or [], key=lambda e: e['x'])
            if not points:
                return BambooShoot(default)
            result = LivingBamboo[float]()
            for i, point in enumerate(points):
                # Control 的 easing 属于抵达该控制点的区间，和普通事件不同
                kind = points[min(i + 1, len(points) - 1)].get('easing', 1)
                result.cut(point['x'], point.get(key, default), get_easing({'easingType': kind}))
            return result

        self.pos_control = control('posControl', 'pos', 1.0)
        self.y_control = control('yControl', 'y', 1.0)
        self.alpha_control = control('alphaControl', 'alpha', 1.0)
        self.size_control = control('sizeControl', 'size', 1.0)
        self.skew_control = control('skewControl', 'skew', 0.0)
        if any(point.get('skew', 0) for point in dic.get('skewControl') or []):
            chart.warn('skewControl is preserved but not rendered; Phira does not define its behavior.')

        xs, ys, rotations, opacities, speeds, floors = [], [], [], [], [], []
        for layer in dic.get('eventLayers') or []:
            if not isinstance(layer, dict):
                continue
            if layer.get('moveXEvents'):
                xs.append(
                    self.convert_events(layer['moveXEvents'], convert=lambda x: x / chart._CHART_WIDTH * chart.width)
                )
            if layer.get('moveYEvents'):
                ys.append(
                    self.convert_events(layer['moveYEvents'], convert=lambda y: -y / chart._CHART_HEIGHT * chart.height)
                )
            if layer.get('rotateEvents'):
                rotations.append(self.convert_events(layer['rotateEvents'], convert=math.radians))
            if layer.get('alphaEvents'):
                opacities.append(self.convert_events(layer['alphaEvents'], convert=lambda a: a / 255))
            if layer.get('speedEvents'):
                speed = self.convert_events(
                    layer['speedEvents'], convert=lambda v: v * chart.height / (9 * 0.83175), is_speed=True
                )
                speeds.append(speed)
                floors.append(IntegratedBamboo(speed))
        self.local_position = TwinBamboo(BambooGrove(xs, 0.0), BambooGrove(ys, 0.0))
        self.local_angle = BambooGrove(rotations, 0.0)
        self.position = BambooFunc(lambda t: self.transform(t)[0])
        self.angle = BambooFunc(lambda t: self.transform(t)[1])
        self.opacity = BambooGrove(opacities, 0.0) if opacities else BambooShoot(1.0)
        self.speed = BambooGrove(speeds, 0.0)
        self.floor = BambooGrove(floors, 0.0)

        extended = dic.get('extended') or {}
        for name in ('paintEvents', 'gifEvents'):
            if extended.get(name):
                chart.warn(f'{name} is not rendered in the simplified preview.')
        if dic.get('isGif'):
            chart.warn('GIF line textures are displayed as still images in the simplified preview.')
        self.incline = self.convert_events(extended.get('inclineEvents'), convert=math.radians, extend_first=False)
        self.scale_x = self.convert_events(extended.get('scaleXEvents'), default=1.0, extend_first=False)
        self.scale_y = self.convert_events(extended.get('scaleYEvents'), default=1.0, extend_first=False)
        if extended.get('textEvents'):
            self.text = self.convert_events(
                extended['textEvents'], default='', interpolate=interpolate_text, extend_first=False
            )
            self.color = BambooShoot((255, 255, 255))
        if extended.get('colorEvents'):
            self.color = self.convert_events(
                extended['colorEvents'],
                default=(255, 255, 255),
                convert=tuple,
                interpolate=lambda a, b, t: tuple(x + (y - x) * t for x, y in zip(a, b)),
                extend_first=False,
            )

        for item in dic.get('notes') or []:
            kind = item.get('type', 1)
            if not isinstance(kind, int) or not 1 <= kind < len(RPE_NOTE_TYPES):
                raise ValueError(f'unknown RPE note type: {kind}')
            start = self.seconds(item['startTime'])
            end = self.seconds(item.get('endTime', item['startTime'])) if kind == 2 else start
            if end < start:
                raise ValueError(f'hold ends before it starts: {start}, {end}')
            for name in ('positionX', 'yOffset', 'alpha', 'size', 'speed'):
                if name in item and not math.isfinite(item[name]):
                    raise ValueError(f'invalid note {name}: {item[name]}')
            visible_time = item.get('visibleTime', math.inf)
            # 负 visibleTime 表示判定时间之后才显示
            if math.isnan(visible_time):
                raise ValueError(f'invalid note visibleTime: {visible_time}')
            tint = tuple(item.get('tint') or item.get('color') or (255, 255, 255))
            if len(tint) != 3 or not all(math.isfinite(v) for v in tint):
                raise ValueError(f'invalid note tint: {tint}')
            if item.get('judgeArea', 1) not in (None, 1):
                chart.warn('Custom judgeArea is not applied by the touch planners.')
            x = item.get('positionX', 0.0) / chart._CHART_WIDTH * chart.width
            note = Note(RPE_NOTE_TYPES[kind], start, end - start, complex(x))
            visual = VisualNote(
                note,
                position_x=x,
                y_offset=item.get('yOffset', 0.0) / chart._CHART_HEIGHT * chart.height,
                speed=item.get('speed', 1.0),
                above=item.get('above', 1) == 1,
                alpha=item.get('alpha', 255) / 255,
                size=item.get('size', 1.0),
                is_fake=bool(item.get('isFake', 0)),
                visible_time=visible_time,
                floor=self.floor @ start,
                end_floor=self.floor @ end,
                tint=tint,
            )
            # Control 和 incline 只有外观变了
            side = -1 if visual.above else 1
            visual.note = note._replace(offset=complex(x, visual.y_offset * visual.speed * side))
            self.visual_notes.append(visual)
            if not visual.is_fake:
                self.notes.append(visual.note)
        self.notes.sort(key=lambda n: n.seconds)
        self.visual_notes.sort(key=lambda n: n.note.seconds)

    def convert_events(
        self, events, default=0.0, convert=None, interpolate=None, extend_first=True, is_speed=False
    ) -> EventBamboo:
        convert = convert or (lambda v: v)
        result = EventBamboo(default, interpolate)
        cursor = -math.inf
        overlaps = 0
        for event in sorted(events or [], key=lambda e: beats(e['startTime'])):
            start, end = self.seconds(event['startTime']), self.seconds(event['endTime'])
            if end < start:
                raise ValueError(f'event ends before it starts: {start}, {end}')
            a, b = convert(event.get('start', default)), convert(event.get('end', default))
            if is_speed:
                if start < cursor:
                    overlaps += 1
                start, end = max(start, cursor), max(end, cursor)
                cursor = end
            if a == b or start == end:
                easing = LINEAR
            else:
                easing = speed_easing(event, self.chart.version) if is_speed else get_easing(event)
                kind = event.get('easingType', 1)
                if not event.get('bezier') and (not isinstance(kind, int) or not 0 <= kind < len(RPE_EASING_FUNCS)):
                    self.chart.warn(f'Unknown RPE easingType {kind}; using linear interpolation.')
            result.cut(start, end, a, b, easing)
        if overlaps:
            self.chart.warn(f'Clipped {overlaps} overlapping RPE speed events in an event layer.')
        if extend_first and not is_speed and result.events:
            result.default = result.events[0].start_value
        return result

    def seconds(self, value) -> float:
        return self.chart._beats_to_seconds(beats(value)) * self.bpm_factor

    def transform(self, seconds: float) -> tuple[Position, float]:
        if self._cached_time == seconds:
            return self._cached_transform
        pending = []
        line = self
        while line is not None and line._cached_time != seconds:
            pending.append(line)
            line = line.father
        for line in reversed(pending):
            position = line.local_position @ seconds
            angle = line.local_angle @ seconds
            if line.father is None:
                position += complex(self.chart.width, self.chart.height) / 2
            else:
                parent_pos, parent_angle = line.father._cached_transform
                position = parent_pos + cmath.exp(parent_angle * 1j) * position
                if line.rotate_with_father:
                    angle += parent_angle
            line._cached_time = seconds
            line._cached_transform = position, angle
        return self._cached_transform

    def pos(self, seconds: float, offset: Position) -> Position:
        position, angle = self.transform(seconds)
        return position + cmath.exp(angle * 1j) * offset

    def beat_duration(self, seconds: float) -> float:
        return self.chart.tempo.beat_duration(seconds / self.bpm_factor) * self.bpm_factor


class RpeChart(Chart):
    _CHART_WIDTH = 1350
    _CHART_HEIGHT = 900

    def __init__(self, dic: RpeChartDict, ratio: tuple[int, int]) -> None:
        super().__init__()
        self.width, self.height = ratio
        self.format = 'rpe'
        meta = dic.get('META') or {}
        self.offset = float(meta.get('offset', 0)) / 1000
        if not math.isfinite(self.offset):
            raise ValueError(f'invalid RPE offset: {self.offset}')
        version = meta.get('RPEVersion')
        try:
            self.version = 160 if version is None else int(version)
        except (ValueError, TypeError, OverflowError):
            self.version = 160
            self.warn(f'Invalid RPEVersion {version!r}; using legacy version 160.')
        self.tempo = TempoMap((beats(item['startTime']), item['bpm']) for item in dic['BPMList'])
        self.bpss = self.tempo.events
        self.lines = []
        for index, line in enumerate(dic['judgeLineList']):
            try:
                self.lines.append(RpeJudgeLine(line, self))
            except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
                raise ValueError(f'RPE line {index}: {error}') from error
        for index, (line, item) in enumerate(zip(self.lines, dic['judgeLineList'])):
            parent = item.get('father', -1)
            if parent is None or parent == -1:
                continue
            if not isinstance(parent, int) or not 0 <= parent < len(self.lines):
                raise ValueError(f'invalid parent {parent} on RPE line {index}')
            line.father = self.lines[parent]
        # 空线得保留，不然父线索引就坏了
        visited = set()
        for index, line in enumerate(self.lines):
            chain = set()
            while line is not None and line not in visited:
                if line in chain:
                    raise ValueError(f'cyclic RPE parent relation at line {index}')
                chain.add(line)
                line = line.father
            visited.update(chain)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def _beats_to_seconds(self, beats: float) -> float:
        return self.tempo.seconds(beats)


__all__ = ['RpeChart']
