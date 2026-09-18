import math
import cmath
import itertools
from typing import NamedTuple, TypeAlias, Iterable, Any
from collections import defaultdict
from enum import Enum

from shapely import (
    Polygon,
    MultiPolygon,
    LineString,
    Point,
    clip_by_rect,
    buffer,
    intersection,
    intersects,
    distance,
    centroid,
    difference,
    union_all,
)
from shapely.ops import nearest_points

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from basis import Chart, NoteType, Position, Vector
from .base import RawAnswerType, TouchAction, VirtualTouchEvent, ScreenUtil, AlgorithmConfigure

PointerID: TypeAlias = int
NoteID: TypeAlias = int

class DownNeed(Enum):
    MUST = 0
    NEVER = 1
    MAY = 2

class SemiNoteType(Enum):
    TAP = 0
    DRAG = 1
    FLICK_START = 2
    FLICK = 3
    FLICK_END = 4
    HOLD_START = 5
    HOLD = 6
    HOLD_END = 7

    @property
    def down_need(self) -> DownNeed:
        if self in (SemiNoteType.TAP, SemiNoteType.HOLD_START):
            return DownNeed.MUST
        if self in (
            SemiNoteType.DRAG, 
            SemiNoteType.HOLD, 
            SemiNoteType.HOLD_END, 
            SemiNoteType.FLICK, 
            SemiNoteType.FLICK_END
        ):
            return DownNeed.NEVER
        return DownNeed.MAY

class SemiNote(NamedTuple):
    type: SemiNoteType
    position: Position
    id: NoteID
    rotation: Vector

# 判定带半宽
# 规划时取最窄的一档再乘余量，规避别的note时取最宽的一档再放大
JUDGE_HALF_TAP = 0.106875
JUDGE_HALF_DRAG = 0.118125
JUDGE_MARGIN = 0.85
JUDGE_EXCLUDE = 1.15
# 指针要停够这么久才能够使用
DOWN_SETTLE = 50
DOWN_LATENCY = 34
# 同一时刻落在同一条判定带里的几个Flick，后面的滑动依次往后错开这么久，最多错开 FLICK_STAGGER_MAX
# 解决下Flick海的问题
FLICK_STAGGER = 30
FLICK_STAGGER_MAX = 60
# 滑动轨迹允许偏离自己判定中线的比例
FLICK_LATERAL = 0.5

class JudgeArea:
    __slots__ = ('center', 'rotation', 'poly', 'half_width')

    def __init__(
        self, center: Position, rotation: Vector, screen_w: float, screen_h: float,
        half: float = JUDGE_HALF_TAP, scale: float = JUDGE_MARGIN
    ) -> None:
        self.center = center
        self.rotation = rotation
        self.half_width = screen_w * half * scale
        perp = rotation * 1j
        limit = math.hypot(screen_w, screen_h) + abs(center)
        d_rot = rotation * self.half_width
        d_perp = perp * limit
        c1 = center + d_rot + d_perp
        c2 = center + d_rot - d_perp
        c3 = center - d_rot - d_perp
        c4 = center - d_rot + d_perp
        self.poly = Polygon([
            (c1.real, c1.imag),
            (c2.real, c2.imag),
            (c3.real, c3.imag),
            (c4.real, c4.imag)
        ])

    def get_valid_poly(self, screen_poly: Polygon, pause_poly: Polygon) -> Polygon:
        inter = self.poly.intersection(screen_poly)
        if inter.is_empty:
            return Polygon()
        valid = inter.difference(pause_poly)
        return valid if not valid.is_empty else Polygon()

    def is_valid_zone(self, valid_poly: Polygon, screen_w: float) -> bool:
        if valid_poly.is_empty:
            return False
        return valid_poly.area > (screen_w * 1e-4) ** 2

class PointerRecord(NamedTuple):
    id: PointerID
    position: Position
    timestamp: int
    line_ref: Any = None
    note_offset: float = 0.0
    note_type: SemiNoteType | None = None

class PointerManager:
    __slots__ = ('occupied', 'idle', 'unused', 'last_active_ts', 'down_ts', 'waiting_liftup', 'current_ts', 'console', 'noway')

    def __init__(self, pointer_ids: Iterable[PointerID], console: Console, noway: bool = False) -> None:
        self.occupied: dict[NoteID, PointerRecord] = {}
        self.idle: set[PointerID] = set(pointer_ids)
        self.unused: dict[PointerID, PointerRecord] = {}
        self.last_active_ts: dict[PointerID, int] = {pid: 0 for pid in pointer_ids}
        self.down_ts: dict[PointerID, int] = {}
        self.waiting_liftup: list[tuple[PointerRecord, int]] = []
        self.current_ts: int = 0
        self.console: Console = console
        self.noway: bool = noway

    def alloc(self, note: SemiNote, new: bool = True, line_ref: Any = None, note_offset: float = 0.0) -> tuple[PointerID | None, bool]:
        nid = note.id
        if nid in self.occupied:
            ptr = self.occupied[nid]
            cur_line = line_ref if line_ref is not None else ptr.line_ref
            cur_offset = note_offset if line_ref is not None else ptr.note_offset
            self.occupied[nid] = PointerRecord(ptr.id, note.position, self.current_ts, cur_line, cur_offset, note.type)
            self.last_active_ts[ptr.id] = self.current_ts
            return ptr.id, False
        if not new and self.unused:
            valid_unused = {
                pid: ptr for pid, ptr in self.unused.items()
                if ptr.timestamp < self.current_ts and not self.settling(pid)
            }
            if valid_unused:
                ptr = min(valid_unused.values(), key=lambda p: abs(note.position - p.position))
                del self.unused[ptr.id]
                self.occupied[nid] = PointerRecord(ptr.id, note.position, self.current_ts, line_ref, note_offset, note.type)
                self.last_active_ts[ptr.id] = self.current_ts
                return ptr.id, False
        if self.idle:
            pid = self.idle.pop()
            self.occupied[nid] = PointerRecord(pid, note.position, self.current_ts, line_ref, note_offset, note.type)
            self.last_active_ts[pid] = self.current_ts
            self.down_ts[pid] = self.current_ts
            return pid, True
        if self.unused:
            # 优先抬起闲置够久的指针
            settled = [
                p for p in self.unused.values()
                if min(self.current_ts - self.down_ts.get(p.id, 0), self.current_ts - p.timestamp) >= 2 * DOWN_SETTLE
            ]
            ptr = min(settled or self.unused.values(), key=lambda p: abs(note.position - p.position))
            if self.current_ts > ptr.timestamp + 1:
                up_ts = (ptr.timestamp + self.current_ts) // 2
            else:
                ptr = min(self.unused.values(), key=lambda p: p.timestamp)
                if self.current_ts > ptr.timestamp + 1:
                    up_ts = (ptr.timestamp + self.current_ts) // 2
                else:
                    up_ts = self.current_ts - 1
            del self.unused[ptr.id]
            up_ts = max(0, up_ts)
            self.waiting_liftup.append((ptr, up_ts))
            self.occupied[nid] = PointerRecord(ptr.id, note.position, self.current_ts, line_ref, note_offset, note.type)
            self.last_active_ts[ptr.id] = self.current_ts
            self.down_ts[ptr.id] = self.current_ts
            return ptr.id, True
        
        if self.noway:
            self.console.print(f"[red]Note({note}) @ {self.current_ts} 规划失败[/red]")
            return None, False
        raise RuntimeError(f'no free pointers @ {self.current_ts}')

    def settling(self, pid: PointerID) -> bool:
        return self.current_ts - self.down_ts.get(pid, -DOWN_SETTLE) < DOWN_SETTLE

    def free(self, note: SemiNote) -> None:
        if note.id in self.occupied:
            ptr = self.occupied.pop(note.id)
            is_still_shared = any(active_ptr.id == ptr.id for active_ptr in self.occupied.values())
            if not is_still_shared:
                # 没有note在占用这个指针了，可以丢进unused里去
                self.unused[ptr.id] = PointerRecord(
                    id=ptr.id, 
                    position=ptr.position, 
                    timestamp=self.current_ts,
                    line_ref=ptr.line_ref,
                    note_offset=ptr.note_offset,
                    note_type=note.type
                )

    def recycle(self) -> Iterable[tuple[PointerRecord, int]]:
        for ptr, up_ts in self.waiting_liftup:
            yield ptr, up_ts
        self.waiting_liftup.clear()

    def finish(self) -> Iterable[tuple[PointerRecord, int]]:
        # pointers.current_ts 是整个谱面的最后一帧的时间戳
        # 不能使用pointer最后活跃的时间，否则谱面末尾如果是Hold就提前松手了，比如李斯特IN
        for ptr in itertools.chain(self.unused.values(), self.occupied.values()):
            yield ptr, self.current_ts + 100

class SweepTarget:
    __slots__ = ('note', 'poly', 'is_swept')
    def __init__(self, note: SemiNote, poly: Polygon) -> None:
        self.note = note
        self.poly = poly
        self.is_swept = False

def solve(chart: Chart, config: AlgorithmConfigure, console: Console) -> tuple[ScreenUtil, RawAnswerType]:
    screen = ScreenUtil(chart.width, chart.height)
    flick_start = config['algo4_flick_start']
    flick_end = config['algo4_flick_end']
    sample_delay = config['algo4_sample_delay']
    noway = config['algo4_continue_when_failed']
    flick_dir = 1j if config['algo4_flick_direction'] == 0 else 1
    flick_duration = flick_end - flick_start
    padding_x = screen.width * 0.05
    padding_y = screen.height * 0.05
    screen_poly = Polygon([
        (padding_x, padding_y), 
        (screen.width - padding_x, padding_y), 
        (screen.width - padding_x, screen.height - padding_y), 
        (padding_x, screen.height - padding_y)
    ])
    # 扣掉暂停键
    pause_w = screen.width * 0.10
    pause_h = screen.height * 0.10
    pause_poly = MultiPolygon([
        Polygon([(0, 0), (pause_w, 0), (pause_w, pause_h), (0, pause_h)]),
        Polygon([
            (screen.width - pause_w, 0),
            (screen.width, 0),
            (screen.width, pause_h),
            (screen.width - pause_w, pause_h)
        ]),
    ])
    hold_keep = screen.width * JUDGE_HALF_TAP * JUDGE_MARGIN * 0.7
    zone_cache: dict[tuple, Polygon] = {}

    def judge_zone(pos: Position, rot: Vector, half: float = JUDGE_HALF_TAP, scale: float = JUDGE_MARGIN) -> Polygon:
        # 判定线不动时同一个 note 的判定区会被反复计算，这里缓存一下
        key = (round(pos.real, 6), round(pos.imag, 6), round(rot.real, 6), round(rot.imag, 6), half, scale)
        poly = zone_cache.get(key)
        if poly is None:
            poly = JudgeArea(pos, rot, screen.width, screen.height, half, scale).get_valid_poly(screen_poly, pause_poly)
            zone_cache[key] = poly
        return poly

    def zone_ok(poly: Polygon) -> bool:
        return not poly.is_empty and poly.area > (screen.width * 1e-4) ** 2

    down_zone_cache: dict[tuple, tuple[Polygon, Polygon]] = {}

    def down_zone(
        line_obj, t: float, offset: Position, window: int = DOWN_LATENCY,
        half: float = JUDGE_HALF_TAP, scale: float = JUDGE_MARGIN
    ) -> tuple[Polygon, Polygon]:
        # DOWN落点都要在判定带里，就取各时刻判定带的交集吧
        # 返回并集用来规避别的note把这段时间扫过的范围都排除掉
        offset = complex(offset)
        key = (id(line_obj), round(t, 4), round(offset.real, 4), round(offset.imag, 4), window, half, scale)
        zones = down_zone_cache.get(key)
        if zones is None:
            inter = union = None
            for k in range(4):
                t_i = t + window * k / 3 / 1000.0
                rot_i = cmath.exp((line_obj.angle @ t_i) * 1j)
                poly_i = judge_zone(line_obj.position @ t_i + rot_i * offset, rot_i, half, scale)
                inter = poly_i if inter is None else intersection(inter, poly_i)
                union = poly_i if union is None else union.union(poly_i)
            zones = (inter, union)
            down_zone_cache[key] = zones
        return zones
    hold_ranges: list[tuple[int, int, int]] = []
    flick_ranges: list[tuple[int, int]] = []
    max_concurrent_holds = 0
    max_frame_must = 0
    max_frame_may = 0
    note_id_to_line: dict[NoteID, Any] = {}
    note_id_to_offset: dict[NoteID, float] = {}
    sweep_registry: defaultdict[int, list[SweepTarget]] = defaultdict(list)
    deferred_flicks: list[dict] = []

    def pick_point(poly: Polygon, near: Position) -> Position:
        pt = Point(near.real, near.imag)
        if poly.contains(pt):
            return near
        inner = poly.buffer(-screen.width * 0.01)
        if not zone_ok(inner):
            inner = poly
        geom = nearest_points(inner, pt)[0]
        return Position(geom.x, geom.y)

    def down_window(note_type: NoteType | SemiNoteType) -> int:
        if note_type in (NoteType.HOLD, SemiNoteType.HOLD_START):
            return DOWN_SETTLE
        return DOWN_LATENCY

    def find_visible_pos(base_sec, base_pos, base_rot, note_offset, line_obj, window: int = DOWN_LATENCY) -> tuple[float, Position, Vector, float, bool]:
        valid_touch_zone, _ = down_zone(line_obj, base_sec, note_offset, window)
        if zone_ok(valid_touch_zone):
            orig_point = Point(base_pos.real, base_pos.imag)
            closest_geom = nearest_points(valid_touch_zone, orig_point)[0]
            closest_pos = Position(closest_geom.x, closest_geom.y)
            line_center = line_obj.position @ base_sec
            delta = closest_pos - line_center
            new_offset = (delta * base_rot.conjugate()).real
            adjusted = abs(closest_pos - base_pos) > 1e-5
            if adjusted:
                console.print(f"[yellow]判定区域微调：note @ {base_sec} of (pos={base_pos},rot={base_rot}) => (pos={closest_pos})[/yellow]")
            return base_sec, closest_pos, base_rot, new_offset, adjusted

        for dt in range(1, 16):
            for sign in (-1, 1):
                new_time = base_sec + (dt / 1000.0) * sign
                new_lp = line_obj.position @ new_time
                new_alpha = line_obj.angle @ new_time
                new_rot = cmath.exp(new_alpha * 1j)
                new_note_pos = new_lp + new_rot * note_offset
                new_valid_zone, _ = down_zone(line_obj, new_time, note_offset, window)
                if zone_ok(new_valid_zone):
                    orig_point_at_t = Point(new_note_pos.real, new_note_pos.imag)
                    closest_geom = nearest_points(new_valid_zone, orig_point_at_t)[0]
                    closest_pos = Position(closest_geom.x, closest_geom.y)
                    delta = closest_pos - new_lp
                    new_offset = (delta * new_rot.conjugate()).real
                    console.print(f"[yellow]判定时间微调：note @ {base_sec} of (pos={base_pos},rot={base_rot})=> note @ {new_time} of (pos={closest_pos},rot={new_rot})[/yellow]")
                    return new_time, closest_pos, new_rot, new_offset, True
        
        console.print(f"[red]判定微调失败：note @ {base_sec} of (pos={base_pos},rot={base_rot})[/red]")
        return base_sec, base_pos, base_rot, note_offset, False

    def flick_pos(pos: Position, offset_ms: int, rot: Vector, f_dir: Vector, start_off: int, shift: float = 0.0) -> Position:
        rate = 1 - 2 * (offset_ms - start_off) / flick_duration
        return pos + rot * f_dir * (screen.flick_radius * rate + shift)

    def fit_flick(pos: Position, rot: Vector, f_dir: Vector, band: Polygon) -> float | None:
        # 轨迹在屏幕外时，沿滑动方向整体平移下，平移量越小越好
        for step in range(0, 16):
            for sign in ((1,) if step == 0 else (-1, 1)):
                shift = sign * step * screen.flick_radius * 0.1
                if all(
                    band.intersects(Point(p.real, p.imag))
                    for p in (flick_pos(pos, off, rot, f_dir, flick_start, shift) for off in flick_eval_offsets)
                ):
                    return shift
        return None
    
    frames: defaultdict[int, list[SemiNote]] = defaultdict(list)
    dense_frame_sizes: defaultdict[int, int] = defaultdict(int)
    current_note_id = 0
    flick_eval_offsets = [flick_start] + list(range(flick_start + 1, flick_end, sample_delay))
    if flick_eval_offsets[-1] != flick_end:
        flick_eval_offsets.append(flick_end)
    
    total_notes = sum(len(line.notes) for line in chart.lines)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task1 = progress.add_task("统计帧...", total=total_notes)
        
        for line in chart.lines:
            for note in line.notes:
                alpha = line.angle @ note.seconds
                rotation: Vector = cmath.exp(alpha * 1j)
                line_pos = line.position @ note.seconds
                note_pos = line_pos + rotation * note.offset
                adj_time, adj_pos, adj_rot, adj_offset, adjusted = find_visible_pos(
                    note.seconds, note_pos, rotation, note.offset, line, down_window(note.type)
                )
                ts = round(adj_time * 1000)
                note_id_to_line[current_note_id] = line
                note_id_to_offset[current_note_id] = adj_offset
                match note.type:
                    case NoteType.TAP:
                        frames[ts].append(SemiNote(SemiNoteType.TAP, adj_pos, current_note_id, adj_rot))
                        dense_frame_sizes[ts] += 1
                    case NoteType.DRAG:
                        poly = judge_zone(adj_pos, adj_rot, JUDGE_HALF_DRAG)
                        sn = SemiNote(SemiNoteType.DRAG, adj_pos, current_note_id, adj_rot)
                        sweep_registry[ts].append(SweepTarget(sn, poly))
                    case NoteType.FLICK:
                        deferred_flicks.append({
                            'ts': ts, 'pos': adj_pos, 'rot': adj_rot, 'id': current_note_id, 'line': line
                        })
                    case NoteType.HOLD:
                        hold_ms = math.ceil(note.hold * 1000)
                        frames[ts].append(SemiNote(SemiNoteType.HOLD_START, adj_pos, current_note_id, adj_rot))
                        dense_frame_sizes[ts] += 1
                        p_touch = adj_pos
                        for off in range(1, hold_ms, sample_delay):
                            t = ((ts + off) // sample_delay) * sample_delay
                            t = max(ts, min(t, ts + hold_ms)) / 1000.0
                            ang = line.angle @ t
                            rot_t = cmath.exp(ang * 1j)
                            pos = line.pos(t, adj_offset)
                            valid_zone_t = judge_zone(pos, rot_t)
                            p_touch = pos
                            if not valid_zone_t.is_empty:
                                orig_point = Point(pos.real, pos.imag)
                                p_touch = Position(nearest_points(valid_zone_t, orig_point)[0].x, nearest_points(valid_zone_t, orig_point)[0].y)
                            sn = SemiNote(SemiNoteType.HOLD, p_touch, current_note_id, rot_t)
                            sweep_registry[int(round(t * 1000))].append(SweepTarget(sn, valid_zone_t))
                        t2 = (ts + hold_ms) / 1000.0
                        rot2 = cmath.exp((line.angle @ t2) * 1j)
                        end_pos = p_touch
                        end_pos_raw = line.pos(t2, adj_offset)
                        valid_zone_end = judge_zone(end_pos_raw, rot2)
                        if not valid_zone_end.is_empty:
                            orig_point_end = Point(end_pos_raw.real, end_pos_raw.imag)
                            end_pos = Position(nearest_points(valid_zone_end, orig_point_end)[0].x, nearest_points(valid_zone_end, orig_point_end)[0].y)
                        frames[ts + hold_ms].append(SemiNote(SemiNoteType.HOLD_END, end_pos, current_note_id, rot2))
                        dense_frame_sizes[ts + hold_ms] += 1

                current_note_id += 1
                progress.advance(task1, 1)
        
        total_flicks = len(deferred_flicks)
        task2 = progress.add_task("规划滑动轨迹...", total=total_flicks)
        
        deferred_flicks.sort(key=lambda f: f['ts'])
        placed_flicks: list[tuple[int, int, Position, Vector]] = []
        for f_info in deferred_flicks:
            base_ts, base_pos, base_rot = f_info['ts'], f_info['pos'], f_info['rot']
            nid, line = f_info['id'], f_info['line']
            # 同时的flick依次往后错开
            orig_ts = base_ts
            while True:
                clash = [
                    ts_p for orig_p, ts_p, pos_p, rot_p in placed_flicks
                    if abs(orig_p - orig_ts) <= 2 and abs(ts_p - base_ts) < FLICK_STAGGER and (
                        judge_zone(pos_p, rot_p, JUDGE_HALF_DRAG, JUDGE_EXCLUDE).intersects(Point(base_pos.real, base_pos.imag))
                        or judge_zone(base_pos, base_rot, JUDGE_HALF_DRAG, JUDGE_EXCLUDE).intersects(Point(pos_p.real, pos_p.imag))
                    )
                ]
                if not clash or max(clash) + FLICK_STAGGER - orig_ts > FLICK_STAGGER_MAX:
                    break
                base_ts = max(clash) + FLICK_STAGGER
            if base_ts != orig_ts:
                t_sec = base_ts / 1000.0
                alpha = line.angle @ t_sec
                rot = cmath.exp(alpha * 1j)
                note_pos = line.position @ t_sec + rot * note_id_to_offset[nid]
                adj_time, base_pos, base_rot, adj_offset, adjusted = find_visible_pos(t_sec, note_pos, rot, note_id_to_offset[nid], line)
                base_ts = round(adj_time * 1000)
                note_id_to_offset[nid] = adj_offset
                console.print(f"[cyan]错开滑动 @ {orig_ts} => {base_ts} (pos={base_pos})")
            placed_flicks.append((orig_ts, base_ts, base_pos, base_rot))
            candidates_targets = []
            # 这里整块逻辑都不能追踪flick的偏转，必须以判定时间为准
            # 不然有的绑线flick判定时间后就飞走了，计算出来的flick_pos就不是直线了
            for off in flick_eval_offsets:
                for target in sweep_registry.get(base_ts + off, []):
                    if not target.is_swept:
                        candidates_targets.append((base_ts + off, target, off))
            candidate_dirs = [flick_dir, -flick_dir]
            for tick_ts, target, off in candidates_targets:
                rate = 1 - 2 * (off - flick_start) / flick_duration
                if abs(rate) < 1e-3: 
                    continue
                vec = (target.note.position - base_pos) / (base_rot * screen.flick_radius * rate)
                if abs(vec) > 0:
                    candidate_dirs.append(vec / abs(vec))
            best_dir = flick_dir
            best_shift = 0.0
            max_swept = -1
            best_swept_targets = []
            flick_band = judge_zone(base_pos, base_rot, JUDGE_HALF_DRAG, JUDGE_MARGIN * FLICK_LATERAL)
            for c_dir in candidate_dirs:
                shift = fit_flick(base_pos, base_rot, c_dir, flick_band)
                if shift is None:
                    continue
                current_swept = []
                for off in flick_eval_offsets:
                    tick_ts = base_ts + off
                    p_flick = flick_pos(base_pos, off, base_rot, c_dir, flick_start, shift)
                    test_point = Point(p_flick.real, p_flick.imag)
                    for target in sweep_registry.get(tick_ts, []):
                        if not target.is_swept and target.poly.intersects(test_point):
                            current_swept.append(target)
                if len(current_swept) > max_swept:
                    max_swept = len(current_swept)
                    best_dir = c_dir
                    best_shift = shift
                    best_swept_targets = current_swept
            if max_swept == -1:
                console.print(f"[red]滑动轨迹出屏：note @ {base_ts} (pos={base_pos})[/red]")
            for target in best_swept_targets:
                target.is_swept = True
            for off in flick_eval_offsets:
                tick_ts = base_ts + off
                p_flick = flick_pos(base_pos, off, base_rot, best_dir, flick_start, best_shift)
                if off == flick_start:
                    frames[tick_ts].append(SemiNote(SemiNoteType.FLICK_START, p_flick, nid, base_rot))
                elif off == flick_end:
                    frames[tick_ts].append(SemiNote(SemiNoteType.FLICK_END, p_flick, nid, base_rot))
                else:
                    frames[tick_ts].append(SemiNote(SemiNoteType.FLICK, p_flick, nid, base_rot))
                dense_frame_sizes[tick_ts] += 1
            progress.advance(task2, 1)
        
        task3 = progress.add_task("平移超载帧...", total=len(frames))
        for ts in sorted(list(frames.keys())):
            while True:
                must_may_notes = [
                    n for n in frames[ts] 
                    if n.type in (SemiNoteType.TAP, SemiNoteType.HOLD_START, SemiNoteType.FLICK_START)
                ]
                if len(frames[ts]) <= 10 or not must_may_notes:
                    break
                note_to_shift = must_may_notes[0]
                nid = note_to_shift.id
                line = note_id_to_line.get(nid)
                orig_offset = note_id_to_offset.get(nid, 0.0)
                best_target_ts = None
                min_density = 999999
                # 优先向前找，4ms一间隔
                for dt in range(1, 11):
                    for sign in (-4, 4): 
                        target_ts = ts + dt * sign
                        if target_ts < 0:
                            continue
                        density = len(frames[target_ts])
                        if density < min_density and density < 10:
                            min_density = density
                            best_target_ts = target_ts
                    if min_density < 10 - 2:
                        break
                if best_target_ts is not None:
                    frames[ts].remove(note_to_shift)
                    dense_frame_sizes[ts] -= 1
                    t_sec = best_target_ts / 1000.0
                    alpha = line.angle @ t_sec
                    rot = cmath.exp(alpha * 1j)
                    line_pos = line.position @ t_sec
                    note_pos = line_pos + rot * orig_offset
                    adj_time, adj_pos, adj_rot, adj_offset, adjusted = find_visible_pos(
                        t_sec, note_pos, rot, orig_offset, line, down_window(note_to_shift.type)
                    )
                    final_ts = round(adj_time * 1000)
                    shifted_note = SemiNote(note_to_shift.type, adj_pos, nid, adj_rot)
                    frames[final_ts].append(shifted_note)
                    dense_frame_sizes[final_ts] += 1
                    note_id_to_offset[nid] = adj_offset
                    console.print(f"[cyan]偏移note @ {ts} ({note_to_shift}) => note @ {final_ts} ({shifted_note})")
                else:
                    break
            progress.advance(task3, 1)
        
        for ts, targets in sweep_registry.items():
            for target in targets:
                frames[ts].append(target.note)
                dense_frame_sizes[ts] += 1
        
        for line in chart.lines:
            for note in line.notes:
                start_ms = round(note.seconds * 1000)
                if note.type == NoteType.HOLD:
                    hold_ranges.append((start_ms, start_ms + math.ceil(note.hold * 1000), -1))
                elif note.type == NoteType.FLICK:
                    flick_ranges.append((start_ms + flick_start, start_ms + flick_end))

        ranges = hold_ranges + [(s, e, -1) for s, e in flick_ranges]
        if ranges:
            timestamps = sorted(set(s for r in ranges for s in r[:2]))
            for ts in timestamps:
                active = sum(1 for s, e, _ in ranges if s <= ts < e)
                max_concurrent_holds = max(max_concurrent_holds, active)

        for frame in frames.values():
            must = sum(1 for n in frame if n.type.down_need == DownNeed.MUST)
            may = sum(1 for n in frame if n.type.down_need == DownNeed.MAY)
            max_frame_must = max(max_frame_must, must)
            max_frame_may = max(max_frame_may, may)

        pointers_count = max_concurrent_holds + max_frame_must + max_frame_may
        max_dense_frame = max(dense_frame_sizes.values()) if dense_frame_sizes else 0
        pointers_count = max(pointers_count, max_dense_frame)
        pointers_count = min(10, pointers_count + 1)
        console.print(f'统计完毕，当前谱面共计{len(frames)}帧，最多需要{pointers_count}押')
        pointers = PointerManager(range(1000, 1000 + pointers_count), console, noway=noway)
        sorted_frames = sorted(frames.items())
        total_frames = len(sorted_frames)
        task4 = progress.add_task("规划触控事件...", total=total_frames)

        result: defaultdict[int, list[VirtualTouchEvent]] = defaultdict(list)
        pointer_pos: dict[PointerID, Position] = {}
        for timestamp, frame in sorted_frames:
            to_free: list[SemiNote] = []
            must_notes: list[SemiNote] = []
            may_notes: list[SemiNote] = []
            active_never: list[SemiNote] = []
            passive_notes: list[SemiNote] = []
            active_physical_touches: dict[PointerID, Position] = {}
            confirmed_pointers: dict[PointerID, Position] = {}
            
            pointers.current_ts = timestamp
            
            t_sec = timestamp / 1000.0
            for record in itertools.chain(pointers.occupied.values(), pointers.unused.values()):
                active_physical_touches[record.id] = pointer_pos.get(record.id, record.position)
            
            current_touches = active_physical_touches.copy()
            for note in frame:
                if note.type in (SemiNoteType.TAP, SemiNoteType.HOLD_START):
                    must_notes.append(note)
                elif note.type == SemiNoteType.FLICK_START:
                    may_notes.append(note)
                elif note.type in (SemiNoteType.FLICK, SemiNoteType.FLICK_END):
                    active_never.append(note)
                elif note.type in (SemiNoteType.HOLD, SemiNoteType.DRAG, SemiNoteType.HOLD_END):
                    passive_notes.append(note)
            
            active_polys = []
            exclude_polys = []
            for n in must_notes:
                line_ref = note_id_to_line.get(n.id)
                offset_val = note_id_to_offset.get(n.id, 0.0)
                window = down_window(n.type)
                inter, _ = down_zone(line_ref, t_sec, offset_val, window)
                if not zone_ok(inter):
                    inter = judge_zone(n.position, n.rotation)
                active_polys.append(inter)
                exclude_polys.append(down_zone(line_ref, t_sec, offset_val, window, JUDGE_HALF_DRAG, JUDGE_EXCLUDE)[1])
            must_targets = [n.position for n in must_notes]
            if len(must_notes) > 1:
                groups: list[list[int]] = [[i] for i in range(len(must_notes))]
                zones: list[Polygon] = []
                while True:
                    merged = False
                    zones = []
                    for gi, members in enumerate(groups):
                        own = active_polys[members[0]]
                        for m in members[1:]:
                            own = intersection(own, active_polys[m])
                        rest = [j for j in range(len(must_notes)) if j not in members]
                        free = difference(own, union_all([exclude_polys[j] for j in rest]))
                        if not zone_ok(free):
                            free = difference(own, union_all([active_polys[j] for j in rest])).buffer(-screen.width * 0.02)
                        if zone_ok(free):
                            zones.append(free)
                            continue
                        # 和判定带重叠最多的那组合并
                        best = None
                        for gj, others in enumerate(groups):
                            if gj == gi:
                                continue
                            cand = own
                            for o in others:
                                cand = intersection(cand, active_polys[o])
                            if zone_ok(cand) and (best is None or cand.area > best[1].area):
                                best = (gj, cand)
                        if best is None:
                            zones.append(own)
                            continue
                        groups[gi] = members + groups[best[0]]
                        del groups[best[0]]
                        merged = True
                        break
                    if not merged:
                        break
                for members, zone in zip(groups, zones):
                    target = pick_point(zone, must_notes[members[0]].position)
                    pt = Point(target.real, target.imag)
                    clash = [j for j in range(len(must_notes)) if j not in members and active_polys[j].intersects(pt)]
                    if clash:
                        console.print(f"[red]多押重叠调整失败：timestamp @ {timestamp}: note(pos={target}) 落在 {len(clash)} 个别的 note 判定带里[/red]")
                    for i in members:
                        if abs(target - must_notes[i].position) > 1e-5:
                            console.print(f"[yellow]多押重叠调整：timestamp @ {timestamp}: note(pos={must_notes[i].position}) => note(pos={target})[/yellow]")
                        must_targets[i] = target
            else:
                for i, note in enumerate(must_notes):
                    must_targets[i] = pick_point(active_polys[i], note.position)
            
            # 刚DOWN的指针不能MOVE
            downed_pids: set[PointerID] = {pid for pid in pointers.down_ts if pointers.settling(pid)}
            for note, target in zip(must_notes, must_targets):
                line_ref = note_id_to_line.get(note.id)
                offset_val = note_id_to_offset.get(note.id, 0.0)
                if note.id in pointers.occupied:
                    pointers.free(note)
                pid, is_down = pointers.alloc(
                    SemiNote(note.type, target, note.id, note.rotation),
                    line_ref=line_ref,
                    note_offset=offset_val
                )
                if pid is None:
                    continue
                result[timestamp].append(VirtualTouchEvent(target, TouchAction.DOWN, pid))
                if note.type == SemiNoteType.TAP:
                    to_free.append(note)
                confirmed_pointers[pid] = target
                downed_pids.add(pid)
            current_touches.update(confirmed_pointers)
            flicking_pids = {
                r.id for r in pointers.occupied.values()
                if r.note_type in (SemiNoteType.FLICK_START, SemiNoteType.FLICK, SemiNoteType.FLICK_END)
            }
            for note in may_notes:
                line_ref = note_id_to_line.get(note.id)
                offset_val = note_id_to_offset.get(note.id, 0.0)
                poly_n = judge_zone(note.position, note.rotation, JUDGE_HALF_DRAG)
                candidates = []
                covering_pid = None
                covering_pos = None
                for pid, p_touch in current_touches.items():
                    if pid in flicking_pids or pid in downed_pids:
                        continue
                    if poly_n.intersects(Point(p_touch.real, p_touch.imag)):
                        dist = abs(p_touch - note.position)
                        candidates.append((dist, pid, p_touch))
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    covering_pid = candidates[0][1]
                    covering_pos = candidates[0][2]
                if covering_pid is not None:
                    if covering_pid in pointers.unused:
                        del pointers.unused[covering_pid]
                    pointers.occupied[note.id] = PointerRecord(covering_pid, note.position, timestamp, line_ref, offset_val, note.type)
                    result[timestamp].append(VirtualTouchEvent(note.position, TouchAction.MOVE, covering_pid))
                    confirmed_pointers[covering_pid] = note.position
                    current_touches[covering_pid] = note.position
                    flicking_pids.add(covering_pid)
                else:
                    pid, is_down = pointers.alloc(note, new=False, line_ref=line_ref, note_offset=offset_val)
                    if pid is None:
                        continue
                    act = TouchAction.DOWN if is_down else TouchAction.MOVE
                    result[timestamp].append(VirtualTouchEvent(note.position, act, pid))
                    confirmed_pointers[pid] = note.position
                    current_touches[pid] = note.position
                    flicking_pids.add(pid)
            for note in active_never:
                line_ref = note_id_to_line.get(note.id)
                offset_val = note_id_to_offset.get(note.id, 0.0)
                if note.type in (SemiNoteType.FLICK, SemiNoteType.FLICK_END):
                    pid, _ = pointers.alloc(note, line_ref=line_ref, note_offset=offset_val)
                    if pid is None:
                        continue
                    result[timestamp].append(VirtualTouchEvent(note.position, TouchAction.MOVE, pid))
                    if note.type == SemiNoteType.FLICK_END:
                        to_free.append(note)
                    confirmed_pointers[pid] = note.position
            current_touches = active_physical_touches.copy()
            current_touches.update(confirmed_pointers)
            hold_notes = [n for n in passive_notes if n.type in (SemiNoteType.HOLD, SemiNoteType.HOLD_END)]
            drag_notes = [n for n in passive_notes if n.type == SemiNoteType.DRAG]

            def bind_holds(pid: PointerID, pos: Position, notes: list[SemiNote]) -> None:
                for n in notes:
                    old = pointers.occupied.get(n.id)
                    if old is not None and old.id != pid:
                        pointers.free(n)
                    if pid in pointers.unused:
                        del pointers.unused[pid]
                    pointers.occupied[n.id] = PointerRecord(
                        pid, pos, timestamp, note_id_to_line.get(n.id), note_id_to_offset.get(n.id, 0.0), n.type
                    )
                    if n.type == SemiNoteType.HOLD_END:
                        to_free.append(n)
                pointers.last_active_ts[pid] = timestamp

            # 一帧内所有Hold一起做覆盖
            # 否则像 RetributionSP，每条 Hold 都要一个指针就完蛋了
            if hold_notes:
                hold_polys = {n.id: judge_zone(n.position, n.rotation) for n in hold_notes}
                pending = [n for n in hold_notes if zone_ok(hold_polys[n.id])]
                groups: list[tuple[list[SemiNote], Polygon, PointerID | None]] = []
                for pid in downed_pids:
                    p_touch = current_touches.get(pid)
                    if p_touch is None:
                        continue
                    pt = Point(p_touch.real, p_touch.imag)
                    anchored = [n for n in pending if hold_polys[n.id].intersects(pt)]
                    if anchored:
                        groups.append((anchored, hold_polys[anchored[0].id], pid))
                        pending = [n for n in pending if n not in anchored]
                while pending:
                    best_group: list[SemiNote] = []
                    best_inter = None
                    for seed in pending:
                        inter = hold_polys[seed.id]
                        group = [seed]
                        for n in sorted(pending, key=lambda n: abs(n.position - seed.position)):
                            if n is seed:
                                continue
                            cand = intersection(inter, hold_polys[n.id])
                            if zone_ok(cand):
                                inter = cand
                                group.append(n)
                        if len(group) > len(best_group):
                            best_group, best_inter = group, inter
                    groups.append((best_group, best_inter, None))
                    grouped = {n.id for n in best_group}
                    pending = [n for n in pending if n.id not in grouped]
                used_pids: set[PointerID] = set()
                for group, inter, fixed_pid in groups:
                    members = {n.id for n in group}
                    candidates = []
                    for pid, p_touch in current_touches.items():
                        if fixed_pid is not None and pid != fixed_pid:
                            continue
                        if pid in used_pids or not inter.intersects(Point(p_touch.real, p_touch.imag)):
                            continue
                        if fixed_pid is None and max(abs(((p_touch - n.position) * n.rotation.conjugate()).real) for n in group) > hold_keep:
                            continue
                        bound = sum(1 for rec in pointers.occupied.values() if rec.id == pid)
                        candidates.append((pid in flicking_pids, -bound, pid))
                    if candidates:
                        pid = min(candidates)[2]
                        target = current_touches[pid]
                        act = None
                    else:
                        # 交集的质心离各条判定带的边最远
                        pt = inter.centroid
                        if not inter.contains(pt):
                            pt = inter.representative_point()
                        target = Position(pt.x, pt.y)
                        # 把组里Hold自己的指针拿过来，优先没有其他Hold按着的
                        owners = []
                        for n in group:
                            rec = pointers.occupied.get(n.id)
                            if rec is None or rec.id in used_pids or rec.id in flicking_pids or rec.id in downed_pids:
                                continue
                            others = sum(1 for nid, r in pointers.occupied.items() if r.id == rec.id and nid not in members)
                            owners.append((others, rec.id))
                        if owners:
                            pid = min(owners)[1]
                            act = TouchAction.MOVE
                        else:
                            head = group[0]
                            if head.id in pointers.occupied:
                                pointers.free(head)
                            pid, is_down = pointers.alloc(
                                SemiNote(head.type, target, head.id, head.rotation), new=False,
                                line_ref=note_id_to_line.get(head.id), note_offset=note_id_to_offset.get(head.id, 0.0)
                            )
                            if pid is None:
                                continue
                            act = TouchAction.DOWN if is_down else TouchAction.MOVE
                    if act == TouchAction.MOVE and abs(current_touches.get(pid, math.inf) - target) < 1e-6:
                        # 已经在质心上了，不用再发一次
                        act = None
                    if act is not None:
                        result[timestamp].append(VirtualTouchEvent(target, act, pid))
                        confirmed_pointers[pid] = target
                        current_touches[pid] = target
                    bind_holds(pid, target, group)
                    used_pids.add(pid)

            for note in drag_notes:
                poly_n = judge_zone(note.position, note.rotation, JUDGE_HALF_DRAG)
                covering_pid = None
                covering_pos = None
                for pid, p_touch in current_touches.items():
                    if poly_n.intersects(Point(p_touch.real, p_touch.imag)):
                        covering_pid = pid
                        covering_pos = p_touch
                        break
                line_ref = note_id_to_line.get(note.id)
                offset_val = note_id_to_offset.get(note.id, 0.0)
                if covering_pid is not None:
                    # unused 随时可能被拿走，需要临时occupied一下
                    if covering_pid in pointers.unused:
                        del pointers.unused[covering_pid]
                        pointers.occupied[note.id] = PointerRecord(
                            id=covering_pid, 
                            position=covering_pos, 
                            timestamp=timestamp, 
                            line_ref=line_ref, 
                            note_offset=offset_val, 
                            note_type=note.type
                        )
                        to_free.append(note)
                    continue
                pid, is_down = pointers.alloc(note, new=False, line_ref=line_ref, note_offset=offset_val)
                if pid is None:
                    continue
                act = TouchAction.DOWN if is_down else TouchAction.MOVE
                result[timestamp].append(VirtualTouchEvent(note.position, act, pid))
                to_free.append(note)
                confirmed_pointers[pid] = note.position
                current_touches[pid] = note.position
            
            for note_to_free in to_free:
                pointers.free(note_to_free)
            
            for event in result.get(timestamp, ()):
                if event.action != TouchAction.UP:
                    pointer_pos[event.pointer_id] = event.pos
            for ptr, up_ts in pointers.recycle():
                result[up_ts].append(VirtualTouchEvent(ptr.position, TouchAction.UP, ptr.id))
            
            progress.advance(task4, 1)
    
    for ptr, up_ts in pointers.finish():
        result[up_ts].append(VirtualTouchEvent(ptr.position, TouchAction.UP, ptr.id))
    
    console.print(f'重构规划完毕，总事件数{sum(len(events) for events in result.values())}.')
    return screen, [(ts, events) for ts, events in sorted(result.items())]

__all__ = ['solve']