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
    get_parts
)
from shapely.ops import nearest_points
from shapely.affinity import rotate

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

class JudgeArea:
    __slots__ = ('center', 'rotation', 'poly', 'w_judge')

    def __init__(self, center: Position, rotation: Vector, screen_w: float, screen_h: float) -> None:
        self.center = center
        self.rotation = rotation
        self.w_judge = screen_w * 0.118125
        perp = rotation * 1j
        limit = math.hypot(screen_w, screen_h)
        d_rot = rotation * (self.w_judge / 2)
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
        min_dim = self.w_judge * 0.8
        geoms = get_parts(valid_poly)
        angle_deg = -math.degrees(cmath.phase(self.rotation))
        for geom in geoms:
            if geom.is_empty:
                continue
            rotated = rotate(geom, angle_deg, origin=(self.center.real, self.center.imag))
            minx, miny, maxx, maxy = rotated.bounds
            w = maxx - minx
            h = maxy - miny
            if w >= min_dim and h >= min_dim:
                return True
        return False

class PointerRecord(NamedTuple):
    id: PointerID
    position: Position
    timestamp: int
    line_ref: Any = None
    note_offset: float = 0.0
    note_type: SemiNoteType | None = None

class PointerManager:
    __slots__ = ('occupied', 'idle', 'unused', 'last_active_ts', 'waiting_liftup', 'current_ts', 'console', 'noway')

    def __init__(self, pointer_ids: Iterable[PointerID], console: Console, noway: bool = False) -> None:
        self.occupied: dict[NoteID, PointerRecord] = {}
        self.idle: set[PointerID] = set(pointer_ids)
        self.unused: dict[PointerID, PointerRecord] = {}
        self.last_active_ts: dict[PointerID, int] = {pid: 0 for pid in pointer_ids}
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
                if ptr.timestamp < self.current_ts
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
            return pid, True
        if self.unused:
            ptr = min(self.unused.values(), key=lambda p: abs(note.position - p.position))
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
            return ptr.id, True
        
        if self.noway:
            self.console.print(f"[red]Note({note}) @ {self.current_ts} 规划失败[/red]")
            return None, False
        raise RuntimeError(f'no free pointers @ {self.current_ts}')

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
            yield ptr, self.current_ts + 10

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
    pause_poly = Polygon([
        (screen.width * 0.85, 0),
        (screen.width * 0.95, 0),
        (screen.width * 0.95, screen.height * 0.10),
        (screen.width * 0.85, screen.height * 0.10)
    ])
    hold_ranges: list[tuple[int, int, int]] = []
    flick_ranges: list[tuple[int, int]] = []
    max_concurrent_holds = 0
    max_frame_must = 0
    max_frame_may = 0
    note_id_to_line: dict[NoteID, Any] = {}
    note_id_to_offset: dict[NoteID, float] = {}
    sweep_registry: defaultdict[int, list[SweepTarget]] = defaultdict(list)
    deferred_flicks: list[dict] = []

    def find_visible_pos(base_sec, base_pos, base_rot, note_offset, line_obj) -> tuple[float, Position, Vector, float, bool]:
        area_obj = JudgeArea(base_pos, base_rot, screen.width, screen.height)
        valid_touch_zone = area_obj.get_valid_poly(screen_poly, pause_poly)
        if area_obj.is_valid_zone(valid_touch_zone, screen.width):
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
                new_area_obj = JudgeArea(new_note_pos, new_rot, screen.width, screen.height)
                new_valid_zone = new_area_obj.get_valid_poly(screen_poly, pause_poly)
                if new_area_obj.is_valid_zone(new_valid_zone, screen.width):
                    orig_point_at_t = Point(new_note_pos.real, new_note_pos.imag)
                    closest_geom = nearest_points(new_valid_zone, orig_point_at_t)[0]
                    closest_pos = Position(closest_geom.x, closest_geom.y)
                    delta = closest_pos - new_lp
                    new_offset = (delta * new_rot.conjugate()).real
                    console.print(f"[yellow]判定时间微调：note @ {base_sec} of (pos={base_pos},rot={base_rot})=> note @ {new_time} of (pos={closest_pos},rot={new_rot})[/yellow]")
                    return new_time, closest_pos, new_rot, new_offset, True
        
        console.print(f"[red]判定微调失败：note @ {base_sec} of (pos={base_pos},rot={base_rot})[/red]")
        return base_sec, base_pos, base_rot, note_offset, False

    def flick_pos(pos: Position, offset_ms: int, rot: Vector, f_dir: Vector, start_off: int) -> Position:
        rate = 1 - 2 * (offset_ms - start_off) / flick_duration
        return pos + rot * f_dir * screen.flick_radius * rate
    
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
                    note.seconds, note_pos, rotation, note.offset, line
                )
                ts = round(adj_time * 1000)
                if note.type == NoteType.HOLD:
                    note_id_to_line[current_note_id] = line
                    note_id_to_offset[current_note_id] = adj_offset
                match note.type:
                    case NoteType.TAP:
                        frames[ts].append(SemiNote(SemiNoteType.TAP, adj_pos, current_note_id, adj_rot))
                        dense_frame_sizes[ts] += 1
                    case NoteType.DRAG:
                        area = JudgeArea(adj_pos, adj_rot, screen.width, screen.height)
                        poly = area.get_valid_poly(screen_poly, pause_poly)
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
                            area = JudgeArea(pos, rot_t, screen.width, screen.height)
                            valid_zone_t = area.get_valid_poly(screen_poly, pause_poly)
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
                        area_end = JudgeArea(end_pos_raw, rot2, screen.width, screen.height)
                        valid_zone_end = area_end.get_valid_poly(screen_poly, pause_poly)
                        if not valid_zone_end.is_empty:
                            orig_point_end = Point(end_pos_raw.real, end_pos_raw.imag)
                            end_pos = Position(nearest_points(valid_zone_end, orig_point_end)[0].x, nearest_points(valid_zone_end, orig_point_end)[0].y)
                        frames[ts + hold_ms].append(SemiNote(SemiNoteType.HOLD_END, end_pos, current_note_id, rot2))
                        dense_frame_sizes[ts + hold_ms] += 1

                current_note_id += 1
                progress.advance(task1, 1)
        
        total_flicks = len(deferred_flicks)
        task2 = progress.add_task("规划滑动轨迹...", total=total_flicks)
        
        for f_info in deferred_flicks:
            base_ts, base_pos, base_rot = f_info['ts'], f_info['pos'], f_info['rot']
            nid, line = f_info['id'], f_info['line']
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
            max_swept = -1
            best_swept_targets = []
            for c_dir in candidate_dirs:
                is_valid = True
                current_swept = []
                for off in flick_eval_offsets:
                    tick_ts = base_ts + off
                    p_flick = flick_pos(base_pos, off, base_rot, c_dir, flick_start)
                    test_area = JudgeArea(p_flick, base_rot, screen.width, screen.height)
                    test_poly = test_area.get_valid_poly(screen_poly, pause_poly)
                    if not test_area.is_valid_zone(test_poly, screen.width):
                        is_valid = False
                        break
                    test_point = Point(p_flick.real, p_flick.imag)
                    for target in sweep_registry.get(tick_ts, []):
                        if not target.is_swept and target.poly.intersects(test_point):
                            current_swept.append(target)
                if is_valid and len(current_swept) > max_swept:
                    max_swept = len(current_swept)
                    best_dir = c_dir
                    best_swept_targets = current_swept
            if max_swept == -1:
                best_dir = flick_dir
            for target in best_swept_targets:
                target.is_swept = True
            for off in flick_eval_offsets:
                tick_ts = base_ts + off
                p_flick = flick_pos(base_pos, off, base_rot, best_dir, flick_start)
                if off == flick_start:
                    frames[tick_ts].append(SemiNote(SemiNoteType.FLICK_START, p_flick, nid, base_rot))
                elif off == flick_end:
                    frames[tick_ts].append(SemiNote(SemiNoteType.FLICK_END, p_flick, nid, base_rot))
                else:
                    frames[tick_ts].append(SemiNote(SemiNoteType.FLICK, p_flick, nid, base_rot))
                dense_frame_sizes[tick_ts] += 1
            progress.advance(task2, 1)
        
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
        task3 = progress.add_task("规划触控事件...", total=total_frames)

        result: defaultdict[int, list[VirtualTouchEvent]] = defaultdict(list)
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
            for nid, record in pointers.occupied.items():
                active_physical_touches[record.id] = record.position
            for pid, record in pointers.unused.items():
                active_physical_touches[pid] = record.position
            
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
            
            active_areas = [
                JudgeArea(n.position, n.rotation, screen.width, screen.height)
                for n in must_notes
            ]
            active_polys = [
                area.get_valid_poly(screen_poly, pause_poly)
                for area in active_areas
            ]
            must_targets = [n.position for n in must_notes]
            max_iters = 5
            for iter_idx in range(max_iters):
                changed = False
                for i in range(len(must_notes)):
                    for j in range(i + 1, len(must_notes)):
                        pi = Point(must_targets[i].real, must_targets[i].imag)
                        pj = Point(must_targets[j].real, must_targets[j].imag)
                        inter_poly = intersection(active_polys[i], active_polys[j])
                        if not inter_poly.is_empty:
                            zone_i = difference(active_polys[i], active_polys[j])
                            zone_j = difference(active_polys[j], active_polys[i])
                            is_valid_i = active_areas[i].is_valid_zone(zone_i, screen.width)
                            is_valid_j = active_areas[j].is_valid_zone(zone_j, screen.width)
                            if is_valid_i and is_valid_j:
                                active_polys[i] = zone_i
                                active_polys[j] = zone_j
                                # representative_point 确保在判定区域内部
                                pt_i = zone_i.representative_point()
                                pt_j = zone_j.representative_point()
                                must_targets[i] = Position(pt_i.x, pt_i.y)
                                must_targets[j] = Position(pt_j.x, pt_j.y)
                                changed = True
                                console.print(
                                    f"[yellow]多押重叠调整：timestamp @ {timestamp}: note(pos={pi}) | note(pos={pj}) => note(pos={must_targets[i]}) | note(pos={must_targets[j]})[/yellow]"
                                )
                            else:
                                if not inter_poly.is_empty:
                                    pt_c = inter_poly.representative_point()
                                    target_c = Position(pt_c.x, pt_c.y)
                                    must_targets[i] = target_c
                                    must_targets[j] = target_c
                                    active_polys[i] = inter_poly
                                    active_polys[j] = inter_poly
                                    changed = True
                                    console.print(
                                        f"[yellow]多押重叠调整：timestamp @ {timestamp}: note(pos={pi}) | note(pos={pj}) => 2x note(pos={target_c})[/yellow]"
                                    )
                if not changed:
                    break
            
            for note, target in zip(must_notes, must_targets):
                line_ref = note_id_to_line.get(note.id)
                offset_val = note_id_to_offset.get(note.id, 0.0)
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
            current_touches.update(confirmed_pointers)
            flicking_pids = {
                r.id for r in pointers.occupied.values()
                if r.note_type in (SemiNoteType.FLICK_START, SemiNoteType.FLICK, SemiNoteType.FLICK_END)
            }
            for note in may_notes:
                line_ref = note_id_to_line.get(note.id)
                offset_val = note_id_to_offset.get(note.id, 0.0)
                poly_n = JudgeArea(note.position, note.rotation, screen.width, screen.height).get_valid_poly(screen_poly, pause_poly)
                candidates = []
                covering_pid = None
                covering_pos = None
                for pid, p_touch in current_touches.items():
                    if pid in flicking_pids:
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
            for note in passive_notes:
                poly_n = JudgeArea(note.position, note.rotation, screen.width, screen.height).get_valid_poly(screen_poly, pause_poly)
                self_record = pointers.occupied.get(note.id)
                self_pid = self_record.id if self_record is not None else None
                is_self_covered = False
                if self_pid is not None and self_pid in current_touches:
                    actual_pos = current_touches[self_pid]
                    if poly_n.intersects(Point(actual_pos.real, actual_pos.imag)):
                        is_self_covered = True
                        self_record = self_record._replace(position=actual_pos)
                is_covered = False
                covering_pid = None
                covering_pos = None
                for pid, p_touch in current_touches.items():
                    if pid == self_pid:
                        continue
                    if poly_n.intersects(Point(p_touch.real, p_touch.imag)):
                        is_covered = True
                        covering_pid = pid
                        covering_pos = p_touch
                        break
                line_ref = note_id_to_line.get(note.id)
                offset_val = note_id_to_offset.get(note.id, 0.0)
                if is_covered:
                    if note.type == SemiNoteType.DRAG:
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
                    elif note.type == SemiNoteType.HOLD:
                        if covering_pid is not None:
                            if note.id in pointers.occupied:
                                pointers.free(note)
                            if covering_pid in pointers.unused:
                                del pointers.unused[covering_pid]
                            pointers.occupied[note.id] = PointerRecord(covering_pid, covering_pos, timestamp, line_ref, offset_val, note.type)
                    elif note.type == SemiNoteType.HOLD_END:
                        if note.id in pointers.occupied:
                            to_free.append(note)
                elif note.type in (SemiNoteType.HOLD, SemiNoteType.HOLD_END) and is_self_covered:
                    pointers.occupied[note.id] = PointerRecord(self_pid, self_record.position, timestamp, line_ref, offset_val, note.type)
                    pointers.last_active_ts[self_pid] = timestamp
                    confirmed_pointers[self_pid] = self_record.position
                    current_touches[self_pid] = self_record.position
                    if note.type == SemiNoteType.HOLD_END:
                        to_free.append(note)
                else:
                    if note.type == SemiNoteType.DRAG:
                        pid, is_down = pointers.alloc(note, new=False, line_ref=line_ref, note_offset=offset_val)
                        if pid is None:
                            continue
                        act = TouchAction.DOWN if is_down else TouchAction.MOVE
                        result[timestamp].append(VirtualTouchEvent(note.position, act, pid))
                        to_free.append(note)
                        confirmed_pointers[pid] = note.position
                        current_touches[pid] = note.position 
                    elif note.type in (SemiNoteType.HOLD, SemiNoteType.HOLD_END):
                        if note.id in pointers.occupied:
                            pointers.free(note)
                        pid, is_down = pointers.alloc(note, new=False, line_ref=line_ref, note_offset=offset_val)
                        if pid is None:
                            continue
                        act = TouchAction.DOWN if is_down else TouchAction.MOVE
                        result[timestamp].append(VirtualTouchEvent(note.position, act, pid))
                        confirmed_pointers[pid] = note.position
                        current_touches[pid] = note.position
                        if note.type == SemiNoteType.HOLD_END:
                            to_free.append(note)
            
            for note_to_free in to_free:
                pointers.free(note_to_free)
            
            for ptr, up_ts in pointers.recycle():
                result[up_ts].append(VirtualTouchEvent(ptr.position, TouchAction.UP, ptr.id))
            
            progress.advance(task3, 1)
    
    for ptr, up_ts in pointers.finish():
        result[up_ts].append(VirtualTouchEvent(ptr.position, TouchAction.UP, ptr.id))
    
    console.print(f'重构规划完毕，总事件数{sum(len(events) for events in result.values())}.')
    return screen, [(ts, events) for ts, events in sorted(result.items())]

__all__ = ['solve']