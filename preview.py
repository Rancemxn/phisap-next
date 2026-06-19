from __future__ import annotations
import sys, json, math, time
from basis import Vector, Position, NoteType
from bamboo import BrokenBamboo
from rich.console import Console
from pgr import PgrChart
from algo import algo4
from algo.base import TouchAction

import skia
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtGui import QImage, QPainter
from PySide6.QtCore import QTimer, Qt

WW, WH = 1280, 720
PR = 16
NC = {
    NoteType.TAP: (10, 195, 255),
    NoteType.DRAG: (240, 237, 105),
    NoteType.HOLD: (0, 255, 255),
    NoteType.FLICK: (254, 67, 101),
    NoteType.UNKNOWN: (100, 100, 100)
}

def rotate_point(x: float, y: float, r: float, deg: float) -> tuple[float, float]:
    rad = math.radians(deg)
    return (
        x + r * math.cos(rad),
        y + r * math.sin(rad)
    )

def find_event(t: float, es: list[dict]) -> int:
    if not es:
        return -1
    l, r = 0, len(es) - 1
    while l <= r:
        m = (l + r) // 2
        e = es[m]
        if e["startTime"] <= t <= e["endTime"]:
            return m
        elif e["startTime"] > t:
            r = m - 1
        else:
            l = m + 1
    return -1

def get_event_val(t: float, es: list[dict], sn: str, en: str, default: float = 0.0) -> float:
    i = find_event(t, es)
    if i == -1:
        if es and t > es[-1]["endTime"]:
            return es[-1][en]
        if es and t < es[0]["startTime"]:
            return es[0][sn]
        return default
    e = es[i]
    st, et = e["startTime"], e["endTime"]
    sv, ev = e[sn], e[en]
    if et == st:
        return sv
    return sv + (t - st) / (et - st) * (ev - sv)

def get_fp(t: float, es: list[dict]) -> float:
    i = find_event(t, es)
    if i == -1:
        if es and t > es[-1]["endTime"]:
            last = es[-1]
            return last["floorPosition"] + (t - last["startTime"]) * last["value"]
        return 0.0
    e = es[i]
    return e["floorPosition"] + (t - e["startTime"]) * e["value"]

def get_line_state(line: VPJL, t: float, fv: int) -> tuple[float, float, float, float]:
    rotate = get_event_val(t, line.rotate_events, "start", "end", 0.0) * -1
    
    move_events = line.move_events
    x, y = 0.5, 0.5
    
    if move_events:
        i = find_event(t, move_events)
        if i == -1:
            if t > move_events[-1]["endTime"]:
                e = move_events[-1]
                if fv == 1:
                    ev = e["end"]
                    x, y = (ev // 1000) / 880.0, 1.0 - (ev % 1000) / 520.0
                else:
                    x = e["end"]
                    y = 1.0 - e.get("end2", 0.0)
            elif t < move_events[0]["startTime"]:
                e = move_events[0]
                if fv == 1:
                    sv = e["start"]
                    x, y = (sv // 1000) / 880.0, 1.0 - (sv % 1000) / 520.0
                else:
                    x = e["start"]
                    y = 1.0 - e.get("start2", 0.0)
        else:
            e = move_events[i]
            st, et = e["startTime"], e["endTime"]
            if fv == 1:
                sv, ev = e["start"], e["end"]
                sx, sy = (sv // 1000) / 880.0, (sv % 1000) / 520.0
                ex, ey = (ev // 1000) / 880.0, (ev % 1000) / 520.0
                if et == st:
                    x, y = sx, sy
                else:
                    ratio = (t - st) / (et - st)
                    x = sx + ratio * (ex - sx)
                    y = sy + ratio * (ey - sy)
                y = 1.0 - y
            else:
                s2, e2 = e.get("start2", 0.0), e.get("end2", 0.0)
                if et == st:
                    x = e["start"]
                    y = 1.0 - s2
                else:
                    ratio = (t - st) / (et - st)
                    x = e["start"] + ratio * (e["end"] - e["start"])
                    y = 1.0 - (s2 + ratio * (e2 - s2))
                    
    alpha = get_event_val(t, line.disappear_events, "start", "end", 1.0)
    return rotate, x, y, alpha

class VN:
    __slots__ = ("t", "time", "offset", "hold", "speed", "floor", "above", "nid", "hold_length")
    def __init__(s, t, time, offset, hold, speed, floor, above, nid, hold_length):
        s.t = t
        s.time = time
        s.offset = offset
        s.hold = hold
        s.speed = speed
        s.floor = floor
        s.above = above
        s.nid = nid
        s.hold_length = hold_length

class VPJL:
    _NT = [NoteType.UNKNOWN, NoteType.TAP, NoteType.DRAG, NoteType.HOLD, NoteType.FLICK]
    _NNI = 0
    def __init__(s, d, fv, lid):
        s.lid = lid
        s.bpm = d["bpm"]
        s.raw_data = d
        bl = 1.875 / s.bpm
        
        s.speed_events = []
        for e in d.get("speedEvents", []):
            s.speed_events.append({
                "startTime": e["startTime"] * bl,
                "endTime": e["endTime"] * bl,
                "value": e["value"],
                "floorPosition": 0.0
            })
        s.speed_events.sort(key=lambda x: x["startTime"])
        
        s.rotate_events = []
        for e in d.get("judgeLineRotateEvents", []):
            s.rotate_events.append({
                "startTime": e["startTime"] * bl,
                "endTime": e["endTime"] * bl,
                "start": e["start"],
                "end": e["end"]
            })
        s.rotate_events.sort(key=lambda x: x["startTime"])
        
        s.move_events = []
        for e in d.get("judgeLineMoveEvents", []):
            s.move_events.append({
                "startTime": e["startTime"] * bl,
                "endTime": e["endTime"] * bl,
                "start": e["start"],
                "end": e["end"],
                "start2": e.get("start2", 0.0),
                "end2": e.get("end2", 0.0)
            })
        s.move_events.sort(key=lambda x: x["startTime"])
        
        s.disappear_events = []
        for e in d.get("judgeLineDisappearEvents", []):
            s.disappear_events.append({
                "startTime": e["startTime"] * bl,
                "endTime": e["endTime"] * bl,
                "start": e["start"],
                "end": e["end"]
            })
        s.disappear_events.sort(key=lambda x: x["startTime"])

        fp = 0.0
        for e in s.speed_events:
            e["floorPosition"] = fp
            fp += (e["endTime"] - e["startTime"]) * e["value"]
            
        s.notes = []
        for is_above, src_key in [(True, "notesAbove"), (False, "notesBelow")]:
            src = d.get(src_key, [])
            for n in src:
                ni = VPJL._NNI
                VPJL._NNI += 1
                note_beat_time = n["time"]
                note_time_sec = note_beat_time * bl
                
                note_floor = get_fp(note_time_sec, s.speed_events)
                
                hold_beat = n.get("holdTime", 0.0)
                hold_sec = hold_beat * bl
                hold_length = hold_sec * n.get("speed", 1.0)
                
                s.notes.append(VN(
                    t=s._NT[n["type"]],
                    time=note_time_sec,
                    offset=n["positionX"],
                    hold=hold_sec,
                    speed=n.get("speed", 1.0),
                    floor=note_floor,
                    above=is_above,
                    nid=ni,
                    hold_length=hold_length
                ))

class VPC:
    def __init__(s, d):
        s.fv = d["formatVersion"]
        s.offset = d["offset"]
        s.lines = [VPJL(l, s.fv, i) for i, l in enumerate(d["judgeLineList"])]
        s.ss = Position(880, 520) if s.fv == 1 else Position(16, 9)

def draw_center_rotate_rect(canvas, cx, cy, w, h, deg, paint):
    canvas.save()
    canvas.translate(cx, cy)
    canvas.rotate(deg)
    canvas.drawRect(skia.Rect.MakeXYWH(-w / 2, -h / 2, w, h), paint)
    canvas.restore()
    
def draw_center_rotate_round_rect(canvas, cx, cy, w, h, rx, ry, deg, paint):
    canvas.save()
    canvas.translate(cx, cy)
    canvas.rotate(deg)
    canvas.drawRoundRect(skia.Rect.MakeXYWH(-w / 2, -h / 2, w, h), rx, ry, paint)
    canvas.restore()

def pg():
    print("  Controls:")
    print("    SPACE - Pause/Resume")
    print("    0 - Reset")
    print("    Arrows - Seek 0.01/0.1s")
    print("    ,/. - Seek 0.001s")
    print("    G/H - Seek -10s/+10s")
    print("    =/- - Speed Tier")
    print("    ESC - Exit")
    print()

class PSM:
    def __init__(s, ans):
        s.events = [(ts, evt) for ts, evts in sorted(ans) for evt in evts]
        s._ei = 0
        s._a = {}
        s._last_state = {}
    def sample(s, t):
        while s._ei < len(s.events):
            ts, evt = s.events[s._ei]
            if ts > t: break
            s._last_state[evt.pointer_id] = (evt.action, evt.pos)
            if evt.action in (TouchAction.DOWN, TouchAction.MOVE):
                s._a[evt.pointer_id] = evt.pos
            elif evt.action in (TouchAction.UP, TouchAction.CANCEL):
                s._a.pop(evt.pointer_id, None)
            s._ei += 1
        return s._a
    def get_states(s):
        return dict(s._last_state)
    def reset(s):
        s._ei = 0
        s._a = {}
        s._last_state = {}

class CW(QWidget):
    def __init__(s, vc, psm, pids, sc):
        super().__init__()
        s.setFocusPolicy(Qt.StrongFocus)
        s.setFixedSize(WW, WH)
        s.setFocus()
        s.vc = vc
        s.psm = psm
        s.pids = pids
        s.sc = sc
        s.surf = skia.Surface(WW, WH)
        s.paused = False
        s.ct = 0.0
        s.lr = time.monotonic()
        
        s.speeds = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
        s.speed_idx = s.speeds.index(1.0)
        s.sp = s.speeds[s.speed_idx]
        
        s.kd = {}
        s.ft = skia.Font(skia.Typeface.MakeFromName("Arial", skia.FontStyle.Normal()), 20)
        s.fl = skia.Font(skia.Typeface.MakeFromName("Arial", skia.FontStyle.Bold()), 13)
        s.tmr = QTimer()
        s.tmr.timeout.connect(s.update)
        s.tmr.timeout.connect(s.tick)
        s.tmr.start(16)

    def paintEvent(s, e):
        if s.paused:
            now = s.ct
        else:
            nr = time.monotonic()
            dt = nr - s.lr
            s.lr = nr
            s.ct += dt * s.sp
            s.ct = max(0.0, s.ct)
            now = s.ct
            
        chart_now = now - s.vc.offset
            
        c = s.surf.getCanvas()
        c.clear(skia.ColorBLACK)
        
        for line in s.vc.lines:
            lineRotate, lineX_norm, lineY_norm, lineAlpha = get_line_state(line, chart_now, s.vc.fv)
            
            lineX = lineX_norm * WW
            lineY = lineY_norm * WH
            
            lw = WH * 0.0075
            lh = WH * 5.76
            lx0, ly0 = rotate_point(lineX, lineY, lh, lineRotate)
            lx1, ly1 = rotate_point(lineX, lineY, lh, lineRotate + 180)
            
            lp = skia.Paint()
            lp.setColor(skia.Color(255, 236, 159, int(255 * lineAlpha)))
            lp.setStyle(skia.Paint.kStroke_Style)
            lp.setStrokeWidth(lw)
            lp.setAntiAlias(True)
            c.drawLine(lx0, ly0, lx1, ly1, lp)
            
            linefp = get_fp(chart_now, line.speed_events)
            
            for n in line.notes:
                is_hold = (n.t == NoteType.HOLD)
                if (not is_hold and n.time < chart_now) or (is_hold and (n.time + n.hold) < chart_now):
                    continue
                
                note_fp_y = n.floor - linefp
                if not is_hold:
                    note_fp_y *= n.speed
                
                note_fp = note_fp_y * 0.6 * WH
                
                if not is_hold and note_fp < -10000.0:
                    continue
                if note_fp > WH * 2:
                    continue
                    
                note_width = WW * 0.1234375
                this_note_width = note_width
                this_note_head_height = this_note_width * 0.14
                
                at_x, at_y = rotate_point(lineX, lineY, n.offset * 0.05625 * WW, lineRotate)
                l2n_rotate = lineRotate - (90 if n.above else -90)
                hx, hy = rotate_point(at_x, at_y, note_fp, l2n_rotate)
                note_draw_rotate = lineRotate + (0 if n.above else 180)
                
                base_op = 1.0
                if chart_now > n.time and not is_hold:
                    base_op *= max(0.0, 1.0 - (chart_now - n.time) / 0.16)
                    
                r, g, b = NC.get(n.t, (100, 100, 100))
                
                paint_fill = skia.Paint()
                paint_fill.setColor(skia.Color(r, g, b, int(base_op * 255)))
                paint_fill.setStyle(skia.Paint.kFill_Style)
                paint_fill.setAntiAlias(True)
                
                draw_head = (n.time > chart_now)
                if draw_head:
                    rx, ry = this_note_head_height * 0.25, this_note_head_height * 0.25
                    draw_center_rotate_round_rect(c, hx, hy, this_note_width, this_note_head_height, rx, ry, note_draw_rotate, paint_fill)
                    
                if is_hold:
                    note_tail_height = this_note_width * 0.14
                    clicked = (chart_now >= n.time)
                    
                    if not clicked:
                        note_body_height = max(0.0, n.hold_length * 0.6 * WH - note_tail_height / 2.0)
                        body_center_offset = this_note_head_height / 2.0 + note_body_height / 2.0
                        bx, by = rotate_point(hx, hy, body_center_offset, l2n_rotate)
                    else:
                        remaining_sec = (n.time + n.hold) - chart_now
                        remaining_len_y = max(0.0, remaining_sec * n.speed)
                        note_body_height = max(0.0, remaining_len_y * 0.6 * WH - note_tail_height / 2.0)
                        body_center_offset = note_body_height / 2.0
                        bx, by = rotate_point(at_x, at_y, body_center_offset, l2n_rotate)
                        
                    if note_body_height > 0.0:
                        paint_body = skia.Paint()
                        paint_body.setColor(skia.Color(r, g, b, int(base_op * 0.4 * 255)))
                        paint_body.setStyle(skia.Paint.kFill_Style)
                        paint_body.setAntiAlias(True)
                        draw_center_rotate_rect(c, bx, by, this_note_width, note_body_height, note_draw_rotate, paint_body)
                        
                        tx, ty = rotate_point(bx, by, note_body_height / 2.0 + note_tail_height / 2.0, l2n_rotate)
                        paint_tail = skia.Paint()
                        paint_tail.setColor(skia.Color(r, g, b, int(base_op * 255)))
                        paint_tail.setStyle(skia.Paint.kFill_Style)
                        paint_tail.setAntiAlias(True)
                        rx, ry = note_tail_height * 0.25, note_tail_height * 0.25
                        draw_center_rotate_round_rect(c, tx, ty, this_note_width, note_tail_height, rx, ry, note_draw_rotate, paint_tail)

        act = s.psm.sample(int(now * 1000))
        for pid in sorted(s.pids):
            if pid in act:
                pos = act[pid]
                sx = pos.real * s.sc[0]
                sy = pos.imag * s.sc[1]
                if -100 <= sx <= WW + 100 and -100 <= sy <= WH + 100:
                    r_paint = skia.Paint(skia.Color(255, 255, 255))
                    r_paint.setStyle(skia.Paint.kStroke_Style)
                    r_paint.setStrokeWidth(3)
                    r_paint.setAntiAlias(True)
                    c.drawCircle(sx, sy, PR + 3, r_paint)
                    
                    d_paint = skia.Paint(skia.Color(255, 0, 0))
                    d_paint.setStyle(skia.Paint.kFill_Style)
                    d_paint.setAntiAlias(True)
                    c.drawCircle(sx, sy, PR, d_paint)
                    
                    lp = skia.Paint(skia.Color(255, 255, 255))
                    c.drawString(str(pid), sx + PR + 8, sy + 5, s.fl, lp)
                    
        tp = skia.Paint(skia.Color(255, 255, 255))
        c.drawString(f"{int(now * 1000)}ms  {s.sp}x", 8, 22, s.ft, tp)
        
        sts = s.psm.get_states()
        for i, pid in enumerate(sorted(sts)):
            act, pos = sts[pid]
            acn = act.name
            c.drawString(f"P{pid}:{acn}({pos.real:.1f},{pos.imag:.1f})", 8, 46 + i * 16, s.fl, tp)
            
        qi = QImage(s.surf.makeImageSnapshot().tobytes(), WW, WH, QImage.Format_ARGB32_Premultiplied)
        qp = QPainter(s)
        qp.drawImage(0, 0, qi)
        qp.end()

    def keyPressEvent(s, ev):
        super().keyPressEvent(ev)
        k = ev.key()
        if k == Qt.Key_Space:
            if s.paused:
                s.paused = False
                s.lr = time.monotonic()
            else:
                s.paused = True
        elif k == Qt.Key_0:
            s.ct = 0.0
            s.psm.reset()
            if not s.paused: s.lr = time.monotonic()
        
        elif k in (Qt.Key_Equal, Qt.Key_Plus):
            if s.speed_idx < len(s.speeds) - 1:
                s.speed_idx += 1
                s.sp = s.speeds[s.speed_idx]
        elif k == Qt.Key_Minus:
            if s.speed_idx > 0:
                s.speed_idx -= 1
                s.sp = s.speeds[s.speed_idx]
                
        elif k == Qt.Key_G:
            s.ct = max(0.0, s.ct - 10.0)
            s.psm.reset()
            if not s.paused: s.lr = time.monotonic()
            
        elif k == Qt.Key_H:
            s.ct += 10.0
            s.psm.reset()
            if not s.paused: s.lr = time.monotonic()
            
        elif k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Comma, Qt.Key_Period, Qt.Key_Up, Qt.Key_Down):
            s.kd[k] = True
            if not s.paused: s.paused = True
        elif k == Qt.Key_Escape:
            s.window().close()

    def keyReleaseEvent(s, ev):
        super().keyReleaseEvent(ev)
        k = ev.key()
        if k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Comma, Qt.Key_Period, Qt.Key_Up, Qt.Key_Down):
            s.kd[k] = False

    def tick(s):
        d = 0.0
        if s.kd.get(Qt.Key_Up): d = -.1
        elif s.kd.get(Qt.Key_Down): d = .1
        elif s.kd.get(Qt.Key_Left): d = -.01
        elif s.kd.get(Qt.Key_Right): d = .01
        elif s.kd.get(Qt.Key_Comma): d = -.001
        elif s.kd.get(Qt.Key_Period): d = .001
        if d != 0.0:
            if s.ct + d < s.ct: s.psm.reset()
            s.ct = max(0.0, s.ct + d)

def dcf(d):
    if "formatVersion" in d: return "pgr"
    if "META" in d and "BPMList" in d: return "rpe"
    return "unknown"

def cdc():
    return {
        "algo1_flick_start": -20, "algo1_flick_end": 20, "algo1_flick_direction": 0,
        "algo1_sample_delay": 5, "algo1_target_score": 1000000, "algo1_strict_mode": True,
        
        "algo2_flick_start": 20, "algo2_flick_end": -20, "algo2_flick_direction": 0,
        "algo2_target_score": 1000000, "algo2_strict_mode": True, "algo2_continue_when_failed": False,
        
        "algo4_flick_start": -20, "algo4_flick_end": 20, "algo4_flick_direction": 0,
        "algo4_sample_delay": 5, "algo4_continue_when_failed": True
    }

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        pg()
        sys.exit(0)
        
    jp = sys.argv[1]
    print(f"Loading chart: {jp}")
    with open(jp, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    fmt = dcf(data)
    if fmt != "pgr":
        print(f"Warning: {fmt} only PGR supported.")
        if fmt == "unknown":
            print("Unknown format.")
            sys.exit(1)
            
    pg()
    console = Console()
    ac = PgrChart(data, (16, 9))
    print(f"Chart: {len(ac.lines)} lines")
    config = cdc()
    
    print("Running algo4...")
    screen, ans = algo4.solve(ac, config, console)
    print(f"Algo4 done: {len(ans)} frames")
    
    vc = VPC(data)
    print(f"Visual: {len(vc.lines)} lines")
    
    psm = PSM(ans)
    pids = set()
    for _, evts in ans:
        for evt in evts:
            pids.add(evt.pointer_id)
    print(f"Events: {len(psm.events)}, Pointers: {len(pids)}")
    
    sc = (WW / vc.ss.real, WH / vc.ss.imag)
    app = QApplication(sys.argv)
    w = CW(vc, psm, pids, sc)
    w.show()
    app.exec()

if __name__ == "__main__":
    main()