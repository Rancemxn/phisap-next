from __future__ import annotations
import sys, math, time, argparse, importlib
from pathlib import Path
from basis import Chart, NoteType
from chart import load_chart
from rich.console import Console
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
    NoteType.UNKNOWN: (100, 100, 100),
}


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


def make_paint(color, alpha=1.0):
    paint = skia.Paint(AntiAlias=True)
    paint.setColor(skia.Color(*(max(0, min(255, round(v))) for v in color), round(max(0, min(1, alpha)) * 255)))
    return paint


class ChartRenderer:
    def __init__(s, chart: Chart, width=WW, height=WH):
        s.chart = chart
        s.width, s.height = width, height
        s.sc = (width / chart.width, height / chart.height)
        s.lines = sorted((line for line in chart.lines if line.attach_ui is None), key=lambda line: line.z_order)
        s.font_manager = skia.FontMgr.RefDefault()
        s.typeface = s.font_manager.matchFamilyStyle(None, skia.FontStyle.Normal())
        s.font = skia.Font(s.typeface, height * 0.045)
        s.fallback_fonts = {}
        s.textures = {}
        for line in s.lines:
            if line.texture != 'line.png':
                s.texture(line.texture)

    def point(s, pos):
        return pos.real * s.sc[0], pos.imag * s.sc[1]

    def font_for(s, char):
        if s.font.unicharToGlyph(ord(char)):
            return s.font
        if char not in s.fallback_fonts:
            face = s.font_manager.matchFamilyStyleCharacter("", skia.FontStyle.Normal(), ["zh", "ja"], ord(char))
            s.fallback_fonts[char] = skia.Font(face or s.typeface, s.font.getSize())
        return s.fallback_fonts[char]

    def draw_text(s, canvas, text, anchor, paint):
        rows = text.split("\n")
        row_height = s.font.getSize() * 1.2
        for index, row in enumerate(rows):
            runs = []
            for char in row:
                font = s.font_for(char)
                if runs and runs[-1][1] is font:
                    runs[-1] = runs[-1][0] + char, font
                else:
                    runs.append((char, font))
            width = sum(font.measureText(part) for part, font in runs)
            x = -width * anchor[0]
            y = (index - (len(rows) - 1) * (1 - anchor[1])) * row_height + s.font.getSize() * (anchor[1] - 0.2)
            for part, font in runs:
                canvas.drawString(part, x, y, font, paint)
                x += font.measureText(part)

    def texture(s, name):
        if name not in s.textures:
            image = None
            if s.chart.source is not None:
                root = s.chart.source.resolve().parent
                path = (root / name).resolve()
                if path.is_relative_to(root) and path.is_file():
                    try:
                        image = skia.Image.open(str(path))
                    except (RuntimeError, ValueError, OSError):
                        pass
            s.textures[name] = image
            if image is None:
                warning = f"Cannot load line texture {name!r}; displaying a plain line."
                if warning not in s.chart.warnings:
                    s.chart.warnings.append(warning)
        return s.textures[name]

    def draw_line(s, canvas, line, seconds):
        alpha = max(0.0, min(1.0, line.opacity @ seconds))
        if alpha == 0 or line.attach_ui is not None:
            return
        position = line.position @ seconds
        angle = math.degrees(line.angle @ seconds)
        sx, sy = line.scale_x @ seconds, line.scale_y @ seconds
        if not all(math.isfinite(v) for v in (position.real, position.imag, angle, sx, sy)):
            return
        color = line.color @ seconds
        paint = make_paint(color, alpha)
        canvas.save()
        canvas.translate(*s.point(position))
        canvas.rotate(angle)
        canvas.scale(sx, sy)
        text = line.text @ seconds
        if text is not None:
            s.draw_text(canvas, text, line.anchor, paint)
        else:
            texture = s.texture(line.texture) if line.texture != "line.png" else None
            if texture is not None:
                # 贴图宽高使用相同的像素倍率，避免在 16:9 屏幕上被压扁
                scale = s.width / 1350
                w, h = texture.width() * scale, texture.height() * scale
                paint.setColorFilter(skia.ColorFilters.Blend(make_paint(color).getColor(), skia.BlendMode.kModulate))
                canvas.drawImageRect(
                    texture, skia.Rect.MakeXYWH(-w * line.anchor[0], -h * (1 - line.anchor[1]), w, h), paint=paint
                )
            else:
                # 简洁预览仍使用长判定线，缩放事件同时作用于长度和线宽
                length = s.height * 5.76
                canvas.drawRect(skia.Rect.MakeXYWH(-length, -s.height * 0.00375, length * 2, s.height * 0.0075), paint)
        canvas.restore()

    def draw(s, canvas, seconds):
        for line in s.lines:
            s.draw_line(canvas, line, seconds)
            floor = line.floor @ seconds
            angle = math.degrees(line.angle @ seconds)
            for visual in line.visual_notes:
                state = line.note_state(seconds, visual, floor)
                if state is None:
                    continue
                hx, hy = s.point(state.head)
                tx, ty = s.point(state.tail)
                note_width = s.width * 0.1234375 * abs(state.width)
                head_height = s.width * 0.1234375 * 0.14 * abs(state.height)
                if not all(math.isfinite(v) for v in (hx, hy, tx, ty, note_width, head_height)):
                    continue
                margin = max(note_width, head_height)
                # 用完整的 Hold 包围盒裁剪，头在屏幕外时身体和尾部仍可能可见
                if (
                    max(hx, tx) < -margin
                    or min(hx, tx) > s.width + margin
                    or max(hy, ty) < -margin
                    or min(hy, ty) > s.height + margin
                ):
                    continue
                color = tuple(a * b / 255 for a, b in zip(NC[visual.note.type], visual.tint))
                paint = make_paint(color, state.alpha)
                rotation = angle + (0 if visual.above else 180)
                if visual.note.type == NoteType.HOLD:
                    length = math.hypot(tx - hx, ty - hy)
                    if length > 0:
                        body = make_paint(color, state.alpha * 0.4)
                        body_angle = math.degrees(math.atan2(ty - hy, tx - hx)) - 90
                        draw_center_rotate_rect(
                            canvas, (hx + tx) / 2, (hy + ty) / 2, note_width, length, body_angle, body
                        )
                        draw_center_rotate_round_rect(
                            canvas, tx, ty, note_width, head_height, head_height / 4, head_height / 4, rotation, paint
                        )
                if state.draw_head:
                    draw_center_rotate_round_rect(
                        canvas, hx, hy, note_width, head_height, head_height / 4, head_height / 4, rotation, paint
                    )


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
        s.events = [(ts, evt) for ts, evts in sorted(ans, key=lambda item: item[0]) for evt in evts]
        s.reset()

    def sample(s, t):
        if t < s._last_t:
            s.reset()
        while s._ei < len(s.events):
            ts, evt = s.events[s._ei]
            if ts > t:
                break
            s._last_state[evt.pointer_id] = (evt.action, evt.pos)
            if evt.action in (TouchAction.DOWN, TouchAction.MOVE):
                s._a[evt.pointer_id] = evt.pos
            elif evt.action in (TouchAction.UP, TouchAction.CANCEL):
                s._a.pop(evt.pointer_id, None)
            s._ei += 1
        s._last_t = t
        return s._a

    def get_states(s):
        return dict(s._last_state)

    def reset(s):
        s._ei = 0
        s._a = {}
        s._last_state = {}
        s._last_t = -math.inf


class CW(QWidget):
    def __init__(s, vc, psm, pids, sc, algorithm=4):
        super().__init__()
        s.setFocusPolicy(Qt.StrongFocus)
        s.setMouseTracking(True)
        s.setFixedSize(WW, WH)
        s.setFocus()
        s.vc, s.psm, s.pids, s.sc = vc, psm, pids, sc
        s.mode = f"ALGO{algorithm}" if algorithm is not None else "CHART ONLY"
        s.mouse_pos = None
        s.renderer = ChartRenderer(vc)
        s.surf = skia.Surface(WW, WH)
        s.paused = False
        s.ct = 0.0
        s.lr = time.monotonic()
        s.speeds = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
        s.speed_idx = s.speeds.index(1.0)
        s.sp = s.speeds[s.speed_idx]
        s.kd = {}
        s.ft = skia.Font(s.renderer.typeface, 20)
        s.fl = skia.Font(s.renderer.typeface, 13)
        s.tmr = QTimer(s)
        s.tmr.timeout.connect(s.tick)
        s.tmr.timeout.connect(s.update)
        s.tmr.start(16)

    def paintEvent(s, e):
        if not s.paused:
            nr = time.monotonic()
            s.ct = max(0.0, s.ct + (nr - s.lr) * s.sp)
            s.lr = nr
        now = s.ct
        chart_now = now - s.vc.offset
        c = s.surf.getCanvas()
        c.clear(skia.ColorBLACK)
        s.renderer.draw(c, chart_now)

        # 算法的结果以谱面时间计，必须与音符使用同一条时间轴
        act = s.psm.sample(round(chart_now * 1000))
        for pid in sorted(s.pids):
            if pid not in act:
                continue
            pos = act[pid]
            sx, sy = pos.real * s.sc[0], pos.imag * s.sc[1]
            if -100 <= sx <= WW + 100 and -100 <= sy <= WH + 100:
                ring = make_paint((255, 255, 255))
                ring.setStyle(skia.Paint.kStroke_Style)
                ring.setStrokeWidth(3)
                c.drawCircle(sx, sy, PR + 3, ring)
                c.drawCircle(sx, sy, PR, make_paint((255, 0, 0)))
                c.drawString(str(pid), sx + PR + 8, sy + 5, s.fl, make_paint((255, 255, 255)))
        text_paint = make_paint((255, 255, 255))
        c.drawString(f"{round(now * 1000)}ms  {s.sp}x  {s.vc.format.upper()}  {s.mode}", 8, 22, s.ft, text_paint)
        for i, (pid, (action, pos)) in enumerate(sorted(s.psm.get_states().items())):
            c.drawString(f"P{pid}:{action.name}({pos.real:.1f},{pos.imag:.1f})", 8, 46 + i * 16, s.fl, text_paint)

        mouse_text = ("Mouse: --", "Chart: --")
        if s.mouse_pos is not None:
            mx, my = s.mouse_pos.x(), s.mouse_pos.y()
            mouse_text = (f"Mouse: ({mx:.1f}, {my:.1f}) px", f"Chart: ({mx / s.sc[0]:.3f}, {my / s.sc[1]:.3f})")
        width = max(s.ft.measureText(text) for text in mouse_text)
        c.drawRect(skia.Rect.MakeXYWH(WW - width - 16, 0, width + 16, 52), make_paint((0, 0, 0), 0.7))
        for i, text in enumerate(mouse_text):
            c.drawString(text, WW - s.ft.measureText(text) - 8, 22 + i * 24, s.ft, text_paint)

        pixels = s.surf.makeImageSnapshot().tobytes()
        qi = QImage(pixels, WW, WH, QImage.Format_ARGB32_Premultiplied)
        qp = QPainter(s)
        qp.drawImage(0, 0, qi)
        qp.end()

    def enterEvent(s, ev):
        s.mouse_pos = ev.position()
        s.update()
        super().enterEvent(ev)

    def mouseMoveEvent(s, ev):
        s.mouse_pos = ev.position()
        s.update()
        super().mouseMoveEvent(ev)

    def leaveEvent(s, ev):
        s.mouse_pos = None
        s.update()
        super().leaveEvent(ev)

    def keyPressEvent(s, ev):
        super().keyPressEvent(ev)
        if ev.isAutoRepeat():
            return
        k = ev.key()
        if k == Qt.Key_Space:
            s.paused = not s.paused
            s.lr = time.monotonic()
        elif k == Qt.Key_0:
            s.ct = 0.0
            s.psm.reset()
            s.lr = time.monotonic()
        elif k in (Qt.Key_Equal, Qt.Key_Plus):
            s.speed_idx = min(s.speed_idx + 1, len(s.speeds) - 1)
            s.sp = s.speeds[s.speed_idx]
        elif k == Qt.Key_Minus:
            s.speed_idx = max(0, s.speed_idx - 1)
            s.sp = s.speeds[s.speed_idx]
        elif k in (Qt.Key_G, Qt.Key_H):
            s.ct = max(0.0, s.ct + (-10 if k == Qt.Key_G else 10))
            s.psm.reset()
            s.lr = time.monotonic()
        elif k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Comma, Qt.Key_Period, Qt.Key_Up, Qt.Key_Down):
            s.kd[k] = True
            s.paused = True
        elif k == Qt.Key_Escape:
            s.window().close()

    def keyReleaseEvent(s, ev):
        super().keyReleaseEvent(ev)
        if not ev.isAutoRepeat():
            s.kd[ev.key()] = False

    def tick(s):
        d = 0.0
        if s.kd.get(Qt.Key_Up):
            d = -0.1
        elif s.kd.get(Qt.Key_Down):
            d = 0.1
        elif s.kd.get(Qt.Key_Left):
            d = -0.01
        elif s.kd.get(Qt.Key_Right):
            d = 0.01
        elif s.kd.get(Qt.Key_Comma):
            d = -0.001
        elif s.kd.get(Qt.Key_Period):
            d = 0.001
        if d != 0:
            s.ct = max(0.0, s.ct + d)

    def closeEvent(s, event):
        s.tmr.stop()
        super().closeEvent(event)


def cdc():
    return {
        "algo1_flick_start": -20,
        "algo1_flick_end": 20,
        "algo1_flick_direction": 0,
        "algo1_sample_delay": 5,
        "algo1_target_score": 1000000,
        "algo1_strict_mode": True,
        "algo2_flick_start": -20,
        "algo2_flick_end": 20,
        "algo2_flick_direction": 0,
        "algo2_target_score": 1000000,
        "algo2_strict_mode": True,
        "algo2_continue_when_failed": False,
        "algo4_flick_start": -20,
        "algo4_flick_end": 20,
        "algo4_flick_direction": 0,
        "algo4_sample_delay": 5,
        "algo4_continue_when_failed": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Preview PGR, RPE and PEC charts with touch pointers.")
    parser.add_argument("chart", type=Path)
    parser.add_argument(
        "--no-solve", action="store_true", help="chart only, without algorithm planning or touch pointers"
    )
    parser.add_argument("--algorithm", type=int, choices=(1, 2, 3, 4), default=4, help="touch planner (default: 4)")
    parser.add_argument("--time", type=float, default=0.0, help="start at this playback time in seconds")
    args = parser.parse_args()
    if not math.isfinite(args.time) or args.time < 0:
        parser.error("--time must be a finite, non-negative number")
    console = Console()
    try:
        chart = load_chart(args.chart.read_text(encoding="utf-8-sig"), (16, 9), args.chart)
        print(
            f"Chart: {chart.format.upper()}, {len(chart.lines)} lines, {sum(len(l.notes) for l in chart.lines)} notes"
        )
        ans = []
        if not args.no_solve:
            print(f"Running algo{args.algorithm}...")
            algo = importlib.import_module(f"algo.algo{args.algorithm}")
            _, ans = algo.solve(chart, cdc(), console)
            print(f"Algo{args.algorithm} done: {len(ans)} frames")
    except (ValueError, OSError, RuntimeError):
        console.print_exception(show_locals=False)
        return 1
    pg()
    psm = PSM(ans)
    pids = {event.pointer_id for _, events in ans for event in events}
    print(f"Events: {len(psm.events)}, Pointers: {len(pids)}")
    app = QApplication(sys.argv)
    window = CW(chart, psm, pids, (WW / chart.width, WH / chart.height), None if args.no_solve else args.algorithm)
    for warning in chart.warnings:
        print(f"Warning: {warning}")
    window.setWindowTitle(f"phisap preview - {window.mode} - {chart.format.upper()} - {args.chart.name}")
    window.ct = args.time
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
