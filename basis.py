from abc import ABCMeta, abstractmethod
from enum import IntEnum
from bamboo import Bamboo, BambooShoot
from typing import TypeAlias, NamedTuple
from dataclasses import dataclass
from pathlib import Path
import math

Position: TypeAlias = complex
Vector: TypeAlias = complex


class NoteType(IntEnum):
    UNKNOWN = -1
    TAP = 0
    DRAG = 1
    HOLD = 2
    FLICK = 3


class Note(NamedTuple):
    type: NoteType
    seconds: float
    hold: float
    offset: Position


@dataclass(slots=True)
class VisualNote:
    note: Note
    position_x: float = 0.0
    y_offset: float = 0.0
    speed: float = 1.0
    above: bool = True
    alpha: float = 1.0
    size: float = 1.0
    is_fake: bool = False
    visible_time: float = math.inf
    floor: float = 0.0
    end_floor: float = 0.0
    tint: tuple[int, int, int] = (255, 255, 255)


class NoteState(NamedTuple):
    head: Position
    tail: Position
    width: float
    height: float
    alpha: float
    draw_head: bool


class JudgeLine(metaclass=ABCMeta):
    notes: list[Note]
    position: Bamboo[Position]
    angle: Bamboo[float]

    def __init__(self) -> None:
        self.notes = []
        self.visual_notes: list[VisualNote] = []
        self.opacity = BambooShoot(1.0)
        self.speed = BambooShoot(0.0)
        self.floor = BambooShoot(0.0)
        self.scale_x = BambooShoot(1.0)
        self.scale_y = BambooShoot(1.0)
        self.incline = BambooShoot(0.0)
        self.color = BambooShoot((255, 236, 159))
        self.text = BambooShoot(None)
        self.pos_control = BambooShoot(1.0)
        self.y_control = BambooShoot(1.0)
        self.size_control = BambooShoot(1.0)
        self.alpha_control = BambooShoot(1.0)
        self.control_scale = 1.0
        self.is_cover = True
        self.z_order = 0
        self.attach_ui = None
        self.texture = 'line.png'
        self.anchor = (0.5, 0.5)

    def note_distances(self, seconds: float, visual: VisualNote, floor: float, y_control: float) -> tuple[float, float]:
        head = (visual.floor - floor) * visual.speed * y_control
        tail = (visual.end_floor - floor) * visual.speed * y_control
        if visual.note.type == NoteType.HOLD and not visual.is_fake and seconds >= visual.note.seconds:
            head = 0.0
        offset = visual.y_offset * visual.speed
        return head + offset, tail + offset

    def note_coordinates(self, seconds: float, visual: VisualNote, floor: float | None = None) -> NoteState:
        if floor is None:
            floor = self.floor @ seconds
        distance = (visual.floor - floor + visual.y_offset) * self.control_scale
        head, tail = self.note_distances(seconds, visual, floor, self.y_control @ distance)
        x = visual.position_x
        is_hold = visual.note.type == NoteType.HOLD
        if not is_hold:
            x *= (self.pos_control @ distance) * (
                1 - math.sin(self.incline @ seconds) * head * self.control_scale / 360
            )
        size = self.size_control @ distance
        alpha = max(0.0, min(1.0, visual.alpha * (self.alpha_control @ distance)))
        side = -1 if visual.above else 1
        return NoteState(
            complex(x, head * side),
            complex(x, tail * side),
            visual.size * size,
            1.0 if is_hold else size,
            alpha,
            visual.is_fake or seconds <= visual.note.seconds,
        )

    def notes_visible(self, seconds: float, visual: VisualNote) -> bool:
        return self.opacity @ seconds >= 0

    def cover_distance(self, seconds: float, visual: VisualNote, floor: float) -> float:
        distance = (visual.floor - floor + visual.y_offset) * self.control_scale
        speed = visual.speed * (self.y_control @ distance)
        target = visual.end_floor if visual.note.type == NoteType.HOLD else visual.floor
        return (target - floor) * speed

    def note_state(self, seconds: float, visual: VisualNote, floor: float | None = None) -> NoteState | None:
        note = visual.note
        if seconds < note.seconds - visual.visible_time:
            return None
        keep_fake = visual.is_fake and not self.is_cover and note.type != NoteType.HOLD
        if seconds > note.seconds + note.hold and not keep_fake:
            return None
        if not self.notes_visible(seconds, visual):
            return None
        if floor is None:
            floor = self.floor @ seconds
        state = self.note_coordinates(seconds, visual, floor)
        if state.alpha <= 0:
            return None
        if self.is_cover and seconds < note.seconds and self.cover_distance(seconds, visual, floor) < -1e-8:
            return None
        return state._replace(head=self.pos(seconds, state.head), tail=self.pos(seconds, state.tail))

    @abstractmethod
    def pos(self, seconds: float, offset: Position) -> Position: ...

    @abstractmethod
    def beat_duration(self, seconds: float) -> float: ...


class Chart(metaclass=ABCMeta):
    width: int
    height: int
    offset: float
    lines: list[JudgeLine]

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.source: Path | None = None
