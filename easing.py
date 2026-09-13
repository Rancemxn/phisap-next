from typing import Callable
from types import CodeType, FunctionType
from enum import Enum, member
from functools import partial
from math import pi, sin, cos, sqrt, isclose
import bisect


def _easing_linear(
    start: tuple[float, float, float], end: tuple[float, float, float], t: float
) -> tuple[float, float, float]:
    return tuple(a + (b - a) * t for a, b in zip(start, end))


def _easing_cubic_bezier(
    start: tuple[float, float, float], end: tuple[float, float, float], t: float
) -> tuple[float, float, float]:
    a = (1 - t) * (1 + 2 * t)
    b = t * (3 - 2 * t)
    start = (start[0] * a, start[1], start[2] * a)
    end = (end[0] * b, end[1], end[2] * b)
    return _easing_linear(start, end, t)


def _easing_sinus(
    start: tuple[float, float, float], end: tuple[float, float, float], t: float, x: str, z: str | None = None
) -> tuple[float, float, float]:
    x0, y0, z0 = start
    x1, y1, z1 = end
    if x == 'si':
        sx = sin(t * pi / 2)
    elif x == 'so':
        sx = 1 - cos(t * pi / 2)
    else:
        raise RuntimeError(f'unknown easing type x = {x}')
    if z == 'si':
        sz = sin(t * pi / 2)
    elif z == 'so':
        sz = 1 - cos(t * pi / 2)
    else:
        sz = t
    return x0 + (x1 - x0) * sx, y0 + (y1 - y0) * t, z0 + (z1 - z0) * sz


def _in(expr: str) -> str:
    return expr


def _out(expr: str) -> str:
    return f'1 - ({expr.replace("x", "(1 - x)")})'


def _inout(expr: str) -> str:
    return f'({_in(expr).replace("x", "(2 * x)")}) / 2 if x < 0.5 else (1 + ({_out(expr).replace("x", "(2 * x - 1)")})) / 2'


def _outin(expr: str) -> str:
    return f'({_out(expr).replace("x", "(2 * x)")}) / 2 if x < 0.5 else (1 + ({_in(expr).replace("x", "(2 * x - 1)")})) / 2'


_EASING_BASIC_FUNCTIONS = {
    'linear': 'x',
    'sine': '1 - cos(x * pi / 2)',
    'quad': 'x ** 2',
    'cubic': 'x ** 3',
    'quart': 'x ** 4',
    'quint': 'x ** 5',
    'circ': '1 - sqrt(1 - x * x)',
    'expo': '0 if x == 0 else 2. ** (10 * x - 10)',
    'back': '(2.70158 * x - 1.70158) * x ** 2',
    'elastic': f'x if x == 0 or x == 1 else -(2 ** (10 * x - 10) * sin({2 * pi / 3} * (x * 10. - 10.75)))',
    'bounce': _out(
        f'A * x ** 2 if x < {1 / 2.75} else (A * (x - {1.5 / 2.75}) ** 2 + 0.75 if x < {2 / 2.75} else (A * (x - {2.25 / 2.75}) ** 2 + 0.9375 if x < {2.5 / 2.75} else A * (x - {2.625 / 2.75}) ** 2 + 0.984375))'.replace(
            'A', str(7.5625)
        )
    ),
}

_EASING_SUFFIXES = [_in, _out, _inout, _outin]


def easing(easing_basic: tuple[str, str], suffix: Callable[[str], str]) -> Callable[[float], float]:
    name, expr = easing_basic
    name += suffix.__name__
    f = compile('lambda x:' + suffix(expr), '<string>', 'eval')
    code = [c for c in f.co_consts if isinstance(c, CodeType)][0]
    return FunctionType(code, globals())


EasingFunction = Callable[[float], float]

EASING_FUNCTIONS: dict[str, EasingFunction] = {
    b + s.__name__: easing((b, fn), s) for s in _EASING_SUFFIXES for b, fn in _EASING_BASIC_FUNCTIONS.items()
}

# Back/Elastic 的 InOut 不是简单拼接两段 In/Out。
EASING_FUNCTIONS['back_inout'] = lambda t: (
    (2 * t) ** 2 * (3.5949095 * 2 * t - 2.5949095) / 2
    if t < 0.5
    else ((2 * t - 2) ** 2 * (3.5949095 * (2 * t - 2) + 2.5949095) + 2) / 2
)
EASING_FUNCTIONS['elastic_inout'] = lambda t: (
    t
    if t == 0 or t == 1
    else (
        -(2 ** (20 * t - 10) * sin((20 * t - 11.125) * (2 * pi / 4.5))) / 2
        if t < 0.5
        else 2 ** (-20 * t + 10) * sin((20 * t - 11.125) * (2 * pi / 4.5)) / 2 + 1
    )
)

LINEAR: EasingFunction = EASING_FUNCTIONS['linear_in']
LVALUE: EasingFunction = lambda _: 0
RVALUE: EasingFunction = lambda _: 1


def easing_with_range(f: EasingFunction, left: float, right: float) -> EasingFunction:
    left, right = max(0.0, min(1.0, left)), max(0.0, min(1.0, right))
    if left >= right:
        return f
    fl = f(left)
    fr = f(right)
    d = fr - fl
    if isclose(d, 0, abs_tol=1e-12):
        return LINEAR
    return lambda t: (f(left + (right - left) * t) - fl) / d


_BEZIER_SAMPLES_COUNT = 21
_BEZIER_SAMPLE_STEP = 1 / (_BEZIER_SAMPLES_COUNT - 1)
_BEZIER_PARAMETER_PREC = 1e-12
_BEZIER_ITERATIONS = 48
_SLOPE_EPS = 1e-7


def cubic_rev_bezier(x1: float, y1: float, x2: float, y2: float) -> EasingFunction:
    f = lambda a, b: ((a - b) * 3 + 1, b * 3 - a * 6, a * 3)
    a1, a2, a3 = f(y1, y2)
    b1, b2, b3 = f(x1, x2)

    sample_table = [
        (((b1 * i + b2) * i) + b3) * i for i in (j * _BEZIER_SAMPLE_STEP for j in range(_BEZIER_SAMPLES_COUNT))
    ]

    def inner(t: float) -> float:
        if t <= 0 or t >= 1:
            return max(0.0, min(1.0, t))
        # 这里找的是 x(t) 的采样区间而不是参数 t 的等距区间
        i = max(0, min(bisect.bisect_right(sample_table, t) - 1, _BEZIER_SAMPLES_COUNT - 2))
        left, right = i * _BEZIER_SAMPLE_STEP, (i + 1) * _BEZIER_SAMPLE_STEP
        span = sample_table[i + 1] - sample_table[i]
        dist = (t - sample_table[i]) / span if span else 0.0
        tt = left + dist * _BEZIER_SAMPLE_STEP
        for _ in range(_BEZIER_ITERATIONS):
            diff = ((b1 * tt + b2) * tt + b3) * tt - t
            if diff == 0:
                break
            if diff > 0:
                right = tt
            else:
                left = tt
            slope = (b1 * 3 * tt + b2 * 2) * tt + b3
            next_tt = tt - diff / slope if slope >= _SLOPE_EPS else (left + right) / 2
            if not left < next_tt < right:
                next_tt = (left + right) / 2
            if abs(next_tt - tt) <= _BEZIER_PARAMETER_PREC:
                tt = next_tt
                break
            tt = next_tt
        return ((a1 * tt + a2) * tt + a3) * tt

    return inner


class Easing3D(Enum):
    Linear = member(partial(_easing_linear))
    CubicBezier = member(partial(_easing_cubic_bezier))
    Si = member(partial(_easing_sinus, x='si'))
    SiSi = member(partial(_easing_sinus, x='si', z='si'))
    SiSo = member(partial(_easing_sinus, x='si', z='so'))
    So = member(partial(_easing_sinus, x='so'))
    SoSo = member(partial(_easing_sinus, x='so', z='so'))
    SoSi = member(partial(_easing_sinus, x='so', z='si'))


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    #
    # x = [i / 1000 for i in range(1001)]
    # circ_in = EASING_FUNCTIONS['circ_in']
    # y = [circ_in(i) for i in x]
    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # plt.show()
    pass
