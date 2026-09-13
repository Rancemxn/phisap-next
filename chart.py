import json
from pathlib import Path

from basis import Chart
from pgr import PgrChart
from pec import PecChart
from rpe import RpeChart


def load_chart(content: str, ratio: tuple[int, int] = (16, 9), source: Path | None = None) -> Chart:
    content = content.lstrip('\ufeff').strip()
    if not content:
        raise ValueError('empty chart')
    if content.startswith(('{', '[')):
        data = json.loads(content)
        if not isinstance(data, dict) or not isinstance(data.get('judgeLineList'), list):
            raise ValueError('unrecognized JSON chart format')
        if 'BPMList' in data:
            chart = RpeChart(data, ratio)
        elif 'formatVersion' in data:
            chart = PgrChart(data, ratio)
        else:
            raise ValueError('unrecognized JSON chart format')
    else:
        chart = PecChart(content, ratio)
    chart.source = Path(source) if source is not None else None
    return chart
