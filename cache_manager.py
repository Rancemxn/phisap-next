from pathlib import Path
import hashlib


class CacheManager:
    _CACHE_VERSION = b'chart-v2\0'

    def __init__(self, cache_dir: str | Path = './.cache.d') -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _calc_hash(self, content: str | bytes) -> str:
        if isinstance(content, str):
            content = content.encode()
        return hashlib.sha512(self._CACHE_VERSION + content).hexdigest()

    def has_cache(self, content: str) -> bool:
        content_hash = self._calc_hash(content)
        cache_file = self.cache_dir / f'{content_hash}.psap'
        return cache_file.exists()

    def find_cache_for_content(self, content: str) -> bytes | None:
        content_hash = self._calc_hash(content)
        cache_file = self.cache_dir / f'{content_hash}.psap'
        if cache_file.exists():
            with cache_file.open('rb') as f:
                return f.read()

    def find_cache_for_file(self, file: str | Path) -> bytes | None:
        return self.find_cache_for_content(Path(file).read_text(encoding='utf-8-sig'))

    def write_cache_of_chart(self, chart: str | Path, cache: bytes):
        self.write_cache_of_content(Path(chart).read_text(encoding='utf-8-sig'), cache)

    def write_cache_of_content(self, content: str, cache: bytes):
        content_hash = self._calc_hash(content)
        with (self.cache_dir / f'{content_hash}.psap').open('wb') as out:
            out.write(cache)


__all__ = ['CacheManager']
