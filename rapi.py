from typing import NamedTuple, Generic, TypeVar, TypedDict, Optional
import json
import zipfile
import io
from pathlib import Path
import requests

class RTokens(NamedTuple):
    token: str
    refresh_token: str

class File(TypedDict):
    url: str

class RChartInfo(TypedDict):
    id: int
    name: str
    level: str
    difficulty: float
    charter: str
    composer: str
    illustrator: str
    description: Optional[str]
    ranked: bool
    reviewed: bool
    stable: bool
    stable_request: bool
    illustration: File
    preview: File
    file: File
    uploader: int
    created: str
    updated: str
    chart_updated: str
    tags: list[str]
    rating: Optional[float]


class RChartInfoDict(TypedDict):
    id: int
    name: str
    level: str
    difficulty: float
    charter: str
    composer: str
    illustrator: str
    description: Optional[str]
    ranked: bool
    reviewed: bool
    stable: bool
    stable_request: bool
    illustration: File
    preview: File
    file: File
    uploader: int
    created: str
    updated: str
    chart_updated: str
    tags: list[str]
    rating: Optional[float]

class RProfile(TypedDict):
    id: int
    name: str
    avatar: Optional[File]
    badge: Optional[str]
    badges: list[str]
    language: str
    bio: Optional[str]
    exp: int
    rks: float
    roles: int
    joined: str
    last_login: str

class RCollection(TypedDict):
    id: int
    cover: Optional[File]
    owner: int
    name: str
    description: str
    created: str
    updated: str
    charts: list[RChartInfo]
    public: bool

class REvent(TypedDict):
    id: int
    creator: int
    name: str
    illustration: File
    time_start: str
    time_end: str

class RMessage(TypedDict):
    id: int
    title: str
    content: str
    time: str
    read: bool
    action: Optional[str]

class RRecord(TypedDict):
    id: int
    player: int
    chart: int
    score: int
    accuracy: float
    perfect: int
    good: int
    bad: int
    miss: int
    speed: float
    max_combo: int
    full_combo: bool
    best: bool
    mods: int
    time: str
    std: Optional[float]
    std_score: Optional[float]
    
T = TypeVar('T')

class RResult(TypedDict, Generic[T]):
    count: int
    results: list[T]

class RApi:
    _BASE_URL = 'https://phira.5wyxi.com'
    _BASE_URL_LEGACY = 'https://api.phira.cn'
    session: requests.Session
    tokens: RTokens | None
    token_cache: Path
    profile: RProfile | None

    def __init__(self, token_cache_path: str, base_url: str | None = None):
        self.session = requests.Session()
        self.session.headers['Accept-Language'] = 'zh-CN'
        if base_url:
            self._BASE_URL = base_url
        self.token_cache = Path(token_cache_path)
        self.profile = None
        self.tokens = None
        if self.token_cache.exists():
            self.load_token()

    @property
    def base_url(self) -> str:
        return self._BASE_URL

    def reqwest(self, method: str, path_or_url: str, **kwargs) -> requests.Response:
        if self.tokens is None:
            raise RuntimeError('NEED LOGIN')
        url = self._BASE_URL + path_or_url if path_or_url.startswith('/') else path_or_url
        authorization = {'Authorization': f'Bearer {self.tokens.token}'}
        if 'headers' in kwargs:
            kwargs['headers'] |= authorization
        else:
            kwargs['headers'] = authorization
        return self.session.request(method, url, **kwargs)

    def req(self, method: str, path: str, **kwargs) -> requests.Response:
        url = self._BASE_URL + path
        return self.session.request(method, url, **kwargs)

    def load_token(self):
        with self.token_cache.open('r') as tin:
            token = json.load(tin)
            self.tokens = RTokens(token['token'], token['refresh_token'])

    def save_token(self):
        if self.tokens is None:
            return
        with self.token_cache.open('w') as out:
            json.dump({'token': self.tokens.token, 'refresh_token': self.tokens.refresh_token}, out)

    def login(self, email: str, password: str) -> None:
        result = self.session.post(self._BASE_URL + '/login', json={'email': email, 'password': password}).json()
        if 'error' in result:
            raise RuntimeError(f'Login failed: {result["error"]}')
        self.tokens = RTokens(result['token'], result['refreshToken'])
        self.save_token()

    def register(self, email: str, username: str, password: str) -> None:
        result = self.session.post(self._BASE_URL + '/register', json={
            'email': email,
            'name': username,
            'password': password,
        }).json()
        if 'error' in result:
            raise RuntimeError(f'Register failed: {result["error"]}')

    def me(self) -> dict:
        return self.reqwest('GET', '/me').json()

    def download_bytes(self, url: str) -> bytes:
        res = self.reqwest('GET', url)
        if res.status_code != 200:
            raise RuntimeError(f'Download Failed: HTTP {res.status_code}')
        return res.content

    def download_file(self, url: str, save_path: Path) -> None:
        res = self.reqwest('GET', url, stream=True)
        if res.status_code != 200:
            raise RuntimeError(f'Download Failed: HTTP {res.status_code}')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)
    
    def chart_search(self, query: dict | None = None) -> RResult[RChartInfo]:
        if query is None:
            query = {}
        if 'page' not in query:
            query['page'] = 1
        return self.session.get(self._BASE_URL + '/chart', params=query).json()

    def chart_search_ranked(self, query: dict | None = None) -> RResult[RChartInfo]:
        return self.chart_search(query)

    def chart_search_special(self, query: dict | None = None) -> RResult[RChartInfo]:
        return self._chart_search_with_suffix('/special', query)

    def chart_search_unreviewed(self, query: dict | None = None) -> RResult[RChartInfo]:
        return self._chart_search_with_suffix('/unreviewed', query)

    def chart_search_popular(self, query: dict | None = None) -> RResult[RChartInfo]:
        if query is None:
            query = {}
        query['order'] = 'plays'
        return self.chart_search(query)

    def _chart_search_with_suffix(self, suffix: str, query: dict | None = None) -> RResult[RChartInfo]:
        if query is None:
            query = {}
        if 'page' not in query:
            query['page'] = 1
        return self.session.get(self._BASE_URL + f'/chart{suffix}', params=query).json()

    def chart_info(self, id: int) -> RChartInfoDict:
        return self.session.get(f'{self._BASE_URL}/chart/{id}').json()

    def chart_multi_get(self, ids: list[int]) -> list[RChartInfo]:
        ids_str = ','.join(str(i) for i in ids)
        return self.reqwest('GET', f'/chart/multi-get?ids={ids_str}').json()

    def chart_rate(self, id: int, score: int) -> dict:
        return self.reqwest('POST', f'/chart/{id}/rate', json={'score': score}).json()

    def chart_get_rate(self, id: int) -> dict:
        return self.reqwest('GET', f'/chart/{id}/rate').json()

    def chart_edit_tags(self, id: int, tags: list[str]) -> dict:
        return self.reqwest('POST', f'/chart/{id}/edit-tags', json={'tags': tags}).json()

    def chart_delete(self, id: int) -> None:
        self.reqwest('DELETE', f'/chart/{id}')

    def chart_upload(self, file_id: str, chart_id: int | None = None) -> dict:
        data = {'file': file_id}
        return self.reqwest('POST', '/chart/upload', json=data).json()

    def chart_review(self, id: int, approve: bool, reason: str = '') -> dict:
        return self.reqwest('POST', f'/chart/{id}/review', json={
            'approve': approve,
            'reason': reason,
        }).json()

    def chart_stabilize(self, id: int, kind: int, reason: str = '') -> dict:
        return self.reqwest('POST', f'/chart/{id}/stabilize', json={
            'kind': kind,
            'reason': reason,
        }).json()

    def chart_req_stabilize(self, id: int) -> None:
        self.reqwest('POST', f'/chart/{id}/req-stabilize', json={})

    def chart_verify_cksum(self, id: int, checksum: str) -> dict:
        return self.reqwest('GET', f'/chart/{id}/verify-cksum?checksum={checksum}').json()

    def chart_stabilize_comment(self, id: int, comment: str) -> None:
        self.reqwest('POST', f'/chart/{id}/stabilize-comment', json={'comment': comment})

    def upload_file(self, name: str, data: bytes) -> str:
        res = self.reqwest('POST', f'/upload/{name}', data=data).json()
        return res['id']

    def record(self, query: dict | None = None) -> dict:
        if query is None:
            query = {}
        if 'page' not in query:
            query['page'] = 1
        return self.session.get(self._BASE_URL + '/record', params=query).json()

    def record_best(self, chart_id: int) -> dict:
        return self.reqwest('GET', f'/record/best/{chart_id}').json()

    def record_list15(self, chart_id: int, std: bool = False) -> dict:
        return self.reqwest('GET', f'/record/list15/{chart_id}', params={'std': std}).json()

    def play_upload(self, data: dict) -> dict:
        return self.reqwest('POST', '/play/upload', json=data).json()

    def collection_get(self, collection_id: int) -> RCollection:
        return self.reqwest('GET', f'/collection/{collection_id}').json()

    def collection_create(self, name: str, description: str, chart_ids: list[int], public: bool = False) -> RCollection:
        return self.reqwest('PUT', f'/collection', json={
            'name': name,
            'description': description,
            'charts': chart_ids,
            'public': public,
        }).json()

    def collection_update_charts(self, collection_id: int, chart_ids: list[int]) -> RCollection:
        return self.reqwest('PATCH', f'/collection/{collection_id}', json={'charts': chart_ids}).json()

    def collection_set_public(self, collection_id: int, public: bool) -> None:
        self.reqwest('PATCH', f'/collection/{collection_id}', json={'public': public})

    def collection_set_cover(self, collection_id: int, chart_id: int) -> None:
        self.reqwest('PATCH', f'/collection/{collection_id}', json={'cover': chart_id})

    def collection_delete(self, collection_id: int) -> None:
        self.reqwest('DELETE', f'/collection/{collection_id}')

    def collection_like(self, collection_id: int, like: bool) -> dict:
        return self.reqwest('POST', f'/collection/{collection_id}/like', json={'like': like}).json()

    def collection_get_like(self, collection_id: int) -> dict:
        return self.reqwest('GET', f'/collection/{collection_id}/like').json()

    def event_uml(self, event_id: int, version: str = '0.0.0') -> dict:
        return self.reqwest('GET', f'/event/{event_id}/uml', params={'version': version}).json()

    def event_status(self, event_id: int) -> dict:
        return self.reqwest('GET', f'/event/{event_id}/status').json()

    def event_list15(self, event_id: int) -> dict:
        return self.reqwest('GET', f'/event/{event_id}/list15').json()

    def event_join(self, event_id: int) -> None:
        self.reqwest('POST', f'/event/{event_id}/join', json={})

    def message_has_new(self, checked: str) -> dict:
        return self.reqwest('GET', '/message/has_new', params={'checked': checked}).json()

    def message_list(self, before: str | None = None) -> list[RMessage]:
        params = {}
        if before:
            params['before'] = before
        return self.reqwest('GET', '/message/list', params=params).json()

    def user_info(self, user_id: int) -> RProfile:
        return self.reqwest('GET', f'/user/{user_id}').json()

    def me_char(self, locale: str = 'zh-CN') -> dict:
        return self.reqwest('GET', '/me/char', params={'locale': locale}).json()

    def fetch_terms(self, locale: str = 'zh-CN') -> str:
        res = self.session.get(f'{self._BASE_URL}/terms/{locale}.txt')
        if res.status_code != 200:
            raise RuntimeError(f'Get Terms Failed: HTTP {res.status_code}')
        return res.text

    def refresh_token(self) -> None:
        if not self.tokens:
            return
        result = self.session.post(self._BASE_URL + '/login', json={'refreshToken': self.tokens.refresh_token}).json()
        if 'error' in result:
            raise RuntimeError(f'Refresh Token Failed: {result["error"]}')
        self.tokens = RTokens(result['token'], result['refreshToken'])
        self.save_token()

    def download_chart(self, chart_id: int, save_dir: str | Path = 'Assets/Rchart', chart_only: bool = True) -> str:
        save_dir = Path(save_dir)
        info = self.chart_info(chart_id)
        print(f'Chart name: {info["name"]}')
        print(f'Chart difficulty: {info["level"]} ({info["difficulty"]})')
        file_obj = info['file']
        if isinstance(file_obj, dict):
            file_url = file_obj['url']
        else:
            file_url = file_obj
        print(f'Downloading Chart...')
        chart_data = self.download_bytes(file_url)
        print(f'Size: {len(chart_data)} bytes')
        if chart_only:
            output_path = save_dir / f'{chart_id}.json'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(chart_data)) as zf:
                json_files = [name for name in zf.namelist() 
                             if not name.endswith('/') and 
                             Path(name).suffix.lower() in {'.json', '.yml', '.yaml'}]
                if not json_files:
                    raise RuntimeError('No JSON/YAML file')
                json_file = next((f for f in json_files if f.lower().endswith('.json')), json_files[0])
                with zf.open(json_file) as src, open(output_path, 'wb') as dst:
                    dst.write(src.read())
                print(f'  Extract: {json_file}')
            return str(output_path)
        else:
            output_dir = save_dir / str(chart_id)
            output_dir.mkdir(parents=True, exist_ok=True)
            for f in output_dir.iterdir():
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    for sub in f.iterdir():
                        sub.unlink()
                    f.rmdir()
            chart_extensions = {'.rpe', '.pec', '.pgr', '.json', '.yml', '.yaml'}
            asset_extensions = {'.mp3', '.ogg', '.wav', '.png', '.jpg', '.jpeg', '.webp'}
            all_allowed = chart_extensions | asset_extensions
            with zipfile.ZipFile(io.BytesIO(chart_data)) as zf:
                for entry in zf.infolist():
                    if entry.is_dir():
                        continue
                    filename = Path(entry.filename).name
                    if filename.startswith('.'):
                        continue
                    ext = Path(entry.filename).suffix.lower()
                    if ext in all_allowed:
                        target_path = output_dir / filename
                        if target_path.exists():
                            parts = Path(entry.filename).parts
                            if len(parts) > 1:
                                target_path = output_dir / ('_'.join(parts))
                            else:
                                target_path = output_dir / f'{filename}'
                        with zf.open(entry) as src, open(target_path, 'wb') as dst:
                            dst.write(src.read())
                        print(f'  Extract: {filename}')
            return str(output_dir)

if __name__ == '__main__':
    import sys
    import os

    token_cache = os.path.join(os.path.dirname(__file__), 'token.json')
    api = RApi(token_cache)

    if not api.tokens:
        email = input('Email: ').strip()
        password = input('Password: ').strip()
        try:
            api.login(email, password)
            print('Login Success')
        except RuntimeError as e:
            print(f'Login Failed: {e}')
            sys.exit(1)
    while True:
        chart_id_str = input('Chart ID: ').strip()
        if chart_id_str.lower() in ('q', 'quit', 'exit'):
            break
        try:
            chart_id = int(chart_id_str)
        except ValueError:
            print('N/A')
            continue
        try:
            path = api.download_chart(chart_id)
        except Exception as e:
            pass
