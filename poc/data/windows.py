"""Saved source offsets and model-local positions for fixed prediction windows."""
from dataclasses import dataclass
import hashlib
import random
import re


@dataclass(frozen=True)
class TokenDocument:
    document_id: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class WindowPolicy:
    context_tokens: int = 512
    max_windows: int = 4
    seed: int = 17

    def __post_init__(self):
        if self.context_tokens != 512 or not 1 <= self.max_windows <= 4:
            raise ValueError('primary windows require context512, at most4 windows')


@dataclass(frozen=True)
class WindowRecord:
    document_id: str
    source_start: int
    token_ids: tuple[int, ...]
    target_indices: tuple[int, ...]
    position_ids: tuple[int, ...]


def _window(document, start, end, target_start):
    tokens = document.token_ids[start:end]
    return WindowRecord(document.document_id,start,tokens,tuple(range(max(1,target_start-start),len(tokens))),tuple(range(len(tokens))))


def make_windows(document: TokenDocument, policy: WindowPolicy) -> list[WindowRecord]:
    n,s = len(document.token_ids),policy.context_tokens
    if n == 0: return []
    if n < s: return [_window(document,0,n,1)]
    rng = random.Random(int(hashlib.sha256(f'{policy.seed}:{document.document_id}'.encode()).hexdigest(),16))
    offset = rng.randrange(n % s + 1)
    starts = list(range(offset,n-s+1,s))
    selected = sorted(rng.sample(starts,min(policy.max_windows,len(starts))))
    return [_window(document,start,start+s,start+1) for start in selected]


def rolling_windows(document: TokenDocument, *, context_tokens=512, stride=256) -> list[WindowRecord]:
    if context_tokens < 2 or not 0 < stride < context_tokens:
        raise ValueError('rolling stride must be positive and below context')
    n = len(document.token_ids)
    if not n: return []
    result, previous_end, start = [],0,0
    while True:
        end = min(start+context_tokens,n)
        result.append(_window(document,start,end,max(1,previous_end)))
        if end == n: break
        previous_end = end
        start += stride
    return result


def parse_wikitext_articles(lines) -> list[str]:
    # Match one '=' at each edge only; subsections are not article boundaries.
    heading = re.compile(r'^\s*=\s+[^=\r\n].*?\s+=\s*$')
    articles, current, seen_heading = [],[],False
    for line in lines:
        if heading.match(line):
            if seen_heading:
                articles.append(''.join(current)); current=[]
            seen_heading=True
        current.append(line)
    if current: articles.append(''.join(current))
    return articles
