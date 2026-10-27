"""Versioned duplicate normalization, MinHash screen, exact verified components."""
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import re
import unicodedata

from poc.data.manifest import ComponentManifest, DataManifest, DocumentRecord
from poc.records import canonical_json


@dataclass(frozen=True)
class DedupPolicy:
    threshold: float = .8
    shingle_words: int = 5
    permutations: int = 64
    band_rows: int = 2
    seed: int = 17
    exhaustive: bool = False

    def __post_init__(self):
        if not 0 < self.threshold <= 1 or self.shingle_words != 5 or self.permutations < 1 or self.permutations % self.band_rows:
            raise ValueError('invalid dedup policy')


def normalize(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def shingles(text: str, n: int = 5) -> frozenset[str]:
    words = normalize(text).split()
    return frozenset(hashlib.sha256(' '.join(words[i:i+n]).encode()).hexdigest()[:32]
                     for i in range(max(0,len(words)-n+1)))


def fingerprint(record: DocumentRecord) -> dict:
    text = record.read_text()
    return {'normalized_sha256':hashlib.sha256(normalize(text).encode()).hexdigest(),
            'shingles':sorted(shingles(text)), 'url':record.url, 'article_id':record.article_id}


def fingerprint_overlap(a: dict, b: dict, threshold: float) -> bool:
    if (a['normalized_sha256'] == b['normalized_sha256'] or
            a.get('url') and a.get('url') == b.get('url') or
            a.get('article_id') and a.get('article_id') == b.get('article_id')):
        return True
    x,y = set(a['shingles']),set(b['shingles'])
    return bool(x and y) and len(x & y) / len(x | y) >= threshold


def _candidate_pairs(fps, policy):
    if policy.exhaustive:
        return {(a,b) for a in range(len(fps)) for b in range(a+1,len(fps))}
    # Seeded affine permutations over a large prime; signatures never determine edges.
    import random
    rng = random.Random(policy.seed)
    prime = (1 << 61)-1
    permutations = [(rng.randrange(1,prime),rng.randrange(prime)) for _ in range(policy.permutations)]
    buckets = defaultdict(list)
    pairs = set()
    for i,fp in enumerate(fps):
        values = [int(s,16) % prime for s in fp['shingles']]
        if not values:  # short texts group by exact hash/identity only
            continue
        signature = tuple(min((a*x+b)%prime for x in values) for a,b in permutations)
        for start in range(0,policy.permutations,policy.band_rows):
            key = (start, signature[start:start+policy.band_rows])
            pairs.update((j,i) for j in buckets[key])
            buckets[key].append(i)
    return pairs


def build_components(records, policy: DedupPolicy) -> ComponentManifest:
    records = tuple(sorted(records,key=lambda r:r.document_id))
    if len({r.document_id for r in records}) != len(records):
        raise ValueError('duplicate document IDs')
    fps = [fingerprint(r) for r in records]
    parent = list(range(len(records)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(a,b):
        a,b = root(a),root(b)
        parent[max(a,b)] = min(a,b)
    for key in ('normalized_sha256','url','article_id'):
        seen = {}
        for i,fp in enumerate(fps):
            value = fp[key]
            if not value:
                continue
            if value in seen:
                join(i,seen[value])
            seen[value] = i
    candidates = _candidate_pairs(fps,policy)
    edges = []
    for a,b in sorted(candidates):
        x,y = set(fps[a]['shingles']),set(fps[b]['shingles'])
        if x and y and len(x & y) / len(x | y) >= policy.threshold:
            join(a,b)
            edges.append((records[a].document_id, records[b].document_id))
    groups = defaultdict(list)
    for i,r in enumerate(records):
        groups[root(i)].append(r.document_id)
    identities = {k:hashlib.sha256(canonical_json({'documents':v})).hexdigest() for k,v in groups.items()}
    membership = {r.document_id:identities[root(i)] for i,r in enumerate(records)}
    return ComponentManifest(records,membership,tuple(edges), {**asdict(policy),'normalization':'NFKC-casefold-whitespace-v1'},len(candidates))


def assign_roles(components: ComponentManifest, salt: str) -> DataManifest:
    if not salt:
        raise ValueError('role salt required')
    assignment = {}
    for component in set(components.membership.values()):
        value = int(hashlib.sha256(canonical_json({'salt':salt,'component':component})).hexdigest(),16)/2**256
        assignment[component] = ('fit' if value < .7 else 'probe_fit' if value < .75 else
                                 'probe_score' if value < .8 else 'dev' if value < .9 else
                                 'confirm_A' if value < .95 else 'confirm_B')
    return DataManifest(components.records,{i:assignment[c] for i,c in components.membership.items()},components.membership,salt)


def quarantine_overlap(manifest: DataManifest, benchmark_records, policy: DedupPolicy) -> DataManifest:
    """Exact inverted-index cross-source screen: no approximate-screen misses here."""
    natural = [fingerprint(r) for r in manifest.records]
    index = defaultdict(set)
    hashes, urls, articles = defaultdict(set),defaultdict(set),defaultdict(set)
    for i,fp in enumerate(natural):
        hashes[fp['normalized_sha256']].add(i)
        if fp['url']: urls[fp['url']].add(i)
        if fp['article_id']: articles[fp['article_id']].add(i)
        for s in fp['shingles']: index[s].add(i)
    hit_components = set()
    for r in benchmark_records:
        fp = fingerprint(r)
        candidates = set(hashes[fp['normalized_sha256']])
        candidates.update(urls.get(fp['url'],()))
        candidates.update(articles.get(fp['article_id'],()))
        for s in fp['shingles']: candidates.update(index.get(s,()))
        for i in candidates:
            if fingerprint_overlap(natural[i],fp,policy.threshold):
                hit_components.add(manifest.membership[manifest.records[i].document_id])
    quarantine = {i:'cross-source benchmark overlap' for i,c in manifest.membership.items() if c in hit_components}
    # Entire natural component excluded, including potential confirmation members;
    # benchmark official roles remain intact.
    return replace(manifest, roles={i:'quarantine' if i in quarantine else v for i,v in manifest.roles.items()},quarantine=quarantine)
