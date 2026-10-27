"""Outcome-independent benchmark split policies; no prediction or answer scoring."""
import hashlib
import math


def official_role(split: str) -> str:
    if split == 'test': return 'protected_benchmark'
    if split in {'train','validation'}: return 'dev'
    raise ValueError('unknown official split')


def blimp_roles(paradigms: dict[str, list[str]], salt: str) -> dict[tuple[str,str],str]:
    roles = {}
    for paradigm, ids in sorted(paradigms.items()):
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate BLiMP item IDs')
        ranked = sorted(ids,key=lambda i:hashlib.sha256(f'{salt}:{paradigm}:{i}'.encode()).digest())
        dev = set(ranked[:math.floor(len(ids)*.1)])
        roles.update({(paradigm,i):'dev' if i in dev else 'protected_benchmark' for i in ids})
    return roles
