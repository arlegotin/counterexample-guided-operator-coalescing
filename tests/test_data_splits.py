from collections import Counter
from dataclasses import replace

import pytest

from poc.data.manifest import DocumentRecord, ExposureLedger, ExposureError
from poc.data.dedup import DedupPolicy, build_components, assign_roles, quarantine_overlap, normalize, shingles
from poc.data.benchmarks import blimp_roles, official_role


def doc(i, text, **kwargs):
    return DocumentRecord(str(i), 'owned', 'a'*40, 'fixture', i, text=text, **kwargs)


def test_exact_url_short_and_empty_components():
    records = [doc(0, '  HéLLo  world '), doc(1, 'héllo world'), doc(2, 'changed', url='https://owned/a'),
               doc(3, 'entirely different', url='https://owned/a'), doc(4, ''), doc(5, 'unrelated')]
    c = build_components(records, DedupPolicy())
    assert c.membership['0'] == c.membership['1']
    assert c.membership['2'] == c.membership['3']
    assert c.membership['4'] != c.membership['5']
    assert normalize('ＦＯＯ\n BAR') == 'foo bar'
    assert shingles('a b') == frozenset()


def test_exact_jaccard_transitive_chain():
    # A~B and B~C pass .8; A~C does not. All must form one component.
    words = [f'w{i}' for i in range(40)]
    a = ' '.join(words)
    b = ' '.join(words[3:] + ['x1','x2','x3'])
    c = ' '.join(words[6:] + ['x1','x2','x3','x4','x5','x6'])
    result = build_components([doc(0,a),doc(1,b),doc(2,c)], DedupPolicy(threshold=.8, exhaustive=True))
    assert len(set(result.membership.values())) == 1
    assert ('0','2') not in result.near_edges
    assert result.near_edges == (('0','1'),('1','2'))


def test_roles_component_integrity_and_proportions():
    records = [doc(i, f'unique short {i}') for i in range(10000)]
    records.append(doc(10000, 'unique short 0'))
    c = build_components(records, DedupPolicy())
    a = assign_roles(c, 'owned-salt')
    b = assign_roles(build_components(reversed(records), DedupPolicy()), 'owned-salt')
    assert a.roles == b.roles
    assert a.roles['0'] == a.roles['10000']
    counts = Counter(a.roles.values())
    for role, fraction in [('fit',.7),('probe_fit',.05),('probe_score',.05),('dev',.1),('confirm_A',.05),('confirm_B',.05)]:
        assert abs(counts[role] / 10001 - fraction) < .015


def test_cross_source_overlap_quarantines_entire_component():
    r = [doc(0, 'the old oak tree grows in the quiet field'), doc(1, 'changed', url='https://owned/x'),
         doc(2, 'the old oak tree grows in the quiet field', url='https://owned/x')]
    manifest = assign_roles(build_components(r, DedupPolicy()), 'salt')
    q = quarantine_overlap(manifest, [doc(9, 'the old oak tree grows in the quiet field')], DedupPolicy())
    assert set(q.roles.values()) == {'quarantine'}


def test_benchmark_roles_fixed_before_outcomes():
    pairs = {'paradigm_a': [str(i) for i in range(1000)], 'paradigm_b': [str(i) for i in range(1000)]}
    roles = blimp_roles(pairs, 'fixed')
    assert roles == blimp_roles(pairs, 'fixed')
    for p in pairs:
        assert sum(roles[p, i] == 'dev' for i in pairs[p]) == 100
    assert official_role('test') == 'protected_benchmark'
    assert official_role('validation') == official_role('train') == 'dev'
    with pytest.raises(ValueError): official_role('unknown')


def test_exposure_rebuild_rejected_even_if_component_ids_change(tmp_path):
    r = [doc(0, 'exposed original document has enough words for shingles')]
    old = assign_roles(build_components(r, DedupPolicy()), 'old')
    ledger = ExposureLedger(tmp_path / 'exposure.jsonl')
    ledger.record(old, ['0'], purpose='fit')
    # New ID, altered original content, but normalization-equivalent content cannot become protected.
    r2 = [doc(999, ' EXPOSED original document has enough words for shingles '), doc(1, 'bridge', url='https://owned/bridge')]
    new = assign_roles(build_components(r2, DedupPolicy()), 'new')
    new = replace(new, roles={i:'confirm_A' for i in new.roles})
    with pytest.raises(ExposureError): ledger.validate_rebuild(new, DedupPolicy())
    ledger.validate_rebuild(replace(new, roles={i:'fit' for i in new.roles}), DedupPolicy())


def test_small_fixture_minhash_screen_matches_exhaustive_verified_components():
    base=[f'word{i}' for i in range(50)]
    records=[doc(i,' '.join(base[i:i+45])) for i in range(6)]
    a=build_components(records,DedupPolicy())
    b=build_components(records,DedupPolicy(exhaustive=True))
    assert a.membership==b.membership
    assert a.near_edges==b.near_edges


def test_roles_are_immutable_snapshots_of_inputs():
    from poc.data.manifest import DataManifest
    roles={'0':'confirm_A'}; membership={'0':'c'}
    m=DataManifest((doc(0,'owned'),),roles,membership,'salt')
    roles['0']='fit'
    assert m.roles['0']=='confirm_A'
    with pytest.raises(TypeError): m.roles['0']='fit'


def test_exposure_ledger_carries_full_component_and_identity_on_rebuild(tmp_path):
    from poc.data.manifest import DataManifest
    first=doc(0,'first original words have a shared identity',url='https://owned/shared')
    sibling=doc(1,'unread sibling uses completely different words',url='https://owned/shared')
    old=DataManifest((first,sibling),{'0':'fit','1':'fit'},{'0':'oldcomponent','1':'oldcomponent'},'salt')
    ledger=ExposureLedger(tmp_path/'exposures.jsonl')
    ledger.record(old,['0'],purpose='fit')
    changed=doc(0,'fully changed content without the old url')
    new=DataManifest((changed,),{'0':'confirm_A'},{'0':'newcomponent'},'new')
    with pytest.raises(ExposureError):ledger.validate_rebuild(new,DedupPolicy())
    sibling_alias=doc(99,sibling.text)
    new=DataManifest((sibling_alias,),{'99':'confirm_B'},{'99':'newcomponent'},'new')
    with pytest.raises(ExposureError):ledger.validate_rebuild(new,DedupPolicy())


def test_new_duplicate_bridge_cannot_relabel_exposure_as_protected(tmp_path):
    words=[f'w{i}' for i in range(40)]
    a=' '.join(words)
    b=' '.join(words[3:]+['x1','x2','x3'])
    c=' '.join(words[6:]+['x1','x2','x3','x4','x5','x6'])
    old=assign_roles(build_components([doc(0,a)],DedupPolicy()),'old')
    ledger=ExposureLedger(tmp_path/'exposures.jsonl');ledger.record(old,['0'],purpose='fit')
    rebuilt=assign_roles(build_components([doc(100,b),doc(101,c)],DedupPolicy(exhaustive=True)),'fresh')
    rebuilt=replace(rebuilt,roles={i:'confirm_A' for i in rebuilt.roles})
    with pytest.raises(ExposureError):ledger.validate_rebuild(rebuilt,DedupPolicy())
