from poc.data.windows import TokenDocument, WindowPolicy, make_windows, rolling_windows, parse_wikitext_articles


def test_natural_nonoverlap_starts_positions_and_short():
    d = TokenDocument('owned', tuple(range(5201)))
    w = make_windows(d, WindowPolicy(seed=17))
    assert w == make_windows(d, WindowPolicy(seed=17))
    assert len(w) == 4 and all(len(x.token_ids) == 512 for x in w)
    assert all(x.position_ids == tuple(range(len(x.token_ids))) for x in w)
    starts = sorted(x.source_start for x in w)
    assert all(b-a >= 512 for a,b in zip(starts,starts[1:]))
    assert all(x.token_ids[0] == x.source_start for x in w)
    short = make_windows(TokenDocument('short',(5,6,7)),WindowPolicy())
    assert len(short) == 1 and short[0].target_indices == (1,2)
    assert make_windows(TokenDocument('empty',()),WindowPolicy()) == []


def test_rolling_counts_each_real_target_once_including_eos():
    d = TokenDocument('article', tuple(range(1100)) + (99999,))
    windows = rolling_windows(d)
    targets = [w.source_start+i for w in windows for i in w.target_indices]
    assert sorted(targets) == list(range(1,len(d.token_ids)))
    assert len(targets) == len(set(targets))
    assert windows[-1].token_ids[-1] == 99999
    assert all(len(w.token_ids)<=512 for w in windows)
    assert all(w.position_ids[0] == 0 for w in windows)


def test_wikitext_top_level_boundaries_preserve_all_original_content():
    lines = ['\n', ' = One = \n','alpha\n',' == Subsection == \n','beta\n',' = Two = \n','omega\n']
    articles = parse_wikitext_articles(lines)
    assert len(articles) == 2
    assert ''.join(articles) == ''.join(lines)
    assert ' == Subsection == \n' in articles[0]
