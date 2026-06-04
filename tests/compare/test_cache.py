from scripts._compare.cache import cache_key, is_stale, mark_done


def test_cache_key_is_deterministic_and_sensitive():
    a = cache_key({"generations": 2}, commit="abc", mode="full")
    b = cache_key({"generations": 2}, commit="abc", mode="full")
    c = cache_key({"generations": 3}, commit="abc", mode="full")
    d = cache_key({"generations": 2}, commit="def", mode="full")
    assert a == b
    assert a != c
    assert a != d
    assert len(a) == 16  # short hex digest


def test_is_stale_true_when_marker_missing(tmp_path):
    assert is_stale(tmp_path / "nope") is True


def test_is_stale_false_when_done_marker_present(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / ".done").write_text("ok")
    assert is_stale(d) is False


def test_is_stale_false_when_token_matches(tmp_path):
    d = tmp_path / "run"
    mark_done(d, token="abc123")
    assert is_stale(d, token="abc123") is False


def test_is_stale_true_when_token_differs(tmp_path):
    d = tmp_path / "run"
    mark_done(d, token="abc123")
    assert is_stale(d, token="xyz999") is True


def test_mark_done_default_token(tmp_path):
    d = tmp_path / "run"
    mark_done(d)
    assert (d / ".done").read_text() == "ok"
    assert is_stale(d) is False
