from v2ecoli.workflow.analysis_runner import scale_history_sql

_FROM = "read_parquet(['x.pq'], hive_partitioning=true)"


def test_single_sql_filters_full_cell():
    sql = scale_history_sql("single", _FROM, (0, 1, 2, "00"))
    assert "variant = 0" in sql and "lineage_seed = 1" in sql
    assert "generation = 2" in sql and "agent_id = '00'" in sql


def test_multiseed_sql_filters_variant_only():
    sql = scale_history_sql("multiseed", _FROM, (3,))
    assert "variant = 3" in sql
    assert "lineage_seed" not in sql and "agent_id" not in sql


def test_multivariant_sql_is_unfiltered():
    sql = scale_history_sql("multivariant", _FROM, ())
    assert "WHERE" not in sql.upper()
    assert _FROM in sql
