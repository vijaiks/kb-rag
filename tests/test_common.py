"""Pure-function tests for the shared helpers (no models, no network)."""
import tempfile

import lancedb

from common import make_chunk_id, sql_str, tech_filter


def test_sql_str_escapes_single_quotes():
    assert sql_str("plain.pdf") == "'plain.pdf'"
    assert sql_str("O'Reilly.pdf") == "'O''Reilly.pdf'"


def test_tech_filter_none_for_empty_input():
    assert tech_filter(None) is None
    assert tech_filter("") is None
    assert tech_filter("   ") is None


def test_tech_filter_builds_like_clause_and_escapes():
    assert tech_filter("NR") == "technology LIKE '%NR%'"
    assert tech_filter("x'; DROP") == "technology LIKE '%x''; DROP%'"


def test_chunk_id_unique_per_path_and_stable():
    same_content = "h" * 64
    a = make_chunk_id("a/x.pdf", same_content, 0)
    b = make_chunk_id("b/x.pdf", same_content, 0)
    assert a != b  # byte-identical copies at different paths must not collide
    assert a == make_chunk_id("a/x.pdf", same_content, 0)
    assert make_chunk_id("a/x.pdf", same_content, 1) != a
    assert a.endswith(":0000")


def test_filters_work_against_lancedb():
    """The generated WHERE clauses are accepted by LanceDB (vector + FTS, prefiltered)."""
    rows = [
        {
            "chunk_id": "1",
            "text": "CORESET for PDCCH in NR",
            "vector": [1.0, 0, 0, 0],
            "file_path": "a/O'Reilly.pdf",
            "technology": "NR",
        },
        {
            "chunk_id": "2",
            "text": "CORESET like region in LTE",
            "vector": [0, 1.0, 0, 0],
            "file_path": "b/lte.pdf",
            "technology": "LTE",
        },
    ]
    tbl = lancedb.connect(tempfile.mkdtemp()).create_table("chunks", data=rows)
    tbl.create_fts_index("text", replace=True)
    where = tech_filter("NR")

    vec = tbl.search([1.0, 0, 0, 0]).limit(5).where(where, prefilter=True).to_list()
    fts = tbl.search("CORESET", query_type="fts").limit(5).where(where, prefilter=True).to_list()
    assert [r["chunk_id"] for r in vec] == ["1"]
    assert [r["chunk_id"] for r in fts] == ["1"]

    hostile = tech_filter("x'; DROP")
    assert tbl.search("CORESET", query_type="fts").limit(5).where(hostile, prefilter=True).to_list() == []

    tbl.delete(f"file_path = {sql_str(chr(97) + '/O' + chr(39) + 'Reilly.pdf')}")
    assert tbl.count_rows() == 1
