from dgc.data.loaders import load_birthdeath, read_timeline


def write_benchmark(tmp_path):
    """Three snapshots; A lives throughout, B dies after t=2, C is born at t=2."""
    (tmp_path / "bd.timeline").write_text("MA:1=1,2=1,3=1\nMB:1=2,2=2,\nMC:2=3,3=2\n")
    comms = {
        1: ["10 11 12", "20 21 22"],
        2: ["10 11 12", "20 21 22", "30 31"],
        3: ["10 11 12", "30 31 21"],
    }
    edges = {1: "10 11\n11 12\n20 21\n12 20\n", 2: "10 11\n30 31\n20 22\n", 3: "10 12\n21 30\n"}
    for t in (1, 2, 3):
        (tmp_path / f"bd.t{t:02d}.comm").write_text("\n".join(comms[t]) + "\n")
        (tmp_path / f"bd.t{t:02d}.edges").write_text(edges[t])


def test_read_timeline_handles_trailing_commas(tmp_path):
    write_benchmark(tmp_path)
    assert read_timeline(tmp_path / "bd.timeline") == {
        "MA": {1: 1, 2: 1, 3: 1},
        "MB": {1: 2, 2: 2},
        "MC": {2: 3, 3: 2},
    }


def test_load_all_communities(tmp_path):
    write_benchmark(tmp_path)
    snapshots, ann = load_birthdeath(tmp_path, n_communities=3, seed=0, prefix="bd")
    assert [s.t for s in snapshots] == [0, 1, 2]
    # raw IDs 10,11,12,20,21,22,30,31 -> 0..7
    assert snapshots[0].nodes == {0, 1, 2, 3, 4, 5}
    assert snapshots[0].edges == {(0, 1), (1, 2), (3, 4), (2, 3)}
    assert snapshots[2].nodes == {0, 1, 2, 4, 6, 7}
    assert ann.labels[2][4] == ann.labels[2][6]  # node 21 moved into C
    kinds = sorted((e.t, e.kind) for e in ann.events if e.kind != "continuation")
    assert kinds == [(1, "birth"), (2, "death")]


def test_subsample_is_induced_and_seeded(tmp_path):
    write_benchmark(tmp_path)
    a = load_birthdeath(tmp_path, n_communities=1, seed=3, prefix="bd")
    b = load_birthdeath(tmp_path, n_communities=1, seed=3, prefix="bd")
    assert a == b
    snapshots, ann = a
    for s in snapshots:
        assert all(u in s.nodes and v in s.nodes for u, v in s.edges)
        assert set(ann.labels[s.t]) == s.nodes
