from twinkle.infra._ray.ray_helper import _get_node_local_topology


def test_get_node_local_topology_for_single_node_workers():
    placements = [
        {'node_rank': 0},
        {'node_rank': 0},
    ]

    assert _get_node_local_topology(placements) == [
        (0, [0, 1]),
        (1, [0, 1]),
    ]


def test_get_node_local_topology_does_not_assume_contiguous_global_ranks():
    placements = [
        {'node_rank': 0},
        {'node_rank': 1},
        {'node_rank': 0},
    ]

    assert _get_node_local_topology(placements) == [
        (0, [0, 2]),
        (0, [1]),
        (1, [0, 2]),
    ]
