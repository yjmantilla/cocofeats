from cocofeats import add


def test_add():
    assert add(1, 2) == 3
    assert add(-1.5, 0.5) == -1.0
