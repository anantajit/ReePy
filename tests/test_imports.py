import reepy


def test_import():
    reepy.ReebGraph(lambda point: 0, lambda points: True, lambda a, b: True)
