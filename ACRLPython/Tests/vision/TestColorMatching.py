from operations.VisionOperations import color_matches


def test_color_matches():
    assert color_matches("blue", "blue")
    assert color_matches("red", "red")
    assert color_matches("green", "green")

    assert color_matches("blue_cube", "blue")
    assert color_matches("red_cube", "red")
    assert color_matches("green_cube", "green")

    assert color_matches("Blue_Cube", "blue")
    assert color_matches("RED_CUBE", "red")
    assert color_matches("blue", "BLUE")

    assert not color_matches("blue_cube", "red")
    assert not color_matches("red", "blue")
    assert not color_matches(None, "blue")
    assert not color_matches("blue", None)
