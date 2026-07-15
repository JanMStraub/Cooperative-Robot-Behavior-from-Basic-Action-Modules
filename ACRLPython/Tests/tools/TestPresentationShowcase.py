import os
import struct
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.dirname(test_dir)
acrlpython_dir = os.path.dirname(tests_dir)
sys.path.insert(0, acrlpython_dir)

from tools.PresentationShowcase import _build_sequence_message


class TestBuildSequenceMessage:

    def test_includes_trailing_flags_len_field(self):
        msg = _build_sequence_message(
            "move to (0, 0, 0)", "Robot1", "TableStereoCamera", 1
        )

        offset = 5

        cmd_len = struct.unpack_from("<I", msg, offset)[0]
        offset += 4 + cmd_len

        robot_len = struct.unpack_from("<I", msg, offset)[0]
        offset += 4 + robot_len

        camera_len = struct.unpack_from("<I", msg, offset)[0]
        offset += 4 + camera_len

        offset += 1

        remaining = len(msg) - offset
        assert remaining >= 4

        flags_len = struct.unpack_from("<I", msg, offset)[0]
        offset += 4
        assert len(msg) - offset == flags_len
