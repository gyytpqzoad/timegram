"""Checkpoints that pickled LogitNormalEts under a script's __main__ load from the package."""
import io
import pickle
import sys

from timegram import library
from timegram.compounds import LogitNormalEts


def test_main_pickled_sampler_resolves_to_the_package_class():
    joint = LogitNormalEts(True, False, None, False, [0.0, 0.0, 0.0, 0.0], [[1.0, 0, 0, 0], [0, 1.0, 0, 0],
                                                                         [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    main = sys.modules["__main__"]                           # as when trained from a script
    LogitNormalEts.__module__, main.LogitNormalEts = "__main__", LogitNormalEts
    try:
        blob = pickle.dumps(joint)
    finally:
        LogitNormalEts.__module__ = "timegram.compounds"
        del main.LogitNormalEts
    assert b"__main__" in blob
    restored = library._RelocatingUnpickler(io.BytesIO(blob)).load()
    assert type(restored) is LogitNormalEts and restored.mean == joint.mean


def test_other_main_classes_are_not_redirected():
    assert ("__main__", "Something") not in library._RELOCATED
