"""Accept scalar Muse-Glimmer sliding-window periods in SGLang 0.5.18."""

from numbers import Integral

import gguf

_Reader = gguf.GGUFReader
_ARCHITECTURE_KEY = "general.architecture"
_PATTERN_KEY = "muse-glimmer.attention.sliding_window_pattern"
_BLOCK_COUNT_KEY = "muse-glimmer.block_count"
_EXPECTED_BLOCK_COUNT = 52
_EXPECTED_PERIOD = 4


class _ExpandedField:
    def __init__(self, field, value):
        self._field = field
        self._value = value

    def contents(self):
        return self._value

    def __getattr__(self, name):
        return getattr(self._field, name)


class _CompatReader:
    def __init__(self, *args, **kwargs):
        self._reader = _Reader(*args, **kwargs)
        self.fields = self._reader.fields
        pattern_field = self.fields.get(_PATTERN_KEY)
        block_count_field = self.fields.get(_BLOCK_COUNT_KEY)
        if pattern_field is None or block_count_field is None:
            return
        period = pattern_field.contents()
        if not isinstance(period, Integral):
            return
        architecture_field = self.fields.get(_ARCHITECTURE_KEY)
        if (
            architecture_field is None
            or architecture_field.contents() != "muse-glimmer"
        ):
            raise ValueError("sliding-window period belongs to a non-Muse-Glimmer GGUF")
        period = int(period)
        block_count = int(block_count_field.contents())
        if period != _EXPECTED_PERIOD or block_count != _EXPECTED_BLOCK_COUNT:
            raise ValueError(
                "unexpected Muse-Glimmer layout: "
                f"period={period}, block_count={block_count}"
            )
        expanded = [(layer + 1) % period != 0 for layer in range(block_count)]
        if expanded[:8] != [True, True, True, False] * 2:
            raise AssertionError("invalid Muse-Glimmer sliding-window expansion")
        self.fields = dict(self.fields)
        self.fields[_PATTERN_KEY] = _ExpandedField(pattern_field, expanded)

    def __getattr__(self, name):
        return getattr(self._reader, name)


gguf.GGUFReader = _CompatReader
