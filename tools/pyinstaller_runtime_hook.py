"""Runtime settings required by the frozen Windows executable."""

# Numba cannot locate source files inside PyInstaller's one-file archive.
# Replace only its disk cache with a no-op before Librosa is imported, while
# preserving normal JIT compilation and runtime performance.
from numba.core import caching, ccallback, dispatcher
from numba.np.ufunc import ufuncbuilder, wrappers


class _FrozenNullCache(caching.NullCache):
    def __init__(self, *args, **kwargs):
        pass


caching.FunctionCache = _FrozenNullCache
ccallback.FunctionCache = _FrozenNullCache
dispatcher.FunctionCache = _FrozenNullCache
ufuncbuilder.FunctionCache = _FrozenNullCache
wrappers.GufWrapperCache = _FrozenNullCache
