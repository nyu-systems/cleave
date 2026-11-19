class CleaveError(Exception):
    pass


class SpecError(CleaveError, ValueError):
    pass


class SuperoptError(CleaveError):
    pass


class LoweringRejected(CleaveError):
    pass


class ScheduleRejected(CleaveError):
    pass


class CodegenError(CleaveError):
    pass


class CompileError(CleaveError):
    pass


class SmemExceeded(CompileError):
    pass


class ValidationError(CleaveError, AssertionError):
    """A kernel failed at the validate stage; the subclasses say how."""


class OutputMismatch(ValidationError):
    """The kernel ran and its output is finite but outside the declared tolerance (or has
    the wrong shape, dtype or number of outputs). ``measured`` holds the error per output."""

    measured = None


class NonFiniteOutput(ValidationError):
    """A NaN on either side, or an infinity the reference does not have at that entry."""

    measured = None


class LaunchCrash(ValidationError):
    """Launching the compiled kernel raised (the worker process itself survived)."""


class CandidateTimeout(CleaveError, TimeoutError):
    pass


class WorkerCrash(CleaveError):
    pass


class PagingRefusal(CleaveError, AssertionError):
    pass


class BenchmarkError(CleaveError):
    pass
