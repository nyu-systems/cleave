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
    pass


class CandidateTimeout(CleaveError, TimeoutError):
    pass


class WorkerCrash(CleaveError):
    pass


class PagingRefusal(CleaveError, AssertionError):
    pass


class BenchmarkError(CleaveError):
    pass
