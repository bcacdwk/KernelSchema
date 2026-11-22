"""Small Linux x86_64 read-only syscall profile, ONLY for CPU logic tests.

No CUDA/Triton compilation capability is claimed. All threads receive the
filter. No file-write descriptors except the controller's stdout/stderr pipes
are inherited by the worker. This is a filesystem boundary, not isolation of
hostile native code from the evaluator's Python address space.
"""
import ctypes
import errno
import os
import platform


def restrict():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("CPU seccomp profile requires Linux x86_64")
    libc = ctypes.CDLL(None, use_errno=True)
    class Filter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]
    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]
    allow, deny = 0x7fff0000, 0x00050000 | errno.EPERM
    instructions = [(0x20, 0, 0, 4), (0x15, 1, 0, 0xc000003e), (0x06, 0, 0, deny),
                    (0x20, 0, 0, 0), (0x35, 0, 1, 0x40000000), (0x06, 0, 0, deny)]
    # Block all mutation paths, process creation/exec, namespace escapes, network,
    # ptrace/process_vm, io_uring and kernel APIs that bypass ordinary open.
    blocked = (56,57,58,59,76,77,82,83,84,85,86,87,88,90,91,92,93,94,101,132,133,
               155,161,165,166,167,168,175,176,188,189,190,191,192,193,197,198,199,
               246,248,249,250,251,252,257+1,259,260,263,264,265,266,268,272,280,
               298,303,304,308,310,311,313,316,319,320,321,322,323,41,42,43,49,50,53,
               425,426,427,428,429,430,431,432,433,435,437,442,452)
    for number in sorted(set(blocked)):
        instructions.extend(((0x15, 0, 1, number), (0x06, 0, 0, deny)))
    # open flags argument 1; openat flags argument 2. Read-only opens still work.
    for number, offset in ((2, 24), (257, 32)):
        instructions.extend(((0x15, 0, 4, number), (0x20, 0, 0, offset),
                             (0x45, 0, 1, 3 | os.O_CREAT | os.O_TRUNC | os.O_APPEND | os.O_TMPFILE),
                             (0x06, 0, 0, deny), (0x06, 0, 0, allow)))
    instructions.append((0x06, 0, 0, allow))
    array = (Filter * len(instructions))(*(Filter(*v) for v in instructions))
    program = Program(len(instructions), array)
    if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.syscall(317, 1, 1, ctypes.byref(program)) != 0:
        raise OSError(ctypes.get_errno(), "cannot install seccomp TSYNC read-only profile")
    return {"profile": "local-cpu-logic-seccomp-v1", "seccomp_tsync": True, "no_new_privs": True,
            "scope": "all filesystem writes and process/network creation denied", "CUDA_support": False}


def prove_readonly(paths):
    result = []
    for path in paths:
        try:
            descriptor = os.open(path, os.O_WRONLY)
        except OSError as error:
            if error.errno not in (errno.EPERM, errno.EACCES, errno.EROFS):
                raise
            result.append({"path": str(path), "write_open_denied": True})
        else:
            os.close(descriptor)
            raise RuntimeError("evaluator write permission unexpectedly available")
    return result
