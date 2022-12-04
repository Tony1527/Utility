import sys
import os
import time
from Utility.Constant import g_MB

try:
    import psutil
except:
    raise ModuleNotFoundError("module psutil is needed!")


try:
    from loguru import logger
except:
    raise ModuleNotFoundError("module loguru is needed!")

_log_full_print = False


def SplitDirFile(s):
    dir_end = s.rfind(os.sep)
    if dir_end != -1:
        return s[:dir_end], s[dir_end + 1 :]
    else:
        return None, s


def MKDirsToFile(s):
    path, file_name = SplitDirFile(s)
    if path != None and not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False


def MKOutput(path="."):
    path = path + os.sep + "output" + os.sep
    return MKDirsToFile(path)


def Log(level="debug", is_full_print=False):
    """
    Initialize logger
    """
    level = level.upper()
    cmd_format = "<level>{level:<8}</level> | <level>{message}</level>"
    err_id, out_id = (-1, -1)

    logger.remove()
    err_id = logger.add(sys.stderr, level=level, format=cmd_format, enqueue=True)

    _log_full_print = is_full_print

    return err_id, out_id


def AddLogFile(file_name="output/file.log", level="debug"):
    level = level.upper()
    if _log_full_print:
        file_format = "{time:YY-MM-DD HH:mm:ss} | <level>{level:<8}</level> | <level>{message}</level> "
    else:
        file_format = "<level>{level:<8}</level> | <level>{message}</level> "
    out_id = (-1, -1)
    MKOutput()
    out_id = logger.add(
        file_name,
        level=level,
        encoding="utf-8",
        enqueue=True,
        format=file_format,
        rotation="500 MB",
        mode="w",
    )
    return out_id


def Debug(s):
    if _log_full_print:
        s = GetProcInfo() + str(s)
    else:
        s = str(s)
    logger.debug(s)
    sys.stdout.flush()
    sys.stderr.flush()


def Info(s=""):
    if _log_full_print:
        s = GetProcInfo() + str(s)
    else:
        s = str(s)
    logger.info(s)
    sys.stdout.flush()
    sys.stderr.flush()


def Success(s):
    if _log_full_print:
        s = GetProcInfo() + str(s)
    else:
        s = str(s)
    logger.success(s)
    sys.stdout.flush()
    sys.stderr.flush()


def Warning(s):
    if _log_full_print:
        s = GetProcInfo() + str(s)
    else:
        s = str(s)
    logger.warning(s)
    sys.stdout.flush()
    sys.stderr.flush()


def Error(s):
    if _log_full_print:
        s = GetProcInfo() + str(s)
    else:
        s = str(s)
    logger.error(s)
    sys.stdout.flush()
    sys.stderr.flush()


def Critical(s):
    if _log_full_print:
        s = GetProcInfo() + str(s)
    else:
        s = str(s)
    logger.critical(s)
    sys.stdout.flush()
    sys.stderr.flush()


def Complete():
    logger.complete()


def GetProcInfo():
    pid = os.getpid()
    p = psutil.Process(pid)
    mem = psutil.virtual_memory()
    mem_info = p.memory_full_info()
    t = time.time() - p.create_time()
    proc_info = {"PID": pid, "PPID": p.ppid(), "Time": t, "USS": mem_info.uss / g_MB}
    return " (P/PP:{}/{}, T(s):{:.1f}, U/F(MB):{}/{})\t| ".format(
        proc_info["PID"],
        proc_info["PPID"],
        proc_info["Time"],
        int(proc_info["USS"]),
        int(mem.free / g_MB),
    )
