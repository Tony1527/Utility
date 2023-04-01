import sys
import os
import time
from Utility.Constant import g_MB, g_ss_run_mode

_default_output_dir="."+ os.sep + "output"

try:
    import psutil
    _no_psutil=False
except:
    _no_psutil=True
    print("@Info: module psutil is not fould. Using os module only.")


try:
    from loguru import logger
    _no_loguru=False
except:
    _no_loguru=True
    import logging
    import logging.handlers
    _logging_level={"DEBUG":logging.DEBUG,"INFO":logging.INFO,"WARNING":logging.WARNING,"ERROR":logging.ERROR,"CRITICAL":logging.CRITICAL}
    logger=logging.getLogger(__name__)
    print("@Info: module loguru is not found. Using logging module.")

if g_ss_run_mode=="Debug":
    _log_full_print= True
else:
    _log_full_print= False


def SplitDirFile(s):
    dir_end = s.rfind(os.sep)
    if dir_end != -1:
        return s[:dir_end], s[dir_end + 1 :]
    else:
        return None, s

def ConcatFilePath(file_name,dir_path,postfix):
    path = dir_path
    format=file_name.split(".")
    if len(format)!=1:
        Warning("Two postfix found -- file_name ({}) and postfix ({}).".format(format[-1],postfix))
    if dir_path[-1]!=os.sep:
        path+=os.sep
    path+=file_name+"."+postfix
    return path

def DivideFilePath(path):
    dir_path,file_name = SplitDirFile(path)
    file_name_parts=file_name.split(".")
    if len(file_name_parts)!=2:
        Error("filename ({}) is not in XXX.xxx form.".format(file_name))
    file_name=file_name_parts[0]
    postfix=file_name_parts[-1]
    return {"file_name":file_name,"dir_path":dir_path,"postfix":postfix}





def MKDirsToFile(path):
    dir_path, file_name = SplitDirFile(path)
    if dir_path != None and not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return True
    else:
        return False


def MKOutput(path=_default_output_dir):
    return MKDirsToFile(path + os.sep)


def Log(level="debug", is_full_print=False):
    """
    Initialize logger
    """
    level = level.upper()
    err_id, out_id = (-1, -1)
    
    if _no_loguru:
        if _no_psutil:
            cmd_format = "%(asctime)s | %(levelname)s | %(message)s"
        else:
            cmd_format = "%(levelname)s | %(message)s"
        logging.basicConfig(level=_logging_level[level],format=cmd_format)
    else:
        cmd_format = "<level>{level:<8}</level> | <level>{message}</level>"
        logger.remove()
        err_id = logger.add(sys.stderr, level=level, format=cmd_format, enqueue=True)
    global _log_full_print
    _log_full_print = is_full_print

def AddLogFile(file_name="file",dir_path = _default_output_dir,postfix="log", level="debug"):
    level = level.upper()
    
    path=ConcatFilePath(file_name,dir_path,postfix)
    out_id = (-1, -1)
    MKDirsToFile(path)
    if _no_loguru:
        if _log_full_print:
            file_format = "%(asctime)s | %(levelname)s | %(message)s"
        else:
            file_format ="%(levelname)s | %(message)s"
        file_handler=logging.handlers.RotatingFileHandler(path)
        file_handler.setFormatter(logging.Formatter(file_format))
        file_handler.setLevel(_logging_level[level])
        logger.addHandler(file_handler)
    else:
        if _log_full_print:
            file_format = "{time:YY-MM-DD HH:mm:ss} | <level>{level:<8}</level> | <level>{message}</level> "
        else:
            file_format = "<level>{level:<8}</level> | <level>{message}</level> "
        out_id = logger.add(
            path,
            level=level,
            encoding="utf-8",
            enqueue=True,
            format=file_format,
            rotation="500 MB",
            mode="w",
        )
    





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
    if _no_loguru:
        logger.info(s)
    else:
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
    if _no_loguru:
        pass
    else:
        logger.complete()


def GetProcInfo():
    pid = os.getpid()
    if _no_psutil:
        return "(PID:{})\t| ".format(pid)
    else:
        p = psutil.Process(pid)
        mem = psutil.virtual_memory()
        mem_info = p.memory_full_info()
        t = time.time() - p.create_time()
        proc_info = {"PID": pid, "PPID": p.ppid(), "Time": t, "USS": mem_info.uss / g_MB}
        return " (P/PP:{}/{}, T(s):{:.1f}, U/F(MB):{}/{}) | ".format(
            proc_info["PID"],
            proc_info["PPID"],
            proc_info["Time"],
            int(proc_info["USS"]),
            int(mem.free / g_MB),
        )


Log()