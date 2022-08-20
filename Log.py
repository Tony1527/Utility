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

def MKOutput():
    if not os.path.exists("./output"):
        os.makedirs("output")
        return True
    else:
        return False


def Log(level="DEBUG",is_back_up=True):
    file_format="{time:YY-MM-DD HH:mm:ss} | <level>{level:<8}</level> | <level>{message}</level> "
    cmd_format="<level>{level:<8}</level> | <level>{message}</level>"
    err_id,out_id=(-1,-1)

    logger.remove()
    err_id=logger.add(sys.stderr,level=level,format=cmd_format,enqueue=True)

    if is_back_up:
        MKOutput()
        out_id=logger.add("output/file.log",level="DEBUG",encoding="utf-8",enqueue=True,format=file_format,rotation="500 MB")
    return err_id,out_id


def Debug(s):
    info=GetProcInfo()
    logger.debug(info+s)
    sys.stdout.flush()
    sys.stderr.flush()

def Info(s):
    info=GetProcInfo()
    logger.info(info+s)
    sys.stdout.flush()
    sys.stderr.flush()

def Success(s):
    info=GetProcInfo()
    logger.success(info+s)
    sys.stdout.flush()
    sys.stderr.flush()

def Warning(s):
    info=GetProcInfo()
    logger.warning(info+s)
    sys.stdout.flush()
    sys.stderr.flush()

def Error(s):
    info=GetProcInfo()
    logger.error(info+s)
    sys.stdout.flush()
    sys.stderr.flush()

def Complete():
    logger.complete()


def GetProcInfo():    
    pid=os.getpid()
    p=psutil.Process(pid)
    mem=psutil.virtual_memory()
    mem_info=p.memory_full_info()
    t=time.time()-p.create_time()
    proc_info={"PID":pid,"PPID":p.ppid(),"Time":t,"USS":mem_info.uss/g_MB}
    return " (P/PP:{}/{}, T(s):{:.1f}, U/F(MB):{}/{})\t| ".format(proc_info["PID"],proc_info["PPID"],proc_info["Time"],int(proc_info["USS"]),int(mem.free/g_MB))
