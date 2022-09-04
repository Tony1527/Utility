import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from Utility.Log import *
from Utility.Constant import *


def PDRead(file_name):
    format = file_name.split(".")[-1]
    if format == "csv":
        return pd.read_csv(file_name)
    elif format in ["xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"]:
        return pd.read_excel(file_name)


def IsIterable(obj):
    if hasattr(obj, "__iter__"):
        return True
    else:
        return False


def EqualTo(iter_X, iter_Y, accuracy=1e-10, abs_zero=1e-14):
    if len(iter_X) != len(iter_Y):
        raise ValueError("length of X and Y is not equal.")
    for i in range(len(iter_X)):
        if not EqualToAbs(iter_X[i], iter_Y[i], accuracy, abs_zero):
            return False
    return True


def ExtSigExp(num, accuracy=10):
    if num == 0:
        return (0, 0)
    sign = 1
    if num < 0:
        sign = -1
    num = np.abs(num)
    base_digit = np.floor(np.log10(num))
    significant_digit = num / 10**base_digit

    significant_digit = np.round(significant_digit * 10**accuracy) / 10**accuracy
    return (sign * significant_digit, base_digit)


def KeepSig(num, accuracy=10):
    significant_digit, base_digit = ExtSigExp(num, accuracy)
    return significant_digit * 10**base_digit


def EqualToR(x, y, accuracy=10):
    return ExtSigExp(x, accuracy) == ExtSigExp(y, accuracy)


def EqualToAbs(x, y, accuracy=1e-10, abs_zero=1e-14):
    if abs(x) < abs_zero:
        x = 0

    if abs(y) < abs_zero:
        y = 0

    if y != 0:
        if abs((x - y) / y) > accuracy:
            return False
    elif abs((x - y)) > accuracy:
        return False
    return True


def static_assign_jobs(k_list, process_num):
    jobs = []
    num_k = len(k_list)
    num_job = int(num_k / process_num)
    num_remain = num_k % process_num
    cnt = 0
    for i in range(0, process_num):
        if num_remain == 0:
            jobs.append(k_list[cnt : cnt + num_job])
            cnt += num_job
        else:
            jobs.append(k_list[cnt : cnt + num_job + 1])
            cnt += num_job + 1
            num_remain -= 1

    # jobs.append(k_list[cnt:len(k_list)])
    return jobs


def OperObj(obj, func, *arg, **kwargs):
    return func(obj, arg, kwargs)


# Usage: using multiprocess method to create several objects of a class, and then invoke certain operator
# def GetEg(obj,*arg,**kwargs):
#   obj.SelfConsistent(0)
#   return *obj.GetParam(),obj.GetGroundEnergy()
# kw={"class":FourBdBLGHFPolar,"operator":GetEg}
# P=np.array(MultiprocTask(p_list,MultiOperObj,4,**kw))
def MultiOperObj(p_list, *arg, **kwargs):
    rtn = []
    begin = time.time()
    Log("DEBUG")
    for i in range(len(p_list)):
        p = p_list[i]
        obj = kwargs["class"](*p)
        rtn.append(
            OperObj(obj, kwargs["operator"], arg, kwargs)
        )  # could be wrong when shallow copy is used
        del obj
        end = time.time()
        Info(
            "============PID %d completes (%d/%d) jobs cost %fs"
            % (os.getpid(), i + 1, len(p_list), end - begin)
        )

    Complete()
    return rtn


def MultiprocTask(p_list, func, process_num=2, is_self_unzip=True, *arg, **kwargs):
    begin = time.time()
    rtn = []

    if g_ss_run_mode == "Release" and process_num > 1:

        process_num = min(os.cpu_count(), process_num)
        pool = Pool(process_num)
        res_l = []

        jobs = static_assign_jobs(p_list, process_num)

        for job in jobs:
            if is_self_unzip:
                res = pool.apply_async(func, (job, arg), kwargs)
                res_l.append(res)
            else:
                for p in job:
                    if IsIterable(job):
                        res = pool.apply_async(func, (*p, *arg), kwargs)
                    else:
                        res = pool.apply_async(func, (p, *arg), kwargs)
                    res_l.append(res)
        pool.close()
        pool.join()

        for res in res_l:
            v = res.get()
            rtn.extend(v)
    elif g_ss_run_mode == "Debug" or process_num <= 1:
        if is_self_unzip:
            res = func(p_list, *arg, **kwargs)
            rtn.extend(res)
        else:
            for p in p_list:
                if IsIterable(p):
                    res = func(*p, *arg, **kwargs)
                else:
                    res = func(p, *arg, **kwargs)
                rtn.extend(res)
    else:
        raise ValueError("unknown g_ss_run_mode value.")

    end = time.time()
    Success(
        "PID:"
        + str(os.getpid())
        + " MultiprocGetE_k using %d Process cost %fs" % (process_num, end - begin)
    )
    return rtn


def GetNodeProcNum():
    is_cluster = os.environ.get("SLURM_CPUS_PER_TASK") != None
    if is_cluster:
        process_num = int(os.environ.get("SLURM_CPUS_PER_TASK"))
    else:
        process_num = os.cpu_count()
    return process_num
