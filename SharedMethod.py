import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import copy
import re
from bisect import bisect_left, bisect_right


plt.rcParams["font.sans-serif"] = "Times"
plt.rcParams["mathtext.fontset"] = "stix"
# plt.rcParams["backend"] = "Qt5Agg"





from Utility.Log import *
from Utility.Constant import *
from Utility.MathUtil import *
try:
    import psutil
    _no_psutil=False
except:
    _no_psutil=True
    Info("module psutil is not fould. Using os module only.")

try:
    from threadpoolctl import threadpool_limits
    _no_threadpoolctl=False
except:
    _no_threadpoolctl=True
    Warning("module threadpoolctl is not fould. Unable to control threads.")


_pool=None
_pool_size=0

_default_output_dir="."+ os.sep + "output"



def LimitThreads(threads_limit,user_api=None):
    if _no_threadpoolctl:
        return None
    else:
        threadpool_limits(threads_limit,user_api)
        return threads_limit

def RegisterPool(process_num,threads_limit=None, user_api=None):
    '''Register a pool with N processes (not larger than logical cores)

    Parameters
    ----------
    process_num : int
        should be smaller than logical cores

    Returns
    -------
    int
        real registered process number
    '''
    global _pool, _pool_size

    if _pool==None:
        log_cores = LogCores()
        if process_num>log_cores:
            Info("Register process number exceed max cores of the computer!")
        process_num = min(log_cores, process_num)
        Info("Register pool with {} processes".format(process_num))
        _pool = Pool(process_num,initializer=LimitThreads,initargs=(threads_limit,user_api))
        _pool_size = process_num
        return process_num
    else:
        return 0

def PhyCores():
    if _no_psutil:
        Warning("No psutil found. Using os.cpu_count() instead. May not be the real physical cores.")
        return os.cpu_count()
    else:
        return psutil.cpu_count(logical=False)

def RecommendedCores():
    return PhyCores()-1

def LogCores():
    return os.cpu_count()

def ClosePool():
    global _pool
    _pool.close()
    _pool.join()
    _pool=None



def Linstep(start_point,end_point,step):
    total_num=int((end_point-start_point)/step)+1

    return np.linspace(start_point,end_point,total_num)



def ZPlot(
    f,
    space,
    f_param=dict(),
    plx=100,
    is_log=False,
    x_normalization=1,
    y_normalization=1,
    z_normalization=1,
    is_show=True,
    xlabel=r"$Re$",
    ylabel=r"$Im$",
    figsize=(13, 5),
    is_fig_equal=True,
    is_one_by_one=False,
    is_save=True,
    filename=None,
):
    rl_max = space[1].real
    im_max = space[1].imag
    rl_min = space[0].real
    im_min = space[0].imag

    X, Y = np.meshgrid(
        np.linspace(rl_min, rl_max, plx), np.linspace(im_min, im_max, plx)
    )
    phase_colors = [
        "navy",
        "blue",
        "blueviolet",
        "purple",
        "red",
        "yellow",
        "green",
        "aqua",
        "navy",
    ]
    mag_colors = ["black", "grey", "white"]
    mag_colormap = clrs.LinearSegmentedColormap.from_list("MagClrsMapping", mag_colors)
    phase_colormap = clrs.LinearSegmentedColormap.from_list(
        "PhaseClrsMapping", phase_colors
    )

    Z = X + 1j * Y

    # begin=time.time()

    if is_one_by_one:
        ret_values = np.vectorize(ZFunc)(Z, f, **f_param) / z_normalization
    else:
        ret_values = ZFunc(Z, f, **f_param) / z_normalization

    # end=time.time()
    # print("Total Time consuming %fs"%(end-begin))


    M = np.abs(ret_values)
    P = np.angle(ret_values)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(figsize[0], figsize[1])

    if is_log:
        M = np.log10(M)
        ax1.set_title("Magnitude(Log10)", fontsize=20)
    else:
        ax1.set_title("Magnitude", fontsize=20)
    ax2.set_title("Phase", fontsize=20)

    ax1_loc = make_axes_locatable(ax1)
    ax2_loc = make_axes_locatable(ax2)
    ax1_cax = ax1_loc.append_axes("right", size="5%", pad=0.05)
    ax2_cax = ax2_loc.append_axes("right", size="5%", pad=0.05)

    # ax2=plt.subplot()
    ax1.set_xlabel(xlabel, fontsize=18)
    ax1.set_ylabel(ylabel, fontsize=18)
    ax2.set_xlabel(xlabel, fontsize=18)
    ax2.set_ylabel(ylabel, fontsize=18)
    ax1.set_xlim(rl_min / x_normalization, rl_max / x_normalization)
    ax1.set_ylim(im_min / y_normalization, im_max / y_normalization)
    ax2.set_xlim(rl_min / x_normalization, rl_max / x_normalization)
    ax2.set_ylim(im_min / y_normalization, im_max / y_normalization)

    pc1 = ax1.pcolor(X / x_normalization, Y / y_normalization, M, cmap=mag_colormap)
    pc2 = ax2.pcolor(
        X / x_normalization,
        Y / y_normalization,
        P,
        cmap=phase_colormap,
        vmin=-np.pi,
        vmax=np.pi,
    )

    ax1_clb = fig.colorbar(pc1, cax=ax1_cax, orientation="vertical")
    ax2_clb = fig.colorbar(pc2, cax=ax2_cax, orientation="vertical")

    ax2_clb.ax.set_ylim([-pi, pi])
    ax2_clb.ax.yaxis.set_ticks([-pi, 0, pi])
    ax2_clb.ax.yaxis.set_ticklabels(["-π", "0", "π"])

    if is_fig_equal:
        ax1.set_aspect("equal")
        ax2.set_aspect("equal")

    if is_show == True:
        plt.show()

    if filename==None:
        filename="ZPlot"

    if is_save:
        SavePDF(fig,filename=filename)

    return fig, (ax1, ax2)




def PDRead(filename,dir_path = _default_output_dir,postfix="csv", *arg, **kwargs):
    '''read data of "csv" or "xls" etc. form.

    Parameters
    ----------
    filename : string
        file name

    Returns
    -------
    DataFrame
        DataFrame form
    '''
    path=ConcatFilePath(filename=filename,dir_path=dir_path,postfix=postfix)
    if postfix == "csv":
        return pd.read_csv(path, *arg, **kwargs)
    elif postfix in ["xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"]:
        return pd.read_excel(path, *arg, **kwargs)
    else:
        raise ValueError("Unknown format caught! ({})".format(postfix))

def Save(data=None,filename=None,dir_path = _default_output_dir,postfix="pickle",*arg,**kwarg):
    if filename==None:
        Error("filename could not be None")
    path = ConcatFilePath(filename=filename,dir_path=dir_path,postfix=postfix)
    MKDirsToFile(path)
    if postfix in ["png","jpg"]:
        plt.savefig(path,*arg,**kwarg)
    elif postfix=="pickle":
        with open(path, "wb") as f:
            pickle.dump(data, f,*arg,**kwarg)
    elif postfix=="pdf":
        with PdfPages(path) as pdf:
            pdf.savefig(data,*arg,**kwarg)
            # pdf.savefig(figure, bbox_inches="tight")    
    elif postfix=="svg":
        plt.savefig(path, format="eps", dpi=200,*arg,**kwarg)



def SavePDF(figure, filename,dir_path = _default_output_dir,postfix="pdf"):
    path=ConcatFilePath(filename,dir_path,postfix)
    MKDirsToFile(path)
    with PdfPages(path) as pdf:
        pdf.savefig(figure)
        # pdf.savefig(figure, bbox_inches="tight")


def SaveSVG(figure, filename ,dir_path = _default_output_dir):
    path=ConcatFilePath(filename,dir_path,postfix="eps")
    MKDirsToFile(path)
    figure.savefig(path, format="eps", dpi=300)

def Load(filename,dir_path = _default_output_dir,postfix="pickle"):
    data = None
    path = ConcatFilePath(filename=filename,dir_path=dir_path,postfix=postfix)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data





def PDSave(T,filename,dir_path = _default_output_dir,postfix="csv",*arg,**kwargs):
    path = ConcatFilePath(filename=filename,dir_path=dir_path,postfix=postfix)
    MKDirsToFile(path)
    if postfix == "csv":
        T.to_csv(path,*arg,**kwargs)
    else:
        writer = pd.ExcelWriter(path)
        T.to_excel(writer,*arg,**kwargs)
        writer.close()


def IsIterable(obj):
    if hasattr(obj, "__iter__"):
        return True
    else:
        return False

def IsFunction(obj):
    if hasattr(obj, "__call__"):
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


def static_assign_jobs(p_list, process_num):
    jobs = []
    num_k = len(p_list)
    num_job = int(num_k / process_num)
    num_remain = num_k % process_num
    cnt = 0
    for i in range(0, process_num):
        if num_remain == 0:
            jobs.append(p_list[cnt : cnt + num_job])
            cnt += num_job
        else:
            jobs.append(p_list[cnt : cnt + num_job + 1])
            cnt += num_job + 1
            num_remain -= 1

    return jobs




# Usage: using multiprocess method to create several objects of a class, and then invoke certain operator
# def GetEg(obj,*arg,**kwargs):
#   obj.SelfConsistent(0)
#   return *obj.GetParam(),obj.GetGroundEnergy()
# kw={"class":FourBdBLGHFPolar,"operator":GetEg}
# P=np.array(MultiprocTask(p_list,MultiOperObj,4,**kw))
# def MultiOperObj(p_list, *arg, **kwargs):
#     rtn = []
#     begin = time.time()
#     Log("DEBUG")
#     for i in range(len(p_list)):
#         p = p_list[i]
#         obj = kwargs["class"](*p)
#         rtn.append(
#             OperObj(obj, kwargs["operator"], arg, kwargs)
#         )  # could be wrong when shallow copy is used
#         del obj
#         end = time.time()
#         Info(
#             "============PID %d completes (%d/%d) jobs cost %fs"
#             % (os.getpid(), i + 1, len(p_list), end - begin)
#         )

#     Complete()
#     return rtn

def OperObj(cls_param, cls, op, **op_kwargs):
    obj = cls(*cls_param)
    rtn = op(obj, **op_kwargs)
    del obj
    return rtn 

def WarpParams(p_list):
        k_list=[]
        for p in p_list:
            k_list.append((p,))
        return k_list



def MultiprocTask(func, p_list=[(... ,), (... ,)], process_num=2, **kwargs):
    '''tackling tasks with multiprocess method. 

    Parameters
    ----------
    p_list : list
        input parameter list. For p in p_list, p should take [(... ,), (... ,)] , [scalar 1 , scalar 2] form.
    func : function
        function
    process_num : int, >=1
        process number

    Returns
    -------
    list
        Output

    Raises
    ------
    ValueError
        unknown g_ss_run_mode value
    '''
    # def count_on_completing_jobs(rtn):
        # completing_job_num+=1
        # Debug("Completing "+str(completing_job_num)+"/"+str(total_jobs_num))
        # return rtn

    begin = time.time()
    rtn = []


    if g_ss_run_mode == "Release" and process_num > 1:

        global _pool
        if _pool==None:
            RegisterPool(process_num)
        else:
            process_num = _pool_size
        res_l = []

        jobs = static_assign_jobs(p_list, process_num)
        total_jobs_num=len(p_list)
        completing_job_num=0

        if not (isinstance(jobs[0][0],tuple) or (not IsIterable(jobs[0][0]))):
            Warning("The p_list is not in [(... ,), (... ,)] , [scalar 1 , scalar 2] form. The parameters may not be what you expect. You may have to use WarpParams(p_list) to wrap p_list.")

        for job in jobs:
            for p in job:
                if IsIterable(p):
                    res = _pool.apply_async(func, (*p, ), kwds=kwargs)
                else:
                    res = _pool.apply_async(func, (p, ), kwds=kwargs)
                res_l.append(res)

        for res in res_l:
            r = res.get()
            # if IsIterable(v):
            #     rtn.extend(v)
            # else:
            #     rtn.append(v)
            completing_job_num+=1
            if total_jobs_num>=10:
                if completing_job_num%int(total_jobs_num*0.1)==0:
                    Debug("Completing "+str(completing_job_num)+"/"+str(total_jobs_num))
            else:
                Debug("Completing "+str(completing_job_num)+"/"+str(total_jobs_num))
            rtn.append(r)
            
    elif g_ss_run_mode == "Debug" or process_num <= 1:
        for p in p_list:
            # p should take (x1, x2, x3, ...), [x1, x2, x3, ...], or x1 form
            if IsIterable(p): 
                res = func(*p, **kwargs)
            else:
                res = func(p, **kwargs)
            # if IsIterable(res):
            #     rtn.extend(res)
            # else:
            #     rtn.append(res)
            rtn.append(res)
    else:
        raise ValueError("unknown g_ss_run_mode value.")

    end = time.time()
    Success(
        "PID:"
        + str(os.getpid())
        + " MultiprocTasks using %d Process cost %fs" % (process_num, end - begin)
    )
    return rtn


# def MultiprocTask(p_list, func, process_num=2, is_self_unzip=True, is_param_form=True, *arg, **kwargs):
#     '''tackling tasks with multiprocess method

#     Parameters
#     ----------
#     p_list : list
#         input parameter list
#     func : function
#         function
#     process_num : int, >=1
#         process number
#     is_self_unzip : bool, optional
#         True: setting for function MultiOperObj.
#         False: setting for usual case.
#     is_param_form : bool, optional
#         True: p should have take (x1, x2, x3, ...), [x1, x2, x3, ...], or x1 form.
#         False: This function will help you to wrap the p_list in above form.

#     Returns
#     -------
#     list
#         Output

#     Raises
#     ------
#     ValueError
#         unknown g_ss_run_mode value
#     '''

#     def warp_plist(p_list):
#         k_list=[]
#         for p in p_list:
#             k_list.append([p])
#         return k_list

#     begin = time.time()
#     rtn = []

#     if is_param_form==False:
#         p_list=warp_plist(p_list)

#     if g_ss_run_mode == "Release" and process_num > 1:

#         global _pool
#         if _pool==None:
#             RegisterPool(process_num)
#         else:
#             process_num = _pool_size
#         res_l = []

#         jobs = static_assign_jobs(p_list, process_num)

#         for job in jobs:
#             if is_self_unzip:
#                 res = _pool.apply_async(func, (job, *arg), kwds=kwargs)
#                 res_l.append(res)
#             else:
#                 for p in job:
#                     # p should take (x1, x2, x3, ...) or [x1, x2, x3, ...] form
#                     if IsIterable(p):
#                         res = _pool.apply_async(func, (*p, *arg), kwds=kwargs)
#                     else:
#                         res = _pool.apply_async(func, (p, *arg), kwds=kwargs)
#                     res_l.append(res)
#         # pool.close()
#         # pool.join()

#         for res in res_l:
#             v = res.get()
#             # if IsIterable(v):
#             #     rtn.extend(v)
#             # else:
#             #     rtn.append(v)
#             rtn.append(v)
            
#     elif g_ss_run_mode == "Debug" or process_num <= 1:
#         if is_self_unzip:
#             res = func(p_list, *arg, **kwargs)
#             # if IsIterable(res):
#             #     rtn.extend(res)
#             # else:
#             #     rtn.append(res)
#             rtn.append(res)
#         else:
#             for p in p_list:
#                 # p should take (x1, x2, x3, ...), [x1, x2, x3, ...], or x1 form
#                 if IsIterable(p):
#                     res = func(*p, *arg, **kwargs)
#                 else:
#                     res = func(p, *arg, **kwargs)
#                 # if IsIterable(res):
#                 #     rtn.extend(res)
#                 # else:
#                 #     rtn.append(res)
#                 rtn.append(res)
#     else:
#         raise ValueError("unknown g_ss_run_mode value.")

#     end = time.time()
#     Success(
#         "PID:"
#         + str(os.getpid())
#         + " MultiprocTasks using %d Process cost %fs" % (process_num, end - begin)
#     )
#     return rtn



def GetNodeProcNum():
    is_cluster = os.environ.get("SLURM_CPUS_PER_TASK") != None
    if is_cluster:
        process_num = int(os.environ.get("SLURM_CPUS_PER_TASK"))
    else:
        process_num = os.cpu_count()
    return process_num



def SetXYTicks(ax, labelsize, *arg, **kwargs):
    is_yticks_set = False
    is_xticks_set = False
    if kwargs.get("ytick_num") != None:
        yticks_num = kwargs["ytick_num"]
        if yticks_num != 0:
            ylim = kwargs["ylim"]
            ax.set_yticks(np.linspace(ylim[0], ylim[1], yticks_num))
            is_yticks_set = True
    if kwargs.get("yticks_step") != None:
        yticks_step = kwargs["yticks_step"]
        if yticks_step != 0:
            if is_yticks_set:
                raise ValueError("yticks has been set")
            ylim = kwargs["ylim"]
            yticks_num = int((ylim[1] - ylim[0]) / yticks_step) + 1
            ax.set_yticks(np.linspace(ylim[0], ylim[1], yticks_num))
            is_yticks_set = True

    if kwargs.get("xtick_num") != None:
        xticks_num = kwargs["xtick_num"]
        if xticks_num != 0:
            xlim = kwargs["xlim"]
            ax.set_xticks(np.linspace(xlim[0], xlim[1], xticks_num))
            is_xticks_set = True

    if kwargs.get("xticks_step") != None:
        xticks_step = kwargs["xticks_step"]
        if xticks_step != 0:
            if is_xticks_set:
                raise ValueError("xticks has been set")
            xlim = kwargs["xlim"]
            xticks_num = int((xlim[1] - xlim[0]) / xticks_step) + 1
            ax.set_xticks(np.linspace(xlim[0], xlim[1], xticks_num))
            is_xticks_set = True

    if kwargs.get("xticks") != None:
        xticks = kwargs.get("xticks")
        if len(xticks) != 0:
            ax.set_xticks(xticks)

        if kwargs.get("xtick_names") != None:
            if is_xticks_set:
                raise ValueError("xticks has been set")
            points_names = kwargs["xtick_names"]
            if len(points_names) != 0:
                font_size = kwargs["fontsize"]
                ax.set_xticklabels(points_names, fontsize=font_size)
                is_xticks_set = True

    ax.tick_params(labelsize=labelsize)


def NormalizeArray(V, v_normalization):
    V_normalized = copy.deepcopy(V)
    min_v=max_v=range_v=None

    if v_normalization==0:
        min_v = np.min(V_normalized)
        max_v = np.max(V_normalized)
        range_v = np.linalg.norm(max_v-min_v)

    if isinstance(V,np.ndarray):
        if v_normalization==0:    
                V_normalized-=min_v
                V_normalized/=range_v
        else:
            V_normalized /= v_normalization
    else:
        for m in range(len(V_normalized)):
            if v_normalization==0:    
                V_normalized[m]-=min_v
                V_normalized[m]/=range_v
            else:
                V_normalized[m] /= v_normalization
    return V_normalized

def CalculateXY(
    x_list,
    get_y_list,
    base_num=100,
    is_save=False,
    filename="Y-X",
    f_param={},
    is_one_by_one=False,
    is_multiproc=False
):

    X = np.linspace(x_list[0], x_list[1], base_num)

    if len(f_param) == 0:
        
        if is_one_by_one:
            # Y = np.vectorize(get_y_list)(X)
            if is_multiproc:
                Y=MultiprocTask(get_y_list, WarpParams(X), RecommendedCores())
            else:
                Y = np.vectorize(get_y_list)(X)
        else:
            if is_multiproc:
                Y=MultiprocTask(get_y_list, WarpParams(X), RecommendedCores())
            else:
                Y = get_y_list(X)
    else:
        if is_one_by_one:
            # Y = np.vectorize(get_y_list)(X)
            if is_multiproc:
                Y=MultiprocTask(get_y_list, WarpParams(X), RecommendedCores(), **f_param)
            else:
                Y = np.vectorize(get_y_list)(X, **f_param)
        else:
            if is_multiproc:
                Y=MultiprocTask(get_y_list, WarpParams(X), RecommendedCores(), **f_param)
            else:
                Y = get_y_list(X, **f_param)
        # if is_one_by_one:
        #     Y = np.vectorize(get_y_list)(X, **f_param)
        # else:
        #     Y = get_y_list(X, **f_param)


    
    

    if is_save:
        Save({"X": X, "Y": Y},filename)

    return X, Y


def PlotXY(
    x_list,
    get_y_list,
    base_num=100,
    xlabel="$X$",
    ylabel="$Y$",
    title="X-Y Plot",
    legend=None,
    xlim=None,
    ylim=None,
    yticks_num=0,
    xticks_num=0,
    label_sz=16,
    font_sz=16,
    line_w=2,
    style="line",
    filename=None,
    is_save=False,
    y_normalization=1,
    x_normalization=1,
    axes=None,
    f_param={},
    is_one_by_one=False,
    is_multiproc=False,
    is_show=True,
    figsize=(8,8),
    is_save_picture=False,
    *arg,
    **kwargs
):

    if filename==None:
        filename=title
    if IsFunction(get_y_list):
        if base_num <= 1:
            base_num = 2

        

        X, Y = CalculateXY(
            x_list,
            get_y_list,
            base_num=base_num,
            is_save=is_save,
            filename=filename,
            f_param=f_param,
            is_one_by_one=is_one_by_one,
            is_multiproc=is_multiproc,
        )
        X=NormalizeArray(np.array(X),x_normalization)
        Y=NormalizeArray(np.array(Y),y_normalization)
    else:
        if is_save:
            Save({"X": x_list, "Y": get_y_list},filename=filename)
        X=NormalizeArray(np.array(x_list),x_normalization)
        Y=NormalizeArray(np.array(get_y_list),y_normalization)
    

    ##start drawing picture##

    ax = axes

    if ax == None:
        plt.figure(figsize=figsize)
        ax = plt.axes()
        ax.tick_params(labelsize=label_sz)
        ax.set_xlabel(xlabel, fontsize=font_sz)
        ax.set_ylabel(ylabel, fontsize=font_sz)

        if xlim != None:
            ax.set_xlim(xlim)
            if xticks_num != 0:
                ax.set_xticks(np.linspace(xlim[0], xlim[1], xticks_num))
        if ylim != None:
            ax.set_ylim(ylim)
            if yticks_num != 0:
                ax.set_yticks(np.linspace(ylim[0], ylim[1], yticks_num))

        ax.set_title(title, fontsize=font_sz + 2)
    else:
        pass

    XYFigPlot(ax, style, X, Y, lw=line_w,label=legend,*arg,**kwargs)
    if legend!=None:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
    if is_show == True:
        plt.show()
    else:
        if is_save_picture:
            Save(filename=filename,dpi=200,postfix="png")

    return ax

def ReplaceYParam(y):
    def decorator(func):
        def wrapper(*args,**kwargs):
            kwargs.update({y:kwargs["_y"]})
            kwargs.pop("_y")
            return func(*args,**kwargs)
        return wrapper
    return decorator

def PlotMultiLines(
    x_list,
    y_list,
    get_z_list,
    base_num_x=100,
    base_num_y=100,
    x_normalization=1,
    y_normalization=1,
    z_normalization=0,
    shift=1,
    xlabel="${x}$",
    zlabel="${z}$",
    unit_y=None,
    title="$X-Z(Y)$",
    xlim=None,
    zlim=None,
    yticks_num=0,
    xticks_num=0,
    label_sz=17,
    axes=None,
    style="line",
    f_param={},
    is_one_by_one=False,
    is_save=False,
    filename=None,
    is_show=True,
    **kwargs
):
    if base_num_y!=0:
        Y = np.linspace(y_list[0],y_list[1],base_num_y)
    else:
        Y=y_list
        base_num_y = y_list
    Z_list=[]
    ax = axes

    if filename==None:
        filename=title
    
    offset=0
    for i,y in enumerate(Y):
        
        f_param.update({"_y":y})
        
        if IsFunction(get_z_list):
            X,Z=CalculateXY(x_list,get_z_list,base_num=base_num_x,is_save=False,
                            f_param=f_param,is_one_by_one=is_one_by_one,filename=filename)

            Z_list.append(Z)
            X=NormalizeArray(np.array(X),x_normalization)
            Z=NormalizeArray(np.array(Z),z_normalization)
        else:
            Z_list.append(get_z_list[i])
            X=NormalizeArray(np.array(x_list),x_normalization)
            Z=NormalizeArray(np.array(get_z_list[i]),z_normalization)

        legend_y=None
        if i==0 and unit_y!=None:
            legend_y="Bottom: "+str(y/y_normalization)+unit_y
        elif i==len(Y)-1 and unit_y!=None:
            legend_y="Top: "+str(y/y_normalization)+unit_y

        if z_normalization==0:
            Z+=offset
            ax = PlotXY(X,Z,base_num=0,xlabel=xlabel,ylabel=zlabel,title=title,xlim=xlim,ylim=(0-0.1,(base_num_y+1)*shift+0.1),
            label_sz=label_sz,axes=ax,is_save=is_save,filename=filename,is_show=False,style=style,xticks_num=xticks_num,yticks_num=yticks_num,legend=legend_y)
            offset+=shift
        else:
            ax = PlotXY(X,Z/z_normalization,base_num=0,xlabel=xlabel,ylabel=zlabel,title=title,xlim=xlim,ylim=zlim,
            label_sz=label_sz,axes=ax,is_save=is_save,filename=filename,is_show=False,style=style,xticks_num=xticks_num,yticks_num=yticks_num,legend=legend_y)

        

    if is_save:
        Save({"X":X,"Y":Y,"Z_list":Z_list},filename=filename)
    
    if is_show:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.show()
    return ax


def XYFigPlot(ax, style, X, Y, *arg, **kwargs):
    def extract_color(str,exclusion=[]):
        parts=str.split("_")
        color=None
        for s in parts:
            if np.all(np.array(exclusion)!=s):
                color=s
                return color
            

    line_pattern=re.compile(r"\w*line")
    scatter_pattern=re.compile(r"\w*scatter")
    bar_pattern=re.compile(r"\w*bar")
    line_list=line_pattern.findall(style)
    scatter_list=scatter_pattern.findall(style)
    bar_list=bar_pattern.findall(style)
    if len(line_list)!=0:
        line_str=line_list[0]
        color=extract_color(line_str,["dash","line"])
        if re.match(r".*dash.*",line_str)!=None:
            ax.plot(X, Y, c=color, linestyle="--", *arg, **kwargs)
        else:
            ax.plot(X, Y, c=color, *arg, **kwargs)
    elif len(scatter_list)!=0:
        scatter_str=scatter_list[0]
        color=extract_color(scatter_str,["scatter"])
        ax.scatter(X, Y, marker="o", c=color, *arg, **kwargs)
    elif len(bar_list)!=0:
        bar_str=bar_list[0]
        color=extract_color(bar_str,["bar"])
        ax.bar(X, Y, marker="o", c=color, *arg, **kwargs)
    else:
        raise ValueError("unknown style {}.".format(style))
    


    # if style == "line":
    #     ax.plot(X, Y, *arg, **kwargs)
    # elif style == "scatter":
    #     ax.scatter(X, Y, marker="o", *arg, **kwargs)
    # elif style == "grey_scatter":
    #     ax.scatter(X, Y, marker="o", c="grey", *arg, **kwargs)
    # elif style == "black_scatter":
    #     ax.scatter(X, Y, marker="o", c="black", *arg, **kwargs)
    # elif style == "blue_scatter":
    #     ax.scatter(X, Y, marker="o", c="blue", *arg, **kwargs)
    # elif style == "red_scatter":
    #     ax.scatter(X, Y, marker="o", c="red", *arg, **kwargs)
    # elif style == "orange_scatter":
    #     ax.scatter(X, Y, marker="o", c="orange", *arg, **kwargs)
    # elif style == "green_scatter":
    #     ax.scatter(X, Y, marker="o", c="green", *arg, **kwargs)
    # elif style == "blue_line":
    #     ax.plot(X, Y, c="blue", *arg, **kwargs)
    # elif style == "red_line":
    #     ax.plot(X, Y, c="red", *arg, **kwargs)
    # elif style == "black_line":
    #     ax.plot(X, Y, c="black", *arg, **kwargs)
    # elif style == "dash_black_line":
    #     ax.plot(X, Y, c="black", linestyle="--", *arg, **kwargs)
    # elif style == "dash_red_line":
    #     ax.plot(X, Y, c="red", linestyle="--", *arg, **kwargs)
    # elif style == "blue_bar":
    #     ax.bar(X, Y, color="mediumblue", *arg, **kwargs)
    # else:
    #     raise ValueError("unknown style.")



def Calculate3D(
    x_list,
    y_list,
    GetZList,
    base_num=100,
    is_save=False,
    filename="Z-XY",
    is_multiproc=False
):

    X, Y = np.meshgrid(
        np.linspace(x_list[0], x_list[1], base_num),
        np.linspace(y_list[0], y_list[1], base_num),
    )
    p_list = np.concatenate(
        (X.reshape(base_num * base_num, 1), Y.reshape(base_num * base_num, 1)), axis=1
    )

    if is_multiproc:
        z_list = MultiprocTask(GetZList, WarpParams(p_list))
    else:
        z_list = MultiprocTask(GetZList, WarpParams(p_list) ,1)

    Z = np.array(z_list)


    Z = Z.reshape(base_num, base_num)

    if is_save:
        Save({"X": X, "Y": Y, "Z_list": Z},filename=filename)

    return Z, X, Y


def Plot3D(
    x_list,
    y_list,
    get_z_list,
    base_num=100,
    xlabel="${x}$",
    ylabel="${y}$",
    zlabel="${E(x)}$(?eV)",
    title="$E(\mathbf{k})$",
    xlim=None,
    ylim=None,
    yticks_num=0,
    tick_sz=15,
    label_sz=17,
    style="pcolor",
    is_save=False,
    filename=None,
    x_normalization=1,
    y_normalization=1,
    z_normalization=1,
    axes=None,
    is_show=True,
    is_multiproc=False,
    yticks_step=0,
    xticks=[],
    xtick_names=[],
    figsize=(8, 8),
    **kwargs
):
    """
    plot 3D picture of z vs x-y
    """

    if filename==None:
        filename=title

    if IsFunction(get_z_list):
        if base_num <= 1:
            base_num = 2
        Z, X, Y = Calculate3D(
            x_list,
            y_list,
            get_z_list,
            base_num,
            is_save=is_save,
            filename=filename,
            is_multiproc=is_multiproc
        )

        X=NormalizeArray(np.array(X),x_normalization)
        Y=NormalizeArray(np.array(Y),y_normalization)
        Z=NormalizeArray(np.array(Z),z_normalization)
    else:
        if is_save:
            Save({"X":x_list,"Y":y_list,"Z":get_z_list},filename=filename)
        X=NormalizeArray(np.array(x_list),x_normalization)
        Y=NormalizeArray(np.array(y_list),y_normalization)
        Z=NormalizeArray(np.array(get_z_list),z_normalization)

    ##start drawing picture##

    if axes == None:

        if style == "surface" or style == "scatter":
            if style=="surface":
                fig = plt.figure(
                    figsize=figsize
                )
            elif style == "scatter":
                fig = plt.figure(
                    figsize=figsize#, bbox_inches="tight", pad_inches=5
                )  # (8,6)
            ax = plt.axes(projection="3d")

            ax.set_zlabel(zlabel, fontsize=label_sz, labelpad=20)
            ax.zaxis.set_tick_params(pad=10)
        else:
            fig = plt.figure(figsize=figsize)
            # fig=plt.figure(figsize=(6,6),dpi=600)
            ax = plt.axes()

        # ax.tick_params(labelsize=tick_sz)
        ax.set_xlabel(xlabel, fontsize=label_sz, labelpad=13)
        ax.set_ylabel(ylabel, fontsize=label_sz, labelpad=13)

        ax.xaxis.set_tick_params(pad=8)
        ax.yaxis.set_tick_params(pad=8)

        if xlim != None:
            ax.set_xlim(xlim)
        if ylim != None:
            ax.set_ylim(ylim)

        ax.set_title(title, fontsize=label_sz + 4)

        SetXYTicks(
            ax,
            labelsize=tick_sz,
            ylim=ylim,
            yticks_num=yticks_num,
            xtick_names=xtick_names,
            xticks=xticks,
            fontsize=label_sz,
            yticks_step=yticks_step,
        )

    z_colors = ["#3050C0", "white", "firebrick"]
    # z_colors = ["black", "grey", "white"]
    z_colormap = clrs.LinearSegmentedColormap.from_list("ZClrsMapping", z_colors)
    if style == "surface":
        # ax.plot_surface(X,Y,Z,cmap='rainbow',rcount=100)
        ax.plot_surface(X, Y, Z, cmap=z_colormap, rcount=100)
        if kwargs.get("azim") != None and kwargs.get("elev") != None:
            ax.view_init(elev=kwargs["elev"], azim=kwargs["azim"])
    elif style == "pcolor":

        # pc=ax.pcolor(X,Y,Z,cmap=cm.coolwarm)
        pc = ax.pcolor(X, Y, Z, cmap=z_colormap)

        ax_loc = make_axes_locatable(ax)
        ax_cax = ax_loc.append_axes("right", size="5%", pad=0.05)
        ax_clb = fig.colorbar(pc, cax=ax_cax, orientation="vertical")
        # ax.set_aspect("equal",adjustable='box')
    elif style == "scatter":
        pc = ax.scatter(X, Y, Z, c="black")

    plt.tight_layout()
    if is_show:
        plt.show()
    if is_save:
        SavePDF(fig,filename=filename)
    return ax


def Index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

def FindLT(a, x):
    'Find rightmost value less than x'
    i = bisect_left(a, x)
    if i:
        return a[i-1],i-1
    raise ValueError

def FindLE(a, x):
    'Find rightmost value less than or equal to x'
    i = bisect_right(a, x)
    if i:
        return a[i-1],i-1
    raise ValueError

def FindGT(a, x):
    'Find leftmost value greater than x'
    i = bisect_right(a, x)
    if i != len(a):
        return a[i],i
    raise ValueError

def FindGE(a, x):
    'Find leftmost item greater than or equal to x'
    i = bisect_left(a, x)
    if i != len(a):
        return a[i],i
    raise ValueError
