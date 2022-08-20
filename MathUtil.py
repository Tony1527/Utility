import numpy as np
from math import sqrt,isinf,pi
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
import pickle

from Utility.SharedMethod import MKOutput

plt.rcParams['font.sans-serif']='Times'
plt.rcParams['mathtext.fontset']='stix'

c_j=complex(0,1)
pi=np.pi

def Norm2(v):
    if type(v[0])==complex or type(v[0])==np.complex128:
        return np.sqrt(abs(v[0])**2+abs(v[1])**2)
    else:
        return np.sqrt(v[0]**2+v[1]**2)

def Norm3(v):
    if type(v[0])==complex or type(v[0])==np.complex128:
        return np.sqrt(abs(v[0])**2+abs(v[1])**2+abs(v[2])**2)
    else:
        return np.sqrt(v[0]**2+v[1]**2+v[2]**2)

def Norm(v):
    sum=0
    if type(v[0])==complex or type(v[0])==np.complex128:
        for i in range(len(v)):
            sum+=v[i].real**2+v[i].imag**2
    else:
        for i in range(len(v)):
            sum+=v[i]**2
    return np.sqrt(sum)

def ZFunc(z,f,f_param=dict()):
    try:
        if len(f_param)==0:
            return f(z)
        else:
            return f(z,f_param)
    except ZeroDivisionError:
        return np.inf

def DeltaX(x,delta=-8):
    return np.power(10,np.floor(np.log10(x))+delta)

def ComplexDrv(z,f,f_param=dict(),delta=1e-10):
    y=ZFunc(z,f,f_param)
    f_x=(ZFunc(z+complex(delta,0),f,f_param)-y)/complex(delta,0)
    return f_x

def LagrangeInterpolation(X,Y,x):
    s=0
    for i in range(len(X)):
        for j in range(len(X)):
            if i!=j:
                s+=(x-X[j])/(X[i]-X[j])*Y[i]
    return s

def ZPlot(f,space,f_param=dict(),plx=100,is_log=False,x_normalization=1,y_normalization=1,z_normalization=1,is_show=True,xlabel=r"$Re$",ylabel=r"$Im$",figsize=(10,5),is_fig_equal=True):
    rl_max=space[1].real
    im_max=space[1].imag
    rl_min=space[0].real
    im_min=space[0].imag

    X,Y=np.meshgrid(np.linspace(rl_min,rl_max,plx),np.linspace(im_min,im_max,plx))
    phase_colors=["navy","blue","blueviolet","purple","red","yellow","green","aqua","navy"]
    mag_colors=["black","grey","white"]
    mag_colormap=clrs.LinearSegmentedColormap.from_list("MagClrsMapping",mag_colors)
    phase_colormap=clrs.LinearSegmentedColormap.from_list("PhaseClrsMapping",phase_colors)

    Z=X+1j*Y
    
    # begin=time.time()
    ret_values=ZFunc(Z,f,f_param)/z_normalization

    # end=time.time()
    # print("Total Time consuming %fs"%(end-begin))
    # print(Z)
    
    

    M=np.abs(ret_values)
    P=np.angle(ret_values)
    

    
    fig,(ax1,ax2)=plt.subplots(1,2)
    fig.set_size_inches(10,5)

    if is_log:
        M=np.log10(M)
        ax1.set_title("Magnitude(Log10)",fontsize=20)
    else:
        ax1.set_title("Magnitude",fontsize=20)
    ax2.set_title("Phase",fontsize=20)

    ax1_loc=make_axes_locatable(ax1)
    ax2_loc=make_axes_locatable(ax2)
    ax1_cax=ax1_loc.append_axes('right',size='5%',pad=0.05)
    ax2_cax=ax2_loc.append_axes('right',size='5%',pad=0.05)


    # ax2=plt.subplot()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax1.set_xlim(rl_min/x_normalization,rl_max/x_normalization)
    ax1.set_ylim(im_min/y_normalization,im_max/y_normalization)
    ax2.set_xlim(rl_min/x_normalization,rl_max/x_normalization)
    ax2.set_ylim(im_min/y_normalization,im_max/y_normalization)


    pc1=ax1.pcolor(X/x_normalization,Y/y_normalization,M,cmap=mag_colormap)
    pc2=ax2.pcolor(X/x_normalization,Y/y_normalization,P,cmap=phase_colormap)

    ax1_clb=fig.colorbar(pc1,cax=ax1_cax,orientation='vertical')
    ax2_clb=fig.colorbar(pc2,cax=ax2_cax,orientation='vertical')

    ax2_clb.ax.set_ylim([-pi,pi])
    ax2_clb.ax.yaxis.set_ticks([-pi,0,pi])
    ax2_clb.ax.yaxis.set_ticklabels(["-π","0","π"])


    if is_fig_equal:
        ax1.set_aspect("equal")
        ax2.set_aspect("equal")

    if is_show==True:
        plt.show()
    
    return fig,(ax1,ax2)





def CalculateXY(x_list,get_y_List,base_num=100,is_save=False,model="",x_normalization=1,y_normalization=1,f_param={},is_one_by_one=False):

    
    X=np.linspace(x_list[0],x_list[1],base_num)
    
    
    if len(f_param)==0:
        if is_one_by_one:
            Y=np.vectorize(get_y_List)(X)
        else:
            Y=get_y_List(X)
    else:
        if is_one_by_one:
            Y=np.vectorize(get_y_List)(X,f_param)
        else:
            Y=get_y_List(X,f_param)
    
    for m in range(len(Y)):
        Y[m]/=y_normalization
    for m in range(len(X)):
        X[m]/=x_normalization

    if is_save:
        name='Y_of_'+model
        save_data={
            'name':name,
            'X':X,
            'Y':Y
        }
        MKOutput()
        with open('output/'+name+'.pickle','wb') as f:
            pickle.dump(save_data,f)


    return Y,X


def PlotXY(x_list,get_y_List,base_num=100,\
             xlabel="${x}$",ylabel="${E(x)}$(?eV)",title="",xlim=None,ylim=None,yticks_num=0,label_sz=16,font_sz=16,line_w=2,style="line",is_save=False,model="",is_multiproc=False,y_normalization=1,x_normalization=1,axes=None,f_param={},is_one_by_one=False,is_show=True):
    '''
        plot trajectory of energy bands vs x
        example:
            def GetE_BList(self,B_list,branch_num=0):
                branches=[[] for i in range(branch_num)]
                ...
                return branches
            xlabel='${B[T]}$'
            ylabel='${E(B))}$/MeV'
            title="Energy bands vs B"
            PlotCutPlane([0,10],GetE_BList,branch_num=200,base_num=200,xlabel=xlabel,ylabel=ylabel,title=title,ylim=[-1200,1200],style="black_line")
    '''

    if base_num!=0:
        if base_num<=1:
            base_num=2
        Y,X=CalculateXY(x_list,get_y_List,base_num,is_save,model,x_normalization,y_normalization,f_param,is_one_by_one)
    else:
        X,Y=np.array(x_list)/x_normalization,np.array(get_y_List)/y_normalization

    


    ##start drawing picture##

    ax=axes

    if ax==None:
        # plt.figure()
        ax = plt.axes()
        ax.tick_params(labelsize=label_sz)
        ax.set_xlabel(xlabel,fontsize=font_sz)
        ax.set_ylabel(ylabel,fontsize=font_sz)

        if xlim!=None:
            ax.set_xlim(xlim)
        if ylim!=None:
            ax.set_ylim(ylim)
            if yticks_num!=0:
                ax.set_yticks(np.linspace(ylim[0],ylim[1],yticks_num))
        
        ax.set_title(title,fontsize=font_sz+2)
    else:
        pass

            
    XYFigPlot(ax,style,X,Y,lw=line_w)
    if is_show==True:
        plt.show()
    return ax


def XYFigPlot(ax,style,X,Y,*arg,**kwargs):
    if style=="line":
        ax.plot(X,Y,*arg,**kwargs)
    elif style=="scatter":
        ax.scatter(X,Y,marker="+",s=10,c="grey",*arg,**kwargs)
    elif style=="blue_line":
        ax.plot(X,Y,c="blue",*arg,**kwargs)
    elif style=="black_line":
        ax.plot(X,Y,c="black",*arg,**kwargs)
    elif style=="dash_black_line":
        ax.plot(X,Y,c="black",linestyle="--")
    elif style=="blue_bar":
        ax.bar(X,Y,color="mediumblue",*arg,**kwargs)