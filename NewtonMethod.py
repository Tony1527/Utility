from scipy.misc import derivative
from numpy import array
import matplotlib.pyplot as plt
import numpy as np
import os
import time

#牛顿方法，求解函数的解
#x0 :  初始点
#f  :  函数对象
#error:误差
def NewtonMethod(x0,f,error=1e-10,step=1e-8,is_plot=False,max_iter=20,is_full_output=False,is_total_print=True,**kwarg):
    if not hasattr(f,'__call__'):
        raise RuntimeError('f is not a function')
    if len(kwarg)!=0:
        def wrap_f(x):
            return f(x,**kwarg)
    else:
        def wrap_f(x):
            return f(x)
    
    if is_total_print:
        print("NewtonMethod begins:")
    n=0                             #round
    fi = wrap_f(x0)                      #function
    gi = derivative(wrap_f,x0,dx=step)   #derivative
    result = np.array([[n,x0,fi,gi]])
    if is_total_print:
        print(str(n)+" "+str(x0)+" "+str(fi))
    xi = x0
    cnt=1
    while abs(fi)>error and cnt<=max_iter:            #iterate until  abs(fi) <= error
        n=n+1
        if gi==0:
            if is_total_print:
                print("zero slope encountered!")
            break

        xi = xi - fi/gi             #New Method Key , linear approximation
        
        fi = wrap_f(xi)
        gi = derivative(wrap_f,xi,dx=step)
        if is_total_print:
            print(str(n)+" "+str(xi)+" "+str(fi))
        result = np.concatenate([result,np.array([[n,xi,fi,gi]])],axis=0)
        cnt+=1
    
    if is_plot:
        xi = result[:,1]
        yi = result[:,2]
        dxi = result[:,3]
        max_x = max(abs(xi))
        x=np.arange(-max_x*0.9,max_x*1.1,max_x/100)         #此处如果采用绝对步长有非常大的效率隐患
        y=wrap_f(x)
        if is_total_print:
            print(len(x))
        # time.sleep(3)
        #展示前min(4,n)步操作
        if n<4:
            for i in range(0,n):
                
                last_x = xi[i]                                  #当前要展示的点(xi,yi)
                last_y = yi[i]                              
                last_dx = dxi[i]
                last_tangent_line = last_y+last_dx*(x-last_x)   #切线


                past_xlist = [xi[m] for m in range(i)]          #newton方法迭代过的点
                past_ylist = [yi[m] for m in range(i)]

                
                plt.figure(i+1)
                plt.plot(x,y,'-b',past_xlist,past_ylist,'*k',last_x,last_y,'*r',x,last_tangent_line,'-.r',x,np.zeros(len(x)),'-k')
                plt.show()
        else:
            plt.figure(1)
            for i in range(0,4):
                last_x = xi[i]                                  #当前要展示的点(xi,yi)
                last_y = yi[i]                              
                last_dx = dxi[i]
                last_tangent_line = last_y+last_dx*(x-last_x)   #切线


                past_xlist = [xi[m] for m in range(i)]          #newton方法迭代过的点
                past_ylist = [yi[m] for m in range(i)]

                ax = plt.subplot(2,2,i+1)
                ax.plot(x,y,'-b',past_xlist,past_ylist,'*k',last_x,last_y,'*r',x,last_tangent_line,'-.r',x,np.zeros(len(x)),'-k')

            #展示整个迭代点
            plt.figure(2)
            plt.plot(x,y,'-b',xi,yi,'*k',xi[-1],yi[-1],'or',x,np.zeros(len(x)),'-k')
            plt.show()
            plt.close()
    
    if is_full_output:
        return result
    else:
        return result[-1][1]

# def g(x,f_param):
#     return 2*x**3+5*x**2-3.4*x-1.1+f_param

# NewtonMethod(0,g,is_plot=True,f_param=5)