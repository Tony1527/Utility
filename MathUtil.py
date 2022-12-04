import numpy as np






c_j = complex(0, 1)
pi = np.pi


def Norm2(v):
    if type(v[0]) == complex or type(v[0]) == np.complex128:
        return np.sqrt(abs(v[0]) ** 2 + abs(v[1]) ** 2)
    else:
        return np.sqrt(v[0] ** 2 + v[1] ** 2)


def Norm3(v):
    if type(v[0]) == complex or type(v[0]) == np.complex128:
        return np.sqrt(abs(v[0]) ** 2 + abs(v[1]) ** 2 + abs(v[2]) ** 2)
    else:
        return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def Norm(v):
    sum = 0
    if type(v[0]) == complex or type(v[0]) == np.complex128:
        for i in range(len(v)):
            sum += v[i].real ** 2 + v[i].imag ** 2
    else:
        for i in range(len(v)):
            sum += v[i] ** 2
    return np.sqrt(sum)


def ZFunc(z, f, **f_param):
    try:
        return f(z, **f_param)
    except ZeroDivisionError:
        return np.inf


def DeltaX(x, delta=-8):
    return np.power(10, np.floor(np.log10(x)) + delta)








