from numpy import sin, cos, divide, division, seterr, tanh, tan, asarray, atleast_1d
from my_operators import safe_div, mylog, mysqrt, mypower2, mypower3
from operator import add, sub, mul
from g_address import get_address




def eval_(strg, x, *p):
    seterr(divide='ignore', invalid='ignore')
    try:
        x_r = eval(strg)
    except TypeError:
        print 'Error.', strg
    return x_r

# xdata, ydata = get_address(p, n, problem, direccion)
# params=[0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]
#
# n_params = asarray(params)
# ind = '(p[0]+(p[1]*(p[2]*((p[3]*sin((p[4]*((p[5]*mylog((p[6]*x[3])))+(p[7]*((p[8]*mylog((p[9]*mypower2((p[10]*x[2])))))+(p[11]*((p[12]*0.09912242223887513)-(p[13]*cos((p[14]*((p[15]*x[5])-(p[16]*x[3])))))))))))))+(p[17]*cos((p[18]*sin((p[19]*tan((p[20]*((p[21]*sin((p[22]*x[5])))-(p[23]*tan((p[24]*-0.3453521693130588)))))))))))))))'
# xr=eval_(ind, xdata, n_params)
# print xr