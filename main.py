import numpy as np
import sympy
from scipy.optimize import minimize, Bounds
from scipy.optimize import curve_fit
import pandas as pd
import pickle

typename = 'test'
theta = sympy.symbols('theta')
nv = 60
samp = 1
NV = int(nv/samp)
n_m = 16
n_b = 10000
if typename == 'SEP':
    nn = np.int64(np.linspace(0, 100, 26))
    nn = np.sort(np.append(nn, [2,25,26,50,74,75,98]))
    nn = nn/2
    tt = nn * 0.02 * np.pi
else:
    nn = np.int64(np.linspace(0, 40, 41))
    tt = nn * 0.025 * np.pi
with open('./theory/'+typename+' p ideal.pickle', 'rb') as f:
    p_theory = pickle.load(f)
with open('./theory/'+typename+' FI ideal.pickle', 'rb') as f:
    FI_theory_ideal = pickle.load(f)
with open('./theory/'+typename+' FI noisy.pickle', 'rb') as f:
    FI_theory_noisy = pickle.load(f)


def f_p(params_, theta):
    return params_[0]*np.sin(4*theta+params_[1]) + params_[2]*np.sin(2*theta+params_[3]) + params_[4]

def f_p_sim(theta, A, B, C):
    return A*np.cos(4*theta+B) + A + C

def f_p_sep(theta, A, B, C):
    return A*np.cos(2*theta+B) + A + C

def f_minimize(params, x, y):
    w1 = 0.005
    w2 = 0.001
    ans = 0
    cons1 = 0
    cons2 = 0
    for i in range(np.size(x)):
        p_sum = 0
        for j in range(0,16):
            params_ = params[5*j:(5*j+5)]
            ans += (f_p(params_, x[i])-y[i,j])**2
            if f_p(params_, x[i]) < 0:
                cons1 += 1
            p_sum +=  f_p(params_, x[i])
        cons2 += (p_sum - 1)**2
    # print(cons1)
    # print(cons2)
    # print(ans)
    return ans + w1*cons1 + w2*cons2

def cal_p():
    count_raw = np.zeros((np.size(nn),n_m,nv))
    count = np.zeros((np.size(nn),n_m,NV))
    for i in range(np.size(nn)):
        for j in range(n_m):
            try:
                if nn[i]%1 == 0:
                    read_txt = pd.read_table('./rawdata/'+typename+'/%d %d.txt' % (nn[i], j+1), sep='\s+')
                else:
                    read_txt = pd.read_table('./rawdata/'+typename+'/%.1f %d.txt' % (nn[i], j+1), sep='\s+')
                count_raw[i, j, :] = read_txt['coincidence']
            except IOError:
                print(nn[i], j+1)
                count_raw[i, j, :] = np.linspace(0,0,nv)
            for k in range(NV):
                count[i, j, k] = np.sum(count_raw[i, j, (k*samp):((k+1)*samp)])
    p = np.zeros((np.size(nn), n_m, NV))
    p_ave = np.zeros((np.size(nn), n_m))
    p_std = np.zeros((np.size(nn), n_m))
    count_sum = np.zeros((np.size(nn),NV))
    for i in range(np.size(nn)):
        for j in range(n_m):
            for k in range(NV):
                count_sum[i,k] = np.sum(count[i,:,k])
                p[i,j,k] = count[i,j,k]/count_sum[i,k]
            p_ave[i,j] = np.mean(p[i,j,:])
            p_std[i,j] = np.std(p[i,j,:])
    count_sum_ave = np.mean(count_sum)
    # print(count_sum_ave)
    return count, p_ave, p_std, count_sum_ave
 
def cal_delta(count):
    theta_ans = np.zeros((np.size(nn), NV))
    theta_ave = np.linspace(0, 0, np.size(nn))
    theta_std = np.linspace(0, 0, np.size(nn))
    p_theory_log_arr = np.zeros((n_m, 3001))
    for i in range(np.size(nn)):
        theta_arr = np.linspace(-1500, 1500, 3001)*np.pi/40000 + tt[i]
        for j in range(n_m):
                if np.sum((sympy.lambdify(theta, p_theory[j], "numpy"))(tt)) >= 0.01:
                    p_theory_log_arr[j, :] = np.log((sympy.lambdify(theta, p_theory[j], "numpy"))(theta_arr))
        for k in range(NV):
            l_neg_arr = theta_arr * 0
            for j in range(n_m):
                if np.sum((sympy.lambdify(theta, p_theory[j], "numpy"))(tt)) >= 0.01:   
                    l_neg_arr -= p_theory_log_arr[j,:] * count[i,j,k]
            l_neg_arr = np.nan_to_num(l_neg_arr, nan=100000)
            theta_ans[i,k] = theta_arr[np.argmin(l_neg_arr)]
        theta_ave[i] = np.mean(theta_ans[i,:])
        theta_std[i] = np.std(theta_ans[i,:])
    theta_bootstrap = np.zeros((np.size(nn),n_b,NV))
    theta_std_bootstrap = np.zeros((np.size(nn),n_b))
    theta_std_ave = np.linspace(0, 0, np.size(nn))
    theta_std_std = np.linspace(0, 0, np.size(nn))
    for i in range(np.size(nn)):
        for j in  range(n_b):
            theta_bootstrap[i,j,:] = np.random.choice(theta_ans[i,:],size=NV)
            theta_std_bootstrap[i,j] = np.std(theta_bootstrap[i,j,:])
        theta_std_ave[i] = np.mean(theta_std_bootstrap[i,:])
        theta_std_std[i] = np.std(theta_std_bootstrap[i,:])
    return theta_std_ave, theta_std_std

def cal_FI(p_ave, typename):
    FI = 0 
    p_fit = p_theory*0
    if typename == 'test':
        p_parameter =  np.zeros((n_m*5, 1))
        x0 = []
        for i in range(0, n_m):  x0.extend([0,0,0,np.pi/4,0.25])
        res = minimize(f_minimize, x0, args=(tt, p_ave),method="powell")
        p_parameter = res.x
        for j in range(n_m):
            p_fit[j] = p_parameter[5*j]*sympy.sin(4*theta+p_parameter[5*j+1]) + p_parameter[5*j+2]*sympy.sin(2*theta+p_parameter[5*j+3]) + p_parameter[5*j+4]
            FI += (sympy.diff(p_fit[j], theta))**2 / p_fit[j]
    elif typename == 'SEP':
        p_parameter =  np.zeros((n_m, 2))
        for j in range(n_m):
            p_parameter = curve_fit(f_p_sep, tt, p_ave[:, j], bounds=((0, 0, 0), (10, np.pi, 10)))
            p_fit[j] = p_parameter[0][0]*sympy.cos(2*theta+p_parameter[0][1]) + p_parameter[0][0] + p_parameter[0][2]
            FI += (sympy.diff(p_fit[j], theta))**2 / p_fit[j]
    else:
        p_parameter =  np.zeros((n_m, 2))
        for j in range(n_m):
            p_parameter = curve_fit(f_p_sim, tt, p_ave[:, j], bounds=((0, -np.pi/2, 0), (1, np.pi/2, 0.1)))
            p_fit[j] = p_parameter[0][0]*sympy.cos(4*theta+p_parameter[0][1]) + p_parameter[0][0] + p_parameter[0][2]
            FI += (sympy.diff(p_fit[j], theta))**2 / p_fit[j]
    return FI, p_fit

def cal_singularity( p_theory):
    singularity = []
    visibility = ((tt*0) <= 1)
    ttt = 32
    t = np.int64(np.linspace(-4*ttt, 104*ttt, 108*ttt+1)) * 0.01 * np.pi / ttt
    for i in range(len(p_theory)):
        if np.sum((sympy.lambdify(theta, p_theory[i], "numpy"))(tt)) >= 0.1:
            for j in range(len(t)):
                if (sympy.lambdify(theta,p_theory[i],"numpy")(t[j])<=1e-7):
                    singularity.append(t[j])
            for j in range(len(tt)):
                if (sympy.lambdify(theta,p_theory[i],"numpy")(tt[j])<=1e-3):
                    visibility[j] = False
    singularity =  sorted(singularity)
    print(singularity)
    return singularity, visibility



def data_analysis(typename, boolfig):
    count, p_ave, p_std, count_sum_ave = cal_p()
    delta_ave, delta_std = cal_delta(count)
    FI, p_fit = cal_FI(p_ave, typename)
    singularity, visibility = cal_singularity(p_theory)
    if boolfig:
        from drawfig import fig_p, fig_delta, fig_FI
        fig_p(singularity, n_m, tt, p_ave, p_std, p_theory, typename)
        fig_delta(singularity, visibility, tt, delta_ave, delta_std, FI_theory_ideal, FI_theory_noisy, count_sum_ave, typename)
        fig_FI(singularity, visibility, tt, FI, FI_theory_ideal, FI_theory_noisy, typename)

if __name__=="__main__":
    data_analysis(typename, boolfig=True)