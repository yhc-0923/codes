import numpy as np
import sympy
from matplotlib import pyplot as plt

linewidth_line = 2
markersize = 6
capsize = 4
capthick = 1.5
linewidth_gca = 1.5
fontsize_gca = 28
fontsize_label = 28
fontsize_legend = 28
fontname = 'Times New Roman'

theta = sympy.symbols('theta')
ttt = 32
t = np.int64(np.linspace(-2*ttt, 102*ttt, 104*ttt+1)) * 0.01 * np.pi / ttt

def fig_p(singularity, n_m, X, Y, Y_std, Y_theory, typename):
    useful_pm = []
    for j in range(n_m):    
        if np.sum((sympy.lambdify(theta, Y_theory[j], "numpy"))(X)) >= 0.1:
            useful_pm.append(j)

    plt.figure(figsize=(8, 5))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    cc = 1
    for j in range(len(useful_pm)): 
        ax = plt.subplot(len(useful_pm), 1, cc,facecolor='none')
        ax.plot(t, (sympy.lambdify(theta, Y_theory[useful_pm[j]], "numpy"))(t), color="#EEAD0E", linewidth=linewidth_line,zorder=2)
        ax.errorbar(X, Y[:, useful_pm[j]], Y_std[:, useful_pm[j]], fmt='o', mec="#000000",color="#d16e00", ecolor="#000000", capsize=capsize, capthick=capthick, linewidth=capthick, markersize=markersize, zorder=cc+2, clip_on=False)
        for i in range(len(singularity)):
            ax.axvline(singularity[i], color="#bf7fc3",linewidth=linewidth_line, linestyle="--",zorder=0)
        if cc > 1:
            ax.spines['top'].set_visible(False)
        ax.set_xlim([-0.02*np.pi, 1.02*np.pi])
        ax.set_ylim([0, 0.7])
        if cc < len(useful_pm):
            ax.set_xticks([])
        else:
            xt = np.linspace(0,8,9)*np.pi/8
            plt.xticks(xt,
               ['0',r'$\frac{\pi}{8}$',r'$\frac{\pi}{4}$',r'$\frac{3\pi}{8}$',r'$\frac{\pi}{2}$',r'$\frac{5\pi}{8}$',r'$\frac{3\pi}{4}$',r'$\frac{7\pi}{8}$',r'$\pi$'],
               fontsize=fontsize_gca, fontname=fontname)
        if typename == '2D':
            ax.set_yticks([0,0.2,0.4])
        elif typename =='test':
            ax.set_yticks([0,0.3])
        else:
            ax.set_yticks([0,0.2])
        plt.yticks(fontsize=24, fontname=fontname)
        ax.grid(False)
        plt.ylabel(r'$P_{%d}$' % (useful_pm[j]+1), fontsize=24, fontname=fontname)
        cc += 1
    plt.xlabel(r'$\theta$', fontsize=fontsize_label, fontname=fontname, loc='right')
    plt.gca().xaxis.set_label_coords(1.04, 0.05)

    if (typename == '2D') or (typename == 'test'):
        plt.tight_layout(h_pad=-2)
    else:
        plt.tight_layout(h_pad=-4)
    plt.savefig('./figures/'+typename+' P.pdf', dpi=350, bbox_inches='tight')
    plt.show()
    return 0

def fig_delta(singularity, visibility, X, Y, Y_std, FI_theory_ideal, FI_theory_noisy, count_sum_ave, typename):
    plt.figure(figsize=(8, 5))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

    FI_sum_ideal = np.sqrt(sympy.lambdify(theta,FI_theory_ideal,"numpy")(t)*count_sum_ave)
    FI_sum_ideal[FI_sum_ideal==np.inf] = np.nan
    FI_sum_noisy=np.sqrt(sympy.lambdify(theta,FI_theory_noisy,"numpy")(t)*count_sum_ave)
    FI_sum_noisy[FI_sum_noisy<=1e-10] = 1e-10

    plt.plot(t, t*0+1/np.sqrt(8*count_sum_ave), color="#2E8B57", linewidth=linewidth_line)
    plt.plot(t, t*0+1/np.sqrt(16*count_sum_ave), color="#CD5555", linewidth=linewidth_line)
    plt.plot(t, 1/FI_sum_ideal, color="#EEAD0E", linewidth=linewidth_line)
    plt.plot(t, 1/FI_sum_noisy, color="#33a7d1", linewidth=linewidth_line)
    plt.errorbar(X[visibility], Y[visibility], Y_std[visibility], fmt='o', mec="#000000", color="#d16e00", ecolor="#000000", capsize=capsize, capthick=capthick, linewidth=capthick, markersize=markersize, zorder=3, clip_on=False)

    for i in range(len(singularity)):
        plt.axvline(singularity[i], color="#bf7fc3",linewidth=linewidth_line, linestyle="--")

    plt.xlim([-0.02*np.pi, 1.02*np.pi])
    plt.ylim([0,0.02])
    plt.xlabel(r'$\theta$', fontsize=fontsize_label, fontname=fontname, loc='right')
    plt.gca().xaxis.set_label_coords(1.04, 0.05)
    plt.ylabel(r'$\delta\phi$', fontsize=fontsize_label, fontname=fontname)
    plt.grid(True, linestyle='--', linewidth=linewidth_gca, color='#d6d6d6')
    plt.gca().tick_params(axis='both', which='major', labelsize=fontsize_gca, width=linewidth_gca)
    plt.gca().tick_params(axis='both', which='minor', labelsize=fontsize_gca, width=linewidth_gca)
    xt = np.linspace(0,8,9)*np.pi/8
    plt.xticks(xt,
               ['0',r'$\frac{\pi}{8}$',r'$\frac{\pi}{4}$',r'$\frac{3\pi}{8}$',r'$\frac{\pi}{2}$',r'$\frac{5\pi}{8}$',r'$\frac{3\pi}{4}$',r'$\frac{7\pi}{8}$',r'$\pi$'],
               fontsize=fontsize_gca, fontname=fontname)
    plt.yticks(fontsize=fontsize_gca, fontname=fontname)
    plt.gca().spines['bottom'].set_linewidth(linewidth_gca)
    plt.gca().spines['top'].set_linewidth(linewidth_gca)
    plt.gca().spines['left'].set_linewidth(linewidth_gca)
    plt.gca().spines['right'].set_linewidth(linewidth_gca)
    plt.gca().tick_params(direction='in',axis='both', which='major', labelsize=fontsize_gca, width=linewidth_gca)
    plt.gca().tick_params(direction='in',axis='both', which='minor', labelsize=fontsize_gca, width=linewidth_gca)

    plt.savefig('./figures/'+typename+' delta.pdf', dpi=350, bbox_inches='tight')
    plt.show()
    return 0

def fig_FI(singularity, visibility, X, FI, FI_theory_ideal, FI_theory_noisy, typename):
    fig, ax = plt.subplots(figsize=(8, 5))    
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    
    FI_ideal = (sympy.lambdify(theta, FI_theory_ideal, "numpy"))(t)
    FI_ideal[FI_ideal==np.inf] = np.nan
    FI_noisy = (sympy.lambdify(theta, FI_theory_noisy, "numpy"))(t)

    ax.plot(t, t*0+8, color='#2E8B57', linewidth=linewidth_line)
    ax.plot(t, t*0+16, color='#CD5555', linewidth=linewidth_line)
    ax.plot(t, FI_ideal , color="#EEAD0E", linewidth=linewidth_line)
    ax.plot(t, FI_noisy, color="#33a7d1", linewidth=linewidth_line)


    ax.plot(X[visibility], (sympy.lambdify(theta, FI, "numpy"))(X)[visibility], mec="#000000", color='#d16e00', marker='o', linestyle='none')
    
    for i in range(len(singularity)):
        ax.axvline(singularity[i], color="#bf7fc3",linewidth=linewidth_line, linestyle="--")

    ax.set_xlim([-0.02*np.pi, 1.02*np.pi])
    ax.set_ylim([0, 17])

    ax.set_xlabel(r'$\theta$', fontsize=fontsize_label, fontname=fontname, loc='right')
    ax.xaxis.set_label_coords(1.04, 0.05)
    ax.set_ylabel('FI', fontsize=fontsize_label, fontname=fontname)
    ax.grid(True, linestyle='--', linewidth=linewidth_gca, color='#d6d6d6')
    ax.tick_params(axis='both', which='major', labelsize=fontsize_gca, width=linewidth_gca)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize_gca, width=linewidth_gca)
    xt = np.linspace(0,8,9)*np.pi/8
    plt.xticks(xt,
               ['0',r'$\frac{\pi}{8}$',r'$\frac{\pi}{4}$',r'$\frac{3\pi}{8}$',r'$\frac{\pi}{2}$',r'$\frac{5\pi}{8}$',r'$\frac{3\pi}{4}$',r'$\frac{7\pi}{8}$',r'$\pi$'],
               fontsize=fontsize_gca, fontname=fontname)
    plt.yticks(np.linspace(0,16,5),fontsize=fontsize_gca, fontname=fontname)
    ax.spines['bottom'].set_linewidth(linewidth_gca)
    ax.spines['top'].set_linewidth(linewidth_gca)
    ax.spines['left'].set_linewidth(linewidth_gca)
    ax.spines['right'].set_linewidth(linewidth_gca)
    ax.tick_params(direction='in',axis='both', which='major', labelsize=fontsize_gca, width=linewidth_gca)
    ax.tick_params(direction='in',axis='both', which='minor', labelsize=fontsize_gca, width=linewidth_gca)

    fig.savefig('./figures/'+typename+' FI.pdf', dpi=350, bbox_inches='tight')
    fig.show()
    return 0