import numpy as np
from alphas import alpha_s, alpha_s_vec
import sys
import os 
import rundec
from scipy.interpolate import CubicSpline

G_F=1.16637e-5 #from wikipedia in GeV^-2
theta=1
NF=3

#pole masses from particle data group:
# mT=172.45, mB=4.78, mC=1.67
mqOS=[1.67,4.78,172.45]
#MSbar masses mC(mC), mB(mB), pole mass mT from pdg
mqpdg=[1.27,4.18,172.45]
#for HIGLU (or HDECAY) mc(3GeV) is needed
crd = rundec.CRunDec()
mc3GeV = crd.AsmMSrunexact(1.27,0.38,1.27,3.0,3,4).mMSexact #run pdg value to 3gev
mqMS=[mc3GeV,4.18,172.45]
#HDECAY calculates OS mass from this input, which is printed to higlu.out (should probably automate readout of this)
mqHIGLU=[1.43131,4.84131,172.45]

deltaEdata=np.loadtxt('deltaE.csv',delimiter=',')

data_mu_tau1 = np.loadtxt('/decay_widths/br.sm1',skiprows=3)
data_mu_tau2 = np.loadtxt('/decay_widths/br.sm2',skiprows=3)

tau_interpolation= CubicSpline(data_mu_tau1[:,0],data_mu_tau1[:,2]*data_mu_tau2[:,-1])
mu_interpolation= CubicSpline(data_mu_tau1[:,0],data_mu_tau1[:,3]*data_mu_tau2[:,-1])

def gg_LO(mh,mu,mq,loop_order=1,fixed_NF=None):
    #x=np.expand_dims(mh,axis=1)**2/(4*np.expand_dims(mq,axis=0)**2)
    return theta* alpha_s_vec(mu,mq,loop=loop_order,fixed_NF=fixed_NF)**2*G_F*np.sqrt(2)/32/np.pi**3 * mh**3 * np.abs(I(mh**2/4/mq[0]**2) + I(mh**2/4/mq[1]**2) + I(mh**2/4/mq[2]**2))**2 #np.abs(np.sum(I(x),axis=1))**2

def I(x):
    return 1/x**2 * (x + (x-1)*f(x))
def f(x):
    if isinstance(x,np.ndarray): 
        x=x+0j
        return np.piecewise(x,[x<=1,x>1], [lambda y: np.arcsin(np.sqrt(y))**2,lambda y: -0.25*(np.log((1+np.sqrt(1-1/y))/(1-np.sqrt(1-1/y))) - np.pi*1j)**2])
    else:
        if x<=1: return np.arcsin(np.sqrt(x))**2
        else: return -0.25*(np.log((1+np.sqrt(1-1/x))/(1-np.sqrt(1-1/x))) - np.pi*1j)**2

def deltaE_higlu():
    interpolation = CubicSpline(deltaEdata[0,:],deltaEdata[1,:])
    return interpolation

def gg_NLO(mh,mu,mq,deltaE=True,loop_order=2,fixed_NF=None):
    if deltaE: delta_E=deltaE_higlu()
    else: delta_E=lambda _: 0
    E=95/4 - 7/6*NF + (33-2*NF)/6 * np.log(mu**2/mh**2) + delta_E(mh)
    als = alpha_s_vec(mu,mq,loop=loop_order,fixed_NF=fixed_NF)
    return gg_LO(mh,mu,mq,loop_order=loop_order,fixed_NF=fixed_NF)*(1+E*als/np.pi)

def relative_gg_NLO(mh,mu,mq,deltaE=True,fixed_NF=None):
    if deltaE: delta_E=deltaE_higlu()
    else: delta_E=lambda _: 0
    E=95/4 - 7/6*NF + (33-2*NF)/6 * np.log(mu**2/mh**2) + delta_E(mh)
    als = alpha_s_vec(mu,mq,loop=2,fixed_NF=fixed_NF)
    return (E*als/np.pi)

def gg_NNLO(mh,mu=None,mq=mqHIGLU,fixed_NF=None):
    if mu is None: mu=mh
    Z3=1.20206
    lmt=np.log(mu**2/mq[2]**2)
    lmh=np.log(mu**2/mh**2)
    K2 = (149533/288 + 3301/16 * lmh + 363/16*lmh**2 + 19/8*lmt - 121/16*np.pi**2 - 495/8*Z3 +
            NF**2*(127/108 + 7/12*lmh + lmh**2/12 - np.pi**2/36) + 
            NF*(-4157/72 - 95/4*lmh - 11/4*lmh**2 + 2/3*lmt + 11/12*np.pi**2 + 5/4*Z3))
    return gg_NLO(mh,mu,mq,loop_order=3,fixed_NF=fixed_NF)+gg_LO(mh,mu,mq,loop_order=3,fixed_NF=fixed_NF)*K2*(alpha_s_vec(mu,mq,loop=3,fixed_NF=fixed_NF)/np.pi)**2

from scipy.special import spence

def Li(x):
    return spence(1-x)

def qq_LO(mh,mq):
    return 3/4*G_F/np.sqrt(2)/np.pi*mq**2 * mh * beta(mq**2/mh**2)**3

def beta(x):
    return np.sqrt(1-4*x)

def delta(beta):
    return 4/3*(alpha(beta)/beta + (3+34*beta**2-13*beta**4)/16/beta**3*np.log(gamma(beta)) + (21*beta**2-3)/8/beta**2)

def alpha(beta):
    return (1+beta**2)*(4*Li(1/gamma(beta)) + 2*Li(-1/gamma(beta)) - np.log(gamma(beta))*np.log(beta**2/(1+beta)**3)) - beta*np.log(64*beta**4/(1-beta**2)**3)

def gamma(beta):
    return (1+beta)/(1-beta)

def qq_NLO(mh,mu,mq,m_all_heavy_quarks,fixed_NF=5):
    return qq_LO(mh,mq)*(1+alpha_s_vec(mu,m_all_heavy_quarks,2,fixed_NF=fixed_NF)/np.pi*(delta(beta(mq**2/mh**2)))+2*np.log(mu/mh))


def ssbar_NNLO(mh,mu=None,m_all_heavy_quarks=mqHIGLU):
    if mu is None: mu=mh
    ms=0.0935
    alpha0=alpha_s(2,f=4,mq=mqHIGLU,loop=4)
    if isinstance(mh,np.ndarray): ms_running=np.vectorize(lambda x: crd.mMS2mMS(ms,alpha0,alpha_s(x,4,mq=m_all_heavy_quarks,loop=3),4,3))(mu)
    else: ms_running=crd.mMS2mMS(ms,alpha0,alpha_s(mu,4,mq=m_all_heavy_quarks,loop=3),4,3)
    lmh=np.log(mh**2/mu**2)
    Z3=1.20206
    NF=4
    mt=m_all_heavy_quarks[2]
    nlo=-2*lmh+17/3
    nnlo=(lmh**2 * (19/4 - NF/6) + lmh * (11*NF/9 - 106/3) + NF * (2*Z3/3 - 65/24) +
            np.pi**2 * (NF/18 - 19/12) - 39*Z3/2 + 10801/144 +
            1/9 * np.log(ms_running**2/mh**2)**2 - 2/3 * np.log(mh**2/mt**2) + 8/3 - np.pi**2/9)
    if isinstance(mh,np.ndarray):
        alpha=alpha_s_vec(mu,m_all_heavy_quarks,3)
    else:
        alpha=alpha_s(mu,4,mq=m_all_heavy_quarks,loop=3)
    return qq_LO(mh,ms_running)*(1+alpha/np.pi*nlo + (alpha/np.pi)**2*nnlo)

def ccbar_NNLO(mh,mu,m_all_heavy_quarks=mqHIGLU):
    mc=1.27
    alpha0=alpha_s(1.27,f=4,mq=mqHIGLU,loop=4)
    mc_running=np.vectorize(lambda x: crd.mMS2mMS(mc,alpha0,alpha_s(x,4,mq=m_all_heavy_quarks,loop=3),4,3))(mu)
    mt=m_all_heavy_quarks[2]
    lmh=np.log(mh**2/mu**2)

    nlo=17/3-2*lmh-6*(20/3-4*lmh)*mc_running**2/mh**2

    Z3=1.20206
    NF=4
    sum_value=1
    nnlo=(11185/144 + lmh**2 * (19/4 - NF/6) + lmh * (-(106/3) + (11 * NF)/9) - np.pi**2/9 + 
          (-19/12 + NF/18) * np.pi**2 + 1/9 * np.log(mc_running**2 / mh**2)**2 - 
          2/3 * np.log(mh**2 / mt**2) + NF * (-65/24 + 2*Z3/3) - 39 * Z3/2 +
          mc_running**2/mh**2 * (-(10/3) - (2 * np.pi**2)/9 - np.pi**4/12 + 16/9 * np.log(mc_running**2 / mh**2) - 
          4/9 * np.log(mc_running**2 / mh**2)**2 + (2/3 - np.pi**2/6) * np.log(mc_running**2 / mh**2)**2 - 
          1/12 * np.log(mc_running**2 / mh**2)**4 + 4 * np.log(mh**2 / mt**2) - 
          6 * (2383/24 + 
          lmh**2 * (27/2 - NF/3) + lmh * (-(371/6) + (5 * NF)/3) + (-(9/2) + NF/9) * np.pi**2 - (2 * sum_value)/3 + 
          NF * (-(313/108) + (2 * Z3)/3) - (83 * Z3)/3)))
          
    if isinstance(mh,np.ndarray):
        alpha=alpha_s_vec(mu,m_all_heavy_quarks,3)
    else:
        alpha=alpha_s(mu,4,mq=m_all_heavy_quarks,loop=3)
    return qq_LO(mh,ms_running)*(1+alpha/np.pi*nlo + (alpha/np.pi)**2*nnlo)

def gamma_hadrons(m):
    return gg_NNLO(m) + ssbar_NNLO(m)

def muons(m):
    return mu_interpolation(m)

def tau(m):
    return tau_interpolation(m)

def gamma_total(m):
    return gamma_hadrons(m)+muons(m)+tau(m)

