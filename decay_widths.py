import numpy as np
from alphas import alpha_s, alpha_s_vec
import sys
import os 
import rundec
from scipy.interpolate import CubicSpline

G_F=1.16637e-5 #from wikipedia in GeV^-2
theta=1

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

deltaEdata=np.loadtxt('decay_widths/deltaE.csv',delimiter=',')

def gg_LO(mh,mu=None,mq=mqHIGLU,loop_order=1,fixed_NF=3):
    if mu is None: mu=mh
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

def deltaE_higlu(x):
    interpolation = CubicSpline(deltaEdata[0,:],deltaEdata[1,:])
    return interpolation(x)

def gg_NLO(mh,mu,mq,deltaE=True,loop_order=2,fixed_NF=None):
    if deltaE: delta_E=deltaE_higlu(mh)
    else: delta_E=lambda _: 0
    E=95/4 - 7/6*NF + (33-2*NF)/6 * np.log(mu**2/mh**2) + delta_E
    als = alpha_s_vec(mu,mq,loop=loop_order,fixed_NF=fixed_NF)
    return gg_LO(mh,mu,mq,loop_order=loop_order,fixed_NF=fixed_NF)*(1+E*als/np.pi)

def gamma_gg(mh,order=4,nf=3,mu=None,deltaE=True):
    if mu is None: mu=mh
    als=alpha_s_vec(mu,mq=mqHIGLU,loop=order+1,fixed_NF=nf)
    lo=gg_LO(mh,mu,mqHIGLU,loop_order=order+1,fixed_NF=nf)
    if deltaE: dE=deltaE_higlu(mh)
    else: dE=0
    k1=(95/4 - 7/6*nf + (33-2*nf)/6 * np.log(mu**2/mh**2) + dE)/np.pi
    if order==0: return lo
    if order==1: return lo*(1+als*k1)
    if order==2:
        k2=K2(mh,mu,nf)
        return lo*(1+als*k1+als**2*k2)
    if order==3:
        k2=K2(mh,mu,nf)
        k3=K3(mh,mu,nf)
        return lo*(1+als*k1+als**2*k2+als**3*k3)
    if order==4:
        k2=K2(mh,mu,nf)
        k3=K3(mh,mu,nf)
        k4=K4(mh,mu,nf)
        return lo*(1+als*k1+als**2*k2+als**3*k3+als**4*k4)

def K2(mh,mu,nf):
    Lq=np.log(mh**2/mu**2)
    Lt=np.log(mh**2/mqHIGLU[2]**2)
    K2=0.00844343*(4442.35 - 2475.75*Lq + 272.25*Lq**2 + 28.5*Lt - 566.237*nf + 285.*Lq*nf - 33.*Lq**2*nf + 8.*Lt*nf + 10.82124*nf**2 - 7.*Lq*nf**2 + 1.*Lq**2*nf**2)
    return K2

def K3(mh,mu,nf):
    Lq=np.log(mh**2/mu**2)
    Lt=np.log(mu**2/mqHIGLU[2]**2)
    K3=0.000597251*(244806.7 - 243235.*Lq + 63762.2*Lq**2 - 4492.13*Lq**3 + 
            3599.63*Lt - 705.375*Lq*Lt + 352.688*Lt**2 - 57392.1*nf + 
            46866.8*Lq*nf - 11369.81*Lq**2*nf + 816.75*Lq**3*nf + 
            788.375*Lt*nf - 155.25*Lq*Lt*nf + 77.625*Lt**2*nf + 2841.571*nf**2 - 
            2431.331*Lq*nf**2 + 628.125*Lq**2*nf**2 - 49.5*Lq**3*nf**2 - 
            37.1875*Lt*nf**2 + 12.*Lq*Lt*nf**2 - 6.*Lt**2*nf**2 - 
            29.04301*nf**3 + 32.46373*Lq*nf**3 - 10.5*Lq**2*nf**3 + 
            1.*Lq**3*nf**3)
    return K3

def K4(mh,mu,nf):
    Lq=np.log(mh**2/mu**2)
    Lt=np.log(mu**2/mqHIGLU[2]**2)
    K4=  0.0000396064*(1.170328e7 - 1.875922e7*Lq + 8.68549e6*Lq**2 - 
            1.448098e6*Lq**3 + 74120.1*Lq**4 + 316300.4*Lt - 141623.3*Lq*Lt + 
            13966.42*Lq**2*Lt + 55824.9*Lt**2 - 9310.95*Lq*Lt**2 + 
            4655.47*Lt**3 - 4.48423e6*nf + 5.53678e6*Lq*nf - 
            2.219637e6*Lq**2*nf + 348161.*Lq**3*nf - 17968.5*Lq**4*nf + 
            38929.6*Lt*nf - 22335.6*Lq*Lt*nf + 2227.5*Lq**2*Lt*nf + 
            8877.82*Lt**2*nf - 1485.*Lq*Lt**2*nf + 742.5*Lt**3*nf + 
            412615.*nf**2 - 477271.*Lq*nf**2 + 188284.9*Lq**2*nf**2 - 
            30055.8*Lq**3*nf**2 + 1633.5*Lq**4*nf**2 - 8947.51*Lt*nf**2 + 
            3750.65*Lq*Lt*nf**2 - 423.9*Lq**2*Lt*nf**2 - 1202.212*Lt**2*nf**2 + 
            282.6*Lq*Lt**2*nf**2 - 141.3*Lt**3*nf**2 - 11022.76*nf**3 + 
            14282.55*Lq*nf**3 - 6232.1*Lq**2*nf**3 + 1096.1*Lq**3*nf**3 - 
            66.*Lq**4*nf**3 + 176.9759*Lt*nf**3 - 93.1*Lq*Lt*nf**3 + 
            14.4*Lq**2*Lt*nf**3 + 27.825*Lt**2*nf**3 - 9.6*Lq*Lt**2*nf**3 + 
            4.8*Lt**3*nf**3 + 65.1635*nf**4 - 116.172*Lq*nf**4 + 
            64.9275*Lq**2*nf**4 - 14.*Lq**3*nf**4 + 1.*Lq**4*nf**4)
    return K4

def gg_heavytop(mh,order,nf=3,mu=None):
    if mu is None: mu=mh
    als=alpha_s_vec(mu,mq=mqHIGLU,loop=order+1,fixed_NF=nf)
    lo=G_F*mh**3/36/np.pi**3/np.sqrt(2)*als**2
    Lq=np.log(mh**2/mu**2)
    Lt=np.log(mu**2/mqHIGLU[2]**2)
    k1=0.1061033*(71.25 - 16.5*Lq - 3.5*nf + 1.*Lq*nf)
    k2=0.00844343*(4442.35 - 2475.75*Lq + 272.25*Lq**2 + 28.5*Lt - 566.237*nf + 285.*Lq*nf - 33.*Lq**2*nf + 8.*Lt*nf + 10.82124*nf**2 - 7.*Lq*nf**2 + 1.*Lq**2*nf**2)
    k3=K3(mh,mu,nf)
    k4=K4(mh,mu,nf)
    if order==0: return lo
    if order==1: return lo*(1+als*k1)
    if order==2: return lo*(1+als*k1+als**2*k2)
    if order==3: return lo*(1+als*k1+als**2*k2+als**3*k3)
    if order==4: return lo*(1+als*k1+als**2*k2+als**3*k3+als**4*k4)

def gamma_ss(mh,mu=None,m_all_heavy_quarks=mqHIGLU,nf=3,order=4):
    if mu is None: mu=mh
    als=alpha_s_vec(mu,mq=mqHIGLU,loop=order+1,fixed_NF=nf)
    Lq=np.log(mh**2/mu**2)
    ms=0.0935
    alpha0=alpha_s(2,f=nf,mq=mqHIGLU,loop=order+1)
    if isinstance(mh,np.ndarray) or isinstance(mu,np.ndarray): ms_running=np.vectorize(lambda x: crd.mMS2mMS(ms,alpha0,alpha_s(x,nf,mq=m_all_heavy_quarks,loop=order+1),nf,order+1))(mu)
    else: ms_running=crd.mMS2mMS(ms,alpha0,alpha_s(mu,nf,mq=m_all_heavy_quarks,loop=order+1),nf,order+1)

    R1 = 0.079577472*als*(22.666667 - 2.*Lq)
    R2=0.006332574*als**2* (575.03938 - 1.*Lq*(2.9583333*(67.333333 - 2.2222222*nf) + 22.666667*(11. - 0.66666667*nf)) + Lq**2*(13. - 0.66666667*nf) - 21.738411*nf)
    R3=    0.00050393023*als**3*(10504.909 - 0.33333333*Lq**3*(4. + 6.*(11. - 0.66666667*nf) + 2.*(11. - 0.66666667*nf)**2) + 
        Lq**2*(147.33333 + 0.25*(67.333333 - 2.2222222*nf) + 68.*(11. - 0.66666667*nf) + 0.125*(67.333333 - 2.2222222*nf)*(11. - 0.66666667*nf) + 
        22.666667*(11. - 0.66666667*nf)**2 - 12.666667*nf) - 1649.3563*nf + 16.574354*nf**2 - 
        1.*Lq*(2.*(575.03938 - 21.738411*nf) + 22.666667*(102. - 12.666667*nf) + 2.8333333*(67.333333 - 2.2222222*nf) + 
        2.*(575.03938 - 21.738411*nf)*(11. - 0.66666667*nf) + 0.03125*(1249. - 146.18378*nf - 1.7283951*nf**2)))
    R4= 0.000040101493*als**4*(10070.239 + 0.16666667*Lq**4*(4. + 12.*(11. - 0.66666667*nf) + 11.*(11. - 0.66666667*nf)**2 + 3.*(11. - 0.66666667*nf)**3) - 
        0.33333333*Lq**3*(90.666667 + 6.*(102. - 12.666667*nf) + 0.75*(67.333333 - 2.2222222*nf) + 272.*(11. - 0.66666667*nf) + 
        5.*(102. - 12.666667*nf)*(11. - 0.66666667*nf) + 1.125*(67.333333 - 2.2222222*nf)*(11. - 0.66666667*nf) + 249.33333*(11. - 0.66666667*nf)**2 + 
        0.375*(67.333333 - 2.2222222*nf)*(11. - 0.66666667*nf)**2 + 68.*(11. - 0.66666667*nf)**3) - 56557.886*nf + 2479.3114*nf**2 - 5.2377231*nf**3 + 
        Lq**2*(1428.5 + 2.*(575.03938 - 21.738411*nf) + 68.*(102. - 12.666667*nf) + 5.6666667*(67.333333 - 2.2222222*nf) + 
        0.125*(102. - 12.666667*nf)*(67.333333 - 2.2222222*nf) + 0.0078125*(67.333333 - 2.2222222*nf)**2 + 5.*(575.03938 - 21.738411*nf)*(11. - 0.66666667*nf) + 
        56.666667*(102. - 12.666667*nf)*(11. - 0.66666667*nf) + 5.6666667*(67.333333 - 2.2222222*nf)*(11. - 0.66666667*nf) + 
        3.*(575.03938 - 21.738411*nf)*(11. - 0.66666667*nf)**2 - 279.61111*nf + 6.0185185*nf**2 + 0.0625*(1249. - 146.18378*nf - 1.7283951*nf**2) + 
        0.046875*(11. - 0.66666667*nf)*(1249. - 146.18378*nf - 1.7283951*nf**2)) - 
        1.*Lq*(2.*(575.03938 - 21.738411*nf)*(102. - 12.666667*nf) + 0.125*(575.03938 - 21.738411*nf)*(67.333333 - 2.2222222*nf) + 
        0.70833333*(1249. - 146.18378*nf - 1.7283951*nf**2) + 22.666667*(1428.5 - 279.61111*nf + 6.0185185*nf**2) + 
        2.*(10504.909 - 1649.3563*nf + 16.574354*nf**2) + 3.*(11. - 0.66666667*nf)*(10504.909 - 1649.3563*nf + 16.574354*nf**2) + 
        0.0078125*(25329.514 - 4891.5103*nf + 70.697613*nf**2 + 1.4830649*nf**3)))
    if order==0: R=3
    if order==1: R=3*(1+R1)
    if order==2: R=3*(1+R1+R2)
    if order==3: R=3*(1+R1+R2+R3)
    if order==4: R=3*(1+R1+R2+R3+R4)
    return G_F*mh*ms**2/4/np.sqrt(2)/np.pi*3*R

from scipy.special import spence

def Li(x):
    return spence(1-x)

def qq_LO(mh,mq):
    mD=1.865
    return np.where(mh>2*mD,3/4*G_F/np.sqrt(2)/np.pi*mq**2 * mh * beta(mD**2/mh**2)**3,0)

def beta(x):
    return np.sqrt(1-4*x)

def delta(beta):
    return 4/3*(alpha(beta)/beta + (3+34*beta**2-13*beta**4)/16/beta**3*np.log(gamma(beta)) + (21*beta**2-3)/8/beta**2)

def alpha(beta):
    return (1+beta**2)*(4*Li(1/gamma(beta)) + 2*Li(-1/gamma(beta)) - np.log(gamma(beta))*np.log(beta**2/(1+beta)**3)) - beta*np.log(64*beta**4/(1-beta**2)**3)

def gamma(beta):
    return (1+beta)/(1-beta)

def gamma_cc(mh,mu=None,mq=mqHIGLU[0],m_all_heavy_quarks=mqHIGLU,fixed_NF=3):
    if mu is None: mu=mh
    mD=1.865
    return np.where(mh>2*mD,qq_LO(mh,mq)*(1+alpha_s_vec(mu,m_all_heavy_quarks,2,fixed_NF=fixed_NF)/np.pi*(delta(beta(mq**2/mh**2))+2*np.log(mu/mh))),0)


data_mu_tau1 = np.loadtxt('decay_widths/br.sm1',skiprows=3)
data_mu_tau2 = np.loadtxt('decay_widths/br.sm2',skiprows=3)
tau_interpolation= CubicSpline(data_mu_tau1[:,0],data_mu_tau1[:,2]*data_mu_tau2[:,-1])
mu_interpolation= CubicSpline(data_mu_tau1[:,0],data_mu_tau1[:,3]*data_mu_tau2[:,-1])

def gamma_mu(m):
    return np.where(m>data_mu_tau1[0,0],mu_interpolation(m),0)

def gamma_tau(m):
    mtau=1.77682
    return np.where(m>2*mtau,tau_interpolation(m),0)

def gamma_hadrons(m):
    gg=gamma_gg(m)
    ss=gamma_ss(m)
    cc=gamma_cc(m)
    return gg+ss+cc

def gamma_total(m):
    hadrons=gamma_hadrons(m)
    return gamma_mu(m)+gamma_tau(m)+hadrons