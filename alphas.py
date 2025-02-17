import numpy as np
import rundec

def alpha_s_vec(mu,mq,loop,fixed_NF=None,alphasMZ=0.1185,MZ=91.1876):
    mC=mq[0]
    mB=mq[1]
    mT=mq[2]
    if isinstance(mu,np.ndarray):
        result = np.zeros(mu.shape)
        als_vec=np.vectorize(lambda x,y: alpha_s(x,y,mq,loop,alphasMZ=alphasMZ,MZ=MZ),otypes=[float])
        if fixed_NF is None:
            result[mu>=mT]=als_vec(mu[mu>=mT],6)
            result[np.logical_and(mB<=mu,mu<mT)]=als_vec(mu[np.logical_and(mB<=mu,mu<mT)],5)
            result[np.logical_and(mC<=mu,mu<mB)]=als_vec(mu[np.logical_and(mC<=mu,mu<mB)],4)
            result[mu<mC]=als_vec(mu[mu<mC],3)
        else: result = als_vec(mu,fixed_NF)
    else:
        if fixed_NF is None:
            if mu>=mT: result = alpha_s(mu,6,mq,loop,alphasMZ=alphasMZ,MZ=MZ)
            elif mB<=mu<mT: result = alpha_s(mu,5,mq,loop,alphasMZ=alphasMZ,MZ=MZ)
            elif mC<=mu<mB: result = alpha_s(mu,4,mq,loop,alphasMZ=alphasMZ,MZ=MZ)
            elif mu<mC: result = alpha_s(mu,3,mq,loop,alphasMZ=alphasMZ,MZ=MZ)
        else: result = alpha_s(mu,fixed_NF,mq,loop,alphasMZ=alphasMZ,MZ=MZ)
    return result

def alpha_s(scale,f,mq,loop,alphasMZ=0.1185,MZ=91.1876):
        """3-loop computation of alpha_s for f flavours
        with initial condition alpha_s(MZ) = 0.1185"""
        #taken from scalar_portal https://github.com/JLTastet/scalar_portal/blob/master/data/qcd.py
        mT=mq[2]; mB=mq[1]; mC=mq[0]
        crd = rundec.CRunDec()
        if f == 5:
            return_value = crd.AlphasExact(alphasMZ, MZ, scale, int(f), int(loop))
        elif f == 6:
            crd.nfMmu.Mth = mT
            crd.nfMmu.muth = mT
            crd.nfMmu.nf = 6
            return_value = crd.AlL2AlH(alphasMZ, MZ, crd.nfMmu, scale, loop)
        elif f == 4:
            crd.nfMmu.Mth = mB
            crd.nfMmu.muth = mB
            crd.nfMmu.nf = 5
            return_value = crd.AlH2AlL(alphasMZ, MZ, crd.nfMmu, scale, loop)
        elif f == 3:
            crd.nfMmu.Mth = mB
            crd.nfMmu.muth = mB
            crd.nfMmu.nf = 5
            asmc =  crd.AlH2AlL(alphasMZ, MZ, crd.nfMmu, mC, loop)
            crd.nfMmu.Mth = mC
            crd.nfMmu.muth = mC
            crd.nfMmu.nf = 4
            return_value = crd.AlH2AlL(asmc, mC, crd.nfMmu, scale, loop)
        else:
            assert(False) # pragma: no cover
        if return_value == 0:
            raise ValueError("Return value is 0, probably `scale={}` is too small.".format(scale))
        else:
            return return_value


