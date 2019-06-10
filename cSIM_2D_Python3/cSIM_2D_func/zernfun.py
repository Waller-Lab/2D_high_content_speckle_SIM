import numpy as np

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def zernfun(n, m, r, theta):
    
    '''
    
    Translation from MATLAB code in https://www.mathworks.com/matlabcentral/fileexchange/7687-zernike-polynomials by Li-Hao Yeh.
    
    '''
    
    # Check and prepare the inputs
    
    if (not n.ndim==1 or not m.ndim==1) and (not n.ndim==0 or not m.ndim==0):
        raise Exception('zernfun: NMlength, N and M must be vectors.')

    n = n.ravel()
    m = m.ravel()
    
    if len(n) != len(m):
        raise Exception('zernfun: NMlength, N and M must be the same length.')

    


    if any(np.mod(n-m,2) == 1):
        raise Exception('zernfun: NMmultiplesof2, All N and M must differ by multiples of 2 (including 0).')

    if any(m>n):
        raise Exception('zernfun: MlessthanN, Each M must be less than or equal to its correspoding N.')

    if any(np.logical_or(r.ravel()>1, r.ravel()<0)):
        raise Exception('zernfun: Rlessthan1, All R must be between 0 and 1.')


    if not r.ndim==1 or not theta.ndim==1:
        raise Exception('zernfun: RTHvector, R and THETA must be vectors.')


    r = r.ravel()
    theta = theta.ravel()

    length_r = len(r)

    if length_r != len(theta):
        raise Exception('zernfun: RTHlength, The number of R- and THETA-values must be equal.')

    ####################################
    # Compute the Zernike Polynomials  #
    ####################################


    # Determine the required powers of r:
    #--------------------------------------
    m_abs = abs(m)
    rpowers = np.array([])
    for j in range(0,len(n)):
        rpowers = np.append(rpowers, np.r_[m_abs[j]:(n[j]+2):2])
    rpowers = np.unique(rpowers)

    rpowern = np.zeros((length_r,len(rpowers)))
    for p in range(0,len(rpowers)):
        rpowern[:,p] = r**(rpowers[p])


    z = np.zeros((length_r, len(n)))
    for j in range(0,len(n)):
        s = np.r_[0:(n[j]+2-m_abs[j])//2]
        pows = np.r_[n[j]:m_abs[j]-2:-2]
        for k in np.flipud(np.r_[0:len(s)]):
            p = (1-2*np.mod(s[k],2))*\
            np.prod(np.r_[2:(n[j]-s[k]+1)])/\
            np.prod(np.r_[2:s[k]+1])/\
            np.prod(np.r_[2:(n[j]-m_abs[j])//2-s[k]+1])/\
            np.prod(np.r_[2:(n[j]+m_abs[j])//2-s[k]+1])
            idx = (pows[k] == rpowers)
            z[:,j] += p*rpowern[:,idx].reshape(length_r,)
        
        
    # Compute the Zernike functions
    # ------------------------------
    idx_pos = m>0
    idx_neg = m<0

    if any(idx_pos):
        z[:,idx_pos] *= np.cos(theta.reshape(len(theta),1).dot(m_abs[idx_pos].reshape(1,len(m_abs))))

    if any(idx_neg):
        z[:,idx_neg] *= np.sin(theta.reshape(len(theta),1).dot(m_abs[idx_neg].reshape(1,len(m_abs))))
        
        
    return z



