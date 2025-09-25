import torch


def poly_mul(p1, p2):
    degree1 = p1.size(0) - 1
    degree2 = p2.size(0) - 1
    
    result_degree = degree1 + degree2
    result = torch.zeros(result_degree + 1, dtype=p1.dtype)
    
    for i in range(p1.size(0)):
        for j in range(p2.size(0)):
            result[i + j] += p1[i] * p2[j]
    
    return result

# highest degree first (not numpy equvivalent: lowest degree first)
def poly_fromroots(roots):
    n = roots.shape[0]

    coeffs = torch.zeros((n,))
    for k in range(n):
        if k == 0:
            Q = torch.tensor([1, -roots[0]])
            coeffs = Q
        else:
            Q = torch.tensor([1, -roots[k]])
            coeffs = poly_mul(Q, coeffs)
    return coeffs

# expects coeffs highest degree first
def poly_val(coeffs, x):
    """Evaluate a polynomial at given x using Horner's method."""
    # x = torch.tensor(x, dtype=coeffs.dtype, device=coeffs.device)  # Ensure x is on the same device
    result = torch.zeros_like(x)
    
    for c in coeffs:
        result = result * x + c

    return result

def poly_der(coeffs):
    """Compute the derivative of a polynomial given its coefficients."""
    order = torch.arange(len(coeffs) - 1, 0, -1, dtype=coeffs.dtype, device=coeffs.device)
    return coeffs[:-1] * order  # Multiply by the respective exponent

'''
x : vector - dilated, translated datapoints of the effective support of the function
a : scalar - real part one of the poles of R
beta : scalar - imag part one of the poles of R
bmin: scalar - minimum absolute value of imaginary part of R's poles
'''
def Q(x, a, beta, bmin):
    def b_k(beta, bmin):
        b_k = beta**2 + bmin
        db_k = 2*beta # derivative of b_k
        return b_k, db_k
    b, db = b_k(beta, bmin)
    Qf = x**4 + x**2*(2*b**2 - 2*a**2) + a**4 + 2*a**2*b**2 + b**4
    dQx = 4*x**3 + 2*x*(2*b**2 - 2*a**2)
    dQa = -4*x**2*a + 4*a**3 + 4*a*b**2
    dQb = db*(4*x**2*b + 4*b**3 + 4*a**2*b)
    return Qf, dQx, dQa, dQb

'''
x : vector - dilated, translated datapoints of the effective support of the function
a : scalar - real part one of the poles of R
beta : scalar - imag part one of the poles of R
bmin: scalar - minimum absolute value of imaginary part of R's poles
'''
def R(x, a, beta, bmin):
    Qf, dQx, dQa, dQb = Q(x, a, beta, bmin)
    Rf = Qf**(-1)
    dRx = -Qf**(-2)*dQx
    dRa = -Qf**(-2)*dQa
    dRb = -Qf**(-2)*dQb
    return Rf, dRx, dRa, dRb

'''
x : dilated, translated datapoints of the effective support of the function
ak : real part of the poles of R
betak : imag part of the poles of R
pk : zeros of the polynom on the positive half-space
bmin: minimum absolute value of imaginary part of R's poles
sigma: parameter of the Gaussian function
'''
def psi_fun(x, ak, betak, pk, bmin, sigma,device):
    n = len(ak) # number of poles of the rational term R
    m = len(pk) # number of zeros of the polynomial term P

    Rfun = torch.ones(len(x),device=device)
    r_k = torch.zeros(n, len(x),device=device)
    dRx_k = torch.zeros(n, len(x),device=device)
    dRa_k = torch.zeros(n, len(x),device=device)
    dRb_k = torch.zeros(n, len(x),device=device)

    zeros = torch.cat((pk, -pk, torch.tensor([0.0])))

    Palg = poly_fromroots(zeros)
    Pf =  poly_val(Palg, x)
    dPx = poly_val(poly_der(Palg), x)

    # Construct the rational term R
    for k in range(n):
        r, rx, ra, rb = R(x, ak[k], betak[k], bmin)
        Rfun = Rfun*r # multiply the n number of elementary weight modifier polynomials 
        r_k[k,:] = r
        dRx_k[k,:] = rx
        dRa_k[k,:] = ra
        dRb_k[k,:] = rb

    # Construct R's derivative w.r.t. x
    dRx = torch.zeros(len(x),device=device)
    for k in range(n):
        rr = torch.cat((r_k[:k], r_k[k+1:])) # erase kth row because of chain rule of derivatives
        dRx += dRx_k[k, :] * torch.prod(rr, dim=0)

    # Construct the mother wavelet and derivatives
    Psi = Pf*Rfun*torch.exp(-x**2/sigma**2)

    # Derivatives w.r.t.x
    dPsix = dPx*Rfun*torch.exp(-x**2/sigma**2) + Pf*dRx*torch.exp(-x**2/sigma**2) - 2*x/sigma**2 * Psi

    # Derivatives w.r.t. a, b
    dPsia = torch.zeros(n, len(x),device=device)
    dPsib = torch.zeros(n, len(x),device=device)
    
    for k in range(n):
        rr = torch.cat((r_k[:k], r_k[k+1:])) # erase kth row because of partial derivatives
        dPsia[k,:] = dRa_k[k,:]*torch.prod(rr,dim=0)*Pf*torch.exp(-x**2/sigma**2)
        dPsib[k,:] = dRb_k[k,:]*torch.prod(rr,dim=0)*Pf*torch.exp(-x**2/sigma**2)

    # Derivatives w.r.t. p
    dPsip = torch.zeros(m, len(x),device=device)
    for k in range(m):
        dPp = -( Pf/( (x - pk[k])*(x + pk[k]) ) )*2*pk[k]
        roots = torch.cat((pk[:k], pk[k+1:]))  
        roots = torch.cat((roots, -roots, torch.tensor([0.0])))
        Pcurr = poly_fromroots(roots)
        Pf = poly_val(Pcurr, x)

        dPp = -Pf*2*pk[k]
        # dPp = dPp.clone().detach().to(device)
        dPsip[k, :] = dPp*Rfun*torch.exp(-x**2/sigma**2)


    # Derivatives w.r.t. sigma
    dPsiSigma = Psi*2*x**2*sigma**(-3)

    return Psi, dPsix, dPsia, dPsib, dPsip, dPsiSigma


'''
p : number of zeros in P
r : number of poles in R
n : number of wavelet coefficients
bmin : minimum absolute value of imaginary part of R's poles
alpha : (p1, ..., pp, r0real, r0imag, ..., rrreal, rrimag, s1, x1, s2, x2, ..., sn, xn, sigma)
'''
def adaRatGaussWav(n,t, params, p, r,bmin, smin=0.01, s_square= False,dtype=torch.float, device=None):
    alpha = params # TODO: ha params is cuda tensor training soran, akkor törölni
    # Some useful constants for indexing
    polebeg = p # p+1 in matlab, because start index is 1, not 0
    poleend = p+1+2*r-1 # ok, because np.array[0:3] eq array(1:3) in matlab
    wavebeg = p+1+2*r-1 # p+1+2*r in matlab, because start index is 1, not 0

    L = 2+2*r+p+1 # number of params of a dilated, translated wavelet

    N = len(t)

    # Initialize Phi, dPhi and Ind
    Phi = torch.zeros(N, n,device=device)
    dPhi = torch.zeros(N, n*L,device=device)
    Ind = torch.zeros(2, n*L,device=device)

    dPhit = torch.zeros(N, n,device=device)

    # common parameters for all dilated, translated wavelets
    sigma = alpha[-1]
    ak = alpha[polebeg:poleend-1:2] # only real part of poles
    betak = alpha[polebeg+1:poleend:2] # only imag part of poles
    pk = alpha[0:p]

    # Generate the wavelets and derivatives w.r.t. alpha
    for k in range(n):
        # Break up alpha to make the code readable
        begindzers = k*L # k*L+1 in matlab, because start index is 1, not 0
        endindzers = begindzers+p # p-1 in matlab
        
        begindpoles = k*L+polebeg
        endindpoles = begindpoles+2*r # 2*r-1 in matlab

        begindsig = k*L+poleend+2 # dil trans miatt

        # Current dilation and translation
        s = alpha[wavebeg+2*k]
        ss = s
        if s_square: ss = s**2 + smin
        x = alpha[wavebeg+2*k+1]
        tt = (t-x)/ss

        # Generate the next wavelet and associated derivatives
        [Psi, dPsix, dPsia, dPsib, dPsip, dPsiSigma] = psi_fun(tt, ak, betak, pk, bmin, sigma,device)

        # Transpose everything that needs to be transposed + save functions
        Psi = Psi/torch.sqrt(ss)
        Phi[:,k] = Psi.real
        dPsix = dPsix.real
        dPsia = torch.transpose(dPsia.real,0,1)/torch.sqrt(ss)
        dPsib = torch.transpose(dPsib.real,0,1)/torch.sqrt(ss)
        dPsip = torch.transpose(dPsip.real,0,1)/torch.sqrt(ss)
        dPsiSigma = dPsiSigma.real/torch.sqrt(ss)

        dPhit[:,k] = dPsix / torch.sqrt(ss)

        # Save derivatives and Ind values
        
        # zeros
        dPhi[:,begindzers:endindzers] = dPsip
        Ind[0,begindzers:endindzers] = k 
        Ind[1,begindzers:endindzers] = torch.arange(0, p)

        # poles
        dPhi[:, begindpoles:endindpoles-1:2] = dPsia
        Ind[0, begindpoles:endindpoles-1:2] = k 
        Ind[1, begindpoles:endindpoles-1:2] = torch.arange(polebeg, poleend-1, step=2)

        dPhi[:, begindpoles+1:endindpoles:2] = dPsib
        Ind[0, begindpoles+1:endindpoles:2] = k
        Ind[1, begindpoles+1:endindpoles:2] = torch.arange(polebeg+1, poleend, step=2)

        # Wavelet parameters
        dPsis = -0.5*ss**(-3/2)*Psi*torch.sqrt(ss) + 1/torch.sqrt(ss)* dPsix*(-1*ss**(-2))*(t-x).t() # .t() pythonban nem biztos, hogy kell
        if s_square: dPsis = dPsis * 2*s
        dPsit = dPsix*(-1)*ss**(-3/2)

        begindwav = k*L+wavebeg
        endindwav = begindwav + 2 # 1 in matlab
        dPsis = torch.unsqueeze(dPsis, 1)
        dPsit = torch.unsqueeze(dPsit, 1)
        dPhi[:, begindwav:endindwav] = torch.cat((dPsis, dPsit), dim=1)
        Ind[0, begindwav:endindwav] = k
        Ind[1, begindwav:endindwav] = torch.arange(wavebeg + 2*k, wavebeg + 2*k+2)

        # Sigma
        dPhi[:, begindsig] = dPsiSigma
        Ind[0, begindsig] = k
        Ind[1, begindsig] = len(alpha) - 1

        Ind = Ind.to(torch.int64)

    return Phi,dPhi,Ind, dPhit


def adaRatGaussWav2D(alphax, alphay, a, b, m, n, px, py, rx, ry, tetha, bmin,device=None):

    # Define base interval as the effective support of the non dilated and non translated wavelet
    x = torch.linspace(a, b, m,device=device)
    y = torch.linspace(a, b, m,device=device)

    N = x.shape[0]
    M = y.shape[0]

    X, Y = torch.meshgrid(x, y, indexing='ij')

    x_rot = X*torch.cos(tetha) - Y*torch.sin(tetha)
    y_rot = X*torch.sin(tetha) + Y*torch.cos(tetha)

    x = torch.flatten(x_rot)
    y = torch.flatten(y_rot)

    Phix,dPhix,Indx, dPhit = adaRatGaussWav(n, x, alphax, px, rx,bmin)
    Phiy,dPhiy,Indy, dPhit = adaRatGaussWav(n, y, alphay, py, ry,bmin)

    Phi = torch.zeros(N,M,n)

    for k in range(n):
        PX = Phix[:,k].reshape(N,M)
        PY = Phiy[:,k].reshape(N,M)
        Phi[:,:,k] = PX*PY
    
    Lx = 2+2*rx+px+1 # number of params of a dilated, translated 1D wavelet (without rotation param)
    Ly = 2+2*ry+py+1

    L = Lx + Ly

    dPhi = torch.zeros(N,M,n*L + n)

    for k in range(n*L):
        if k < n*Lx: # dPhix.shape[1] = n*Lx
            dPX = dPhix[:,k].reshape(N,M)
            PY = Phiy[:,Indy[0,k]].reshape(M,N)
            dPhi[:,:,k] = dPX * PY
        else: 
            dPY = dPhiy[:,k - n*Lx].reshape(N,M)
            PX = Phix[:,Indx[0,k - n*Lx]].reshape(M,N)
            dPhi[:,:,k] = dPY * PX

    Indy[1,:] = Indy[1,:] + Lx
    Ind = torch.cat((Indx,Indy),1)

    # Derivatives w.r.t. orientation
    # TODO

    return Phi, dPhi, Ind
