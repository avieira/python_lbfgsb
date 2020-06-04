import numpy as np
from scipy.optimize import minpack2
from collections import deque

def compute_Cauchy_point(x, g, l, u, W, M, theta):
    """
        Computes the generalized Cauchy point (GCP), defined as the first 
        local minimizer of the quadratic
        
        .. math::
        :nowrap:
                  \[\langle g,s\rangle + \frac{1}{2} \langle s, (\theta I + WMW^\intercal)s\rangle\]

       along the projected gradient direction 
       .. math:: $P_[l,u](x-\theta g).$    


    :param x: starting point for the GCP computation 
    :type x: np.array
    
    :param g: gradient of f(x). g must be a nonzero vector 
    :type g: np.array
    
    :param l: the lower bound of x 
    :type l: np.array
    
    :param u: the upper bound of x 
    :type u: np.array
    
    :param W: part of limited memory BFGS Hessian approximation 
    :type W: np.array   
    
    :param M: part of limited memory BFGS Hessian approximation 
    :type M: np.array   
    
    :param theta: part of limited memory BFGS Hessian approximation
    :type theta: float 
    
    :return: dict containing a computed value of:
            - 'xc' the GCP
            - 'c' = W^(T)(xc-x), used for the subspace minimization
            - 'F' set of free variables
    :rtype: dict
    ..todo check F

    .. seealso:: 

       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
       memory algorithm for bound constrained optimization'',
       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
       Subroutines for Large Scale Bound Constrained Optimization''
       Tech. Report, NAM-11, EECS Department, Northwestern University,
       1994.
    """
    eps_f_sec = 1e-30 
    t = np.empty(x.size)
    d = np.empty(x.size)
    x_cp = x.copy()
    for i in range(x.size):
        if g[i]<0:
            t[i] = (x[i]-u[i])/g[i]
        elif g[i]>0:
            t[i] = (x[i]-l[i])/g[i]
        else:
            t[i]=np.inf
        if t[i]==0:
            d[i]=0
        else:
            d[i]=-g[i]

    F = np.argsort(t)
    F = [i for i in F if t[i] >0]
    t_old = 0
    F_i = 0
    b=F[0]
    t_min = t[b]
    Dt = t_min
    
    p = np.transpose(W).dot(d)
    c = np.zeros(p.size)
    f_prime = -d.dot(d)
    f_second = -theta*f_prime-p.dot(M.dot(p))
    f_sec0 = f_second
    Dt_min = -f_prime/f_second

    while Dt_min>=Dt and F_i<len(F):
        if d[b]>0:
            x_cp[b] = u[b]
        elif d[b]<0:
            x_cp[b] = l[b]
        x_bcp = x_cp[b]
        
        zb = x_bcp - x[b]
        c += Dt*p
        W_b = W[b,:]
        g_b = g[b]
        
        f_prime += Dt*f_second+ g_b*(g_b+theta*zb-W_b.dot(M.dot(c)))
        f_second -= g_b*(g_b*theta+W_b.dot(M.dot(2*p+g_b*W_b)))
        f_second = min(f_second, eps_f_sec*f_sec0)
        
        Dt_min = -f_prime/f_second
        
        p += g_b*W_b
        d[b] = 0
        t_old = t_min
        F_i+=1
        
        if F_i<len(F):
            b=F[F_i]
            t_min = t[b]
            Dt = t_min-t_old
        else:
            t_min = np.inf

    Dt_min = 0 if Dt_min<0 else Dt_min
    t_old += Dt_min
    
    for i in range(x.size):
        if t[i]>=t_min:
            x_cp[i] = x[i] + t_old*d[i]
    
    F = [i for i in F if t[i]!=t_min]
            
    c += Dt_min*p
    return {'xc':x_cp, 'c':c, 'F':F}

def minimize_model(x, xc, c, g, l, u, W, M, theta):
    """
        Computes an approximate solution of the subspace problem
        .. math::
        :nowrap:

       \[\begin{aligned}
            \min& &\langle r, (x-xcp)\rangle + 1/2 \langle x-xcp, B (x-xcp)\rangle\\
            \text{s.t.}& &l<=x<=u\\
                       & & x_i=xcp_i \text{for all} i \in A(xcp)
        \]
                     
       along the subspace unconstrained Newton direction 
       .. math:: $d = -(Z'BZ)^(-1) r.$    


    :param x: starting point for the GCP computation 
    :type x: np.array
    
    :param xc: Cauchy point
    :type xc: np.array
    
    :param c: W^T(xc-x), computed with the Cauchy point
    :type c: np.array
    
    :param g: gradient of f(x). g must be a nonzero vector 
    :type g: np.array
    
    :param l: the lower bound of x 
    :type l: np.array
    
    :param u: the upper bound of x u
    :type u: np.array
    
    :param W: part of limited memory BFGS Hessian approximation 
    :type W: np.array   
    
    :param M: part of limited memory BFGS Hessian approximation 
    :type M: np.array   
    
    :param theta: part of limited memory BFGS Hessian approximation 
    :type theta: float 
    
    :return: dict containing a computed value of:
            - 'xbar' the minimizer
    :rtype: dict
    
    ..todo Normaly, free_vars is already defined in compute_Cauchy_point in F.


    .. seealso:: 

       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
       memory algorithm for bound constrained optimization'',
       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
       Subroutines for Large Scale Bound Constrained Optimization''
       Tech. Report, NAM-11, EECS Department, Northwestern University,
       1994.
    """
    invThet = 1.0/theta
    
    Z = list()
    free_vars = list()
    n = xc.size
    unit = np.zeros(n)
    for i in range(n):
        unit[i] = 1
        if ((xc[i] != u[i]) and (xc[i] != l[i])):
            free_vars.append(i)
            Z.append(unit.copy())
        unit[i]=0
        
    if len(free_vars) == 0:
        return {'xbar':xc}
    
    Z = np.asarray(Z).T
    WTZ = W.T.dot(Z)
    
    rHat = [(g + theta*(xc-x) - W.dot(M.dot(c)))[ind] for ind in free_vars]
    v = WTZ.dot(rHat)
    v = M.dot(v)
    
    N = invThet*WTZ.dot(np.transpose(WTZ))
    N = np.eye(N.shape[0])-M.dot(N)
    v = np.linalg.solve(N, v)
    
    dHat = -invThet * (rHat + invThet * np.transpose(WTZ).dot(v))
    
    #Find alpha
    alpha_star = 1
    for i in range(len(free_vars)):
        idx = free_vars[i]
        if dHat[i] > 0:
            alpha_star = min(alpha_star, (u[idx]-xc[idx])/dHat[i])
        elif dHat[i] < 0:
            alpha_star = min(alpha_star, (l[idx]-xc[idx])/dHat[i])
    
    d_star = alpha_star*dHat;
    xbar = xc;
    for i in range(len(free_vars)):
        idx = free_vars[i];
        xbar[idx] += d_star[i];

    return {'xbar':xbar}


def max_allowed_steplength(x, d, l, u, max_steplength):
    """
        Computes the biggest 0<=k<=max_steplength such that:
                l<= x+kd <= u

    :param x: starting point 
    :type x: np.array
    
    :param d: direction
    :type d: np.array
        
    :param l: the lower bound of x 
    :type l: np.array
    
    :param u: the upper bound of x 
    :type u: np.array
    
    :param max_steplength: maximum steplength allowed 
    :type max_steplength: float 
    
    :return: maximum steplength allowed 
    :rtype: float
    

    .. seealso:: 

       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
       memory algorithm for bound constrained optimization'',
       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
       Subroutines for Large Scale Bound Constrained Optimization''
       Tech. Report, NAM-11, EECS Department, Northwestern University,
       1994.
    """
    max_stpl = max_steplength
    for i in range(x.size):
        if d[i] > 0:
            max_stpl = min(max_stpl, (u[i] - x[i]) / d[i])
        elif d[i] < 0 :
            max_stpl = min(max_stpl, (l[i] - x[i]) / d[i])
    
    return max_stpl

def line_search(x0, f0, g0, d, above_iter, max_steplength,\
                fct_f, fct_grad,\
                alpha = 1e-4, beta = 0.9,\
                xtol_minpack = 1e-5, max_iter = 30):
    """
        Finds a step that satisfies a sufficient decrease condition and a curvature condition.

        The algorithm is designed to find a step that satisfies the sufficient decrease condition 
        
              f(x0+stp*d) <= f(x0) + alpha*stp*\langle f'(x0),d\rangle,
        
        and the curvature condition
        
              abs(f'(x0+stp*d)) <= beta*abs(\langle f'(x0),d\rangle).
        
        If alpha is less than beta and if, for example, the functionis bounded below, then
        there is always a step which satisfies both conditions. 

    :param x0: starting point 
    :type x0: np.array
    
    :param f0: f(x0) 
    :type f0: float
    
    :param g0: f'(x0), gradient 
    :type g0: np.array
    
    :param d: search direction
    :type d: np.array
    
    :param above_iter: current iteration in optimization process
    :type above_iter: integer
    
    :param max_steplength: maximum steplength allowed 
    :type max_steplength: float 
    
    :param fct_f: callable, function f(x) 
    :type fct_f: function returning float
    
    :param fct_grad: callable, function f'(x) 
    :type fct_grad: function returning np.array
    
    :param alpha, beta: parameters of the decrease and curvature conditions 
    :type alpha, beta: floats
    
    :param xtol_minpack: tolerance used in minpack2.dcsrch
    :type xtol_minpack: float
    
    :param max_iter: number of iteration allowed for finding a steplength
    :type max_iter: integer
    
    :return: optimal steplength meeting both decrease and curvature condition 
    :rtype: float
    
    
    

    .. seealso:: 
        
       [minpack] scipy.optimize.minpack2.dcsrch

       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
       memory algorithm for bound constrained optimization'',
       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
       Subroutines for Large Scale Bound Constrained Optimization''
       Tech. Report, NAM-11, EECS Department, Northwestern University,
       1994.
    """
    
    steplength_0 = 1 if max_steplength > 1 else 0.5*max_steplength
    f_m1 = f0
    dphi = g0.dot(d)
    dphi_m1 = dphi
    i = 0

    if(above_iter == 0):
        max_steplength = 1.0
        steplength_0 = min(1.0/np.sqrt(d.dot(d)), 1.0)

    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b'START'
    
    while i<max_iter:
        steplength, f0, dphi, task = minpack2.dcsrch(steplength_0, f_m1, dphi_m1,
                                                   alpha, beta, xtol_minpack, task,
                                                   0, max_steplength, isave, dsave)
        if task[:2] == b'FG':
            steplength_0 = steplength
            f_m1 = fct_f(x0 + steplength*d)
            dphi_m1 = fct_grad(x0 + steplength*d).dot(d)
        else:
            break
    else:
        # max_iter reached, the line search did not converge
        steplength = None
    
    
    if task[:5] == b'ERROR' or task[:4] == b'WARN':
        if task[:21] != b'WARNING: STP = STPMAX':
            print(task)
            steplength = None  # failed
        
    return steplength


def update_SY(sk, yk, S, Y, m,\
              W, M, thet,\
              eps = 2.2e-16):
    """
        Update lists S and Y, and form the L-BFGS Hessian approximation thet, W and M.

    :param sk: correction in x = new_x - old_x 
    :type sk: np.array
    
    :param yk: correction in gradient = f'(new_x) - f'(old_x) 
    :type yk: np.array
    
    :param S, Y: lists defining the L-BFGS matrices, updated during process (IN/OUT)
    :type S, Y: list
    
    :param m: Maximum size of lists S and Y: keep in memory only m previous iterations
    :type m: integer
    
    :param W, M: L-BFGS matrices 
    :type W, M: np.array
    
    :param thet: L-BFGS float parameter 
    :type thet: float
    
    :param eps: Positive stability parameter for accepting current step for updating matrices.
    :type eps: float >0
    
    :return: updated [W, M, thet]
    :rtype: tuple 

    .. seealso:: 

       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
       memory algorithm for bound constrained optimization'',
       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
       Subroutines for Large Scale Bound Constrained Optimization''
       Tech. Report, NAM-11, EECS Department, Northwestern University,
       1994.
    """
    sTy = sk.dot(yk)
    yTy = yk.dot(yk)
    if (sTy > eps*yTy):
        S.append(sk)
        Y.append(yk)
        if len(S) > m :
            S.popleft()
            Y.popleft()
        Sarray = np.asarray(S).T
        Yarray = np.asarray(Y).T
        STS = np.transpose(Sarray).dot(Sarray)
        L = np.transpose(Sarray).dot(Yarray)
        D = np.diag(-np.diag(L))
        L = np.tril(L, -1)
        
        thet = yTy/sTy
        W = np.hstack([Yarray, thet*Sarray])
        M = np.linalg.inv(np.hstack([np.vstack([D, L]), np.vstack([L.T, thet*STS])]))

    return [W, M, thet]

def L_BFGS_B(x0, f, df, l, u, m=10,\
             epsg = 1e-5, epsf = 1e7, max_iter = 50,\
             alpha_linesearch = 1e-4, beta_linesearch = 0.9,\
             max_steplength = 1e8,\
             xtol_minpack = 1e-5, max_iter_linesearch = 30, eps_SY = 2.2e-16):
    """
       Solves bound constrained optimization problems by using the compact formula 
       of the limited memory BFGS updates. 

    :param x0: initial guess
    :type sk: np.array
    
    :param f: cost function to optimize f(x)
    :type f: function returning float
    
    :param df: gradient of cost function to optimize f'(x)
    :type df: function returning np.array
    
    :param l: the lower bound of x 
    :type l: np.array
    
    :param u: the upper bound of x 
    :type u: np.array
    
    :param m: Maximum size of lists for L-BFGS Hessian approximation 
    :type m: integer
    
    :param epsg: Tolerance on projected gradient: programs converges when
                P(x-g, l, u)<epsg.
    :type epsg: float
    
    :param epsf: Tolerance on function change: programs ends when (f_k-f_{k+1})/max(|f_k|,|f_{k+1}|,1) < epsf * epsmch, where 
                epsmch is the machine precision. 
    :type epsf: float
    
    :param alpha_linesearch, beta_linesearch: Parameters for linesearch. 
                                              See ``alpha`` and ``beta`` in :func:`line_search`
    :type alpha_linesearch, beta_linesearch: float 

    :param max_steplength: Maximum steplength allowed. See ``max_steplength`` in :func:`max_allowed_steplength`
    :type max_steplength: float
    
    :param xtol_minpack: Tolerence used by minpack2. See ``xtol_minpack`` in :func:`line_search`
    :type xtol_minpack: float
    
    :param max_iter_linesearch: Maximum number of trials for linesearch. 
                                See ``max_iter_linesearch`` in :func:`line_search`
    :type max_iter_linesearch: integer 
    
    :param eps_SY: Parameter used for updating the L-BFGS matrices. See ``eps`` in :func:`update_SY`
    :type eps_SY: float
    
    :return: dict containing:
            - 'x': optimal point
            - 'f': optimal value at x
            - 'df': gradient f'(x)
    :rtype: dict 


    ..todo Check matrices update and different safeguards may be missing
    
    .. seealso:: 
       Function tested on Rosenbrock and Beale function with different starting points. All tests passed. 
        
       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
       memory algorithm for bound constrained optimization'',
       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
       Subroutines for Large Scale Bound Constrained Optimization''
       Tech. Report, NAM-11, EECS Department, Northwestern University,
       1994.
    """
    n = x0.size
    if x0.dtype != np.float64:
        x = x0.astype(np.float64, copy = True)
        x = np.clip(x, l, u)
    else:
        x = np.clip(x0, l, u)
    k=0
    S = deque()
    Y = deque()
    W = np.zeros([n, 1])
    M = np.zeros([1, 1])
    theta = 1
    epsmch = np.finfo(1.0).resolution
    
    f0 = f(x)
    g = df(x)
    i=0
    while np.max(np.abs(np.clip(x-g,l,u)-x))>epsg and i<max_iter:
        oldf0 = f0
        oldx = x.copy()
        oldg = g.copy()
        dictCP = compute_Cauchy_point(x, g, l, u, W, M, theta)
        dictMinMod = minimize_model(x, dictCP['xc'], dictCP['c'], g, l, u, W, M, theta)
        
        d = dictMinMod['xbar'] - x
        max_stpl = max_allowed_steplength(x, d, l, u, max_steplength)
        steplength = line_search(x, f0, g, d, i, max_stpl, f, df,\
                alpha_linesearch, beta_linesearch,\
                xtol_minpack, max_iter_linesearch)
        
        if steplength==None:
            if len(S)==0:
                #Hessian already rebooted: abort.
                print("Error: can not compute new steplength : abort")
                return {'x':x, 'f':f(x), 'df':df(x)}
            else:
                #Reboot BFGS-Hessian:
                S.clear()
                Y.clear()
                W = np.zeros([n, 1])
                M = np.zeros([1, 1])
                theta = 1
        else:
            x += steplength*d
            f0 = f(x)
            g = df(x)
            [W, M, theta] = update_SY(x-oldx, g-oldg, S, Y, m,\
                           W, M, theta, eps_SY)
        
        
            print("Iteration #%d (max: %d): ||x||=%.3e, f(x)=%.3e, ||df(x)||=%.3e, cdt_arret=%.3e (eps=%.3e)"%\
                  (i, max_iter, np.linalg.norm(x, np.inf), f0, np.linalg.norm(g, np.inf),\
                   np.max(np.abs(np.clip(x-g,l,u)-x)), epsg))
            if((oldf0-f0)/max(abs(oldf0),abs(f0),1)<epsmch * epsf):
                print("Relative reduction of f below tolerence: abort.")
                break
            i += 1
        
    if i==max_iter:
        print("Maximum iteration reached.")
        
    return {'x':x, 'f':f0, 'df':g}