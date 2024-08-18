###Fit Green's function for Poisson and Helmholtz equation
import numpy as np
from scipy import integrate
from scipy import linalg
from scipy.constants import pi
import matplotlib.pyplot as plt
from tqdm import tqdm


###Poisson and Helmholtz equation

def gaussian(x,x_0,sigma):
    """gaussian"""
    return 1/(np.sqrt(2*pi*sigma**2))*np.exp(-(x-x_0)**2/(2*sigma**2))


def poisson_solver(x_grid,bv,f,f_args):
    """solves 1d poisson boundary value problem
    x_grid: grid points
    y_init: initial guess
    bv: array of boundary values bv[0] = bv(a), bv[1] = bv(b)
    f: source term f(x,f_args)
    f_args: additional function arguments
    """

    #Equation
    def poisson(x,y):
        """poisson equation"""
        return [y[1],f(x,*f_args)]
    
    def bc(ya,yb):
        """evaluate boundary conditions [ya,yb]-[0,0] = [0,0]"""
        return [ya[0]-bv[0],yb[0]-bv[1]]
    
    y_init = np.zeros((2,x_grid.size))
    
    return integrate.solve_bvp(poisson,bc,x_grid,y_init)


def helmholtz_solver(x_grid,bv,k,f,f_args):
    """solves 1d poisson boundary value problem
    x_grid: grid points
    y_init: initial guess
    bv: array of boundary values bv[0] = bv(a), bv[1] = bv(b)
    k: parameter helmholtz equation
    f: source term f(x,f_args)
    f_args: additional function arguments
    """

    #Equation
    def helmholtz(x,y):
        """Helmholtz eq."""

        return [y[1], -k**2*y[0]+f(x,*f_args)]
    
    def bc(ya,yb):
        """evaluate boundary conditions [ya,yb]-[0,0] = [0,0]"""
        return [ya[0]-bv[0],yb[0]-bv[1]]
    
    y_init = np.zeros((2,x_grid.size))
    
    return integrate.solve_bvp(helmholtz,bc,x_grid,y_init)



###Green fit basis functions
def generate_phi(D):
    """generate list of basis functions until order D"""
    phi = []

    for i in range(D):
        pow_x = i

        for pow_xprime in range(i+1):
            phi.append(lambda x,x_prime,pow_x=pow_x,pow_xprime=pow_xprime: (x**pow_x)*(x_prime**pow_xprime))    #store basis functions in list
            pow_x -= 1
    
    return phi

def psi(phi_j,x_k,x_prime,x_0,sigma):
    """Fit function for Greens function
    phi_j: basisfunction with index j
    x_k: Data point where psi is evaluated
    x_prime: range to integrate
    x_0: center of inhomogenity
    sigma: std of inhomogenity
    """
    y = phi_j(x_k,x_prime)*gaussian(x_prime,x_0,sigma)  #integrand
   
    if x_prime.size == 0:
        psi_j = 0   #Integral is 0 for integration from 0 to 0 or L to L
    
    else:
        psi_j = integrate.simpson(y,x_prime)
   
    return psi_j



###Fit
def green_fit(phi,x_data,u_data,n_int,L,x_0,sigma):
    """Fit Greens function by solving Ag = b
    phi: list of basis funcitons
    x_data: array of x coordinates of data points
    u_data: array of y coordinates for N data points for M source terms: (M,N) array
    n_int: number of points for integration of psi
    L: boundary
    x_0: array of centers of source term
    sigma: array of std of gaussian"""

    D = len(phi)    #Number of basis functions
    N = x_data.size #Number of data points
    M = x_0.size    #Number of different source terms

    #calculate psi
    psi_fit = np.zeros((2*D,M,N)) #every psi with different source terms for every x_k
    x_int = np.linspace(0,L,n_int)  #array for integration of psi

    for i in tqdm(range(2*D)):
        for m in range(M):
            for k in range(N):
                if i <= (D-1):
                    x_int_psi = x_int[x_int<x_data[k]]
                    psi_fit[i,m,k] = psi(phi[i],x_data[k],x_int_psi,x_0[m],sigma[m])
                else:
                    x_int_psi = x_int[x_int>=x_data[k]]
                    psi_fit[i,m,k] = psi(phi[i-D],x_data[k],x_int_psi,x_0[m],sigma[m])

    #determine matrix A and vector b
    A = np.zeros((2*D,2*D))
    b = np.zeros(2*D)

    for i in range(2*D):
        b[i] = np.sum(psi_fit[i,:,:]*u_data)

        for j in range(2*D):
            A[i,j] = np.sum(psi_fit[i,:,:]*psi_fit[j,:,:]) 
    
    return linalg.solve(A,b)



###Greens function
def greens_func(g,phi,x,x_prime):
    """Evaluates Greens function at point (x,x_strich)
    g: fit parameter
    phi: list of basisfunctions
    x: data point
    x_prime
    """

    D = len(phi)
    G = np.zeros_like(x_prime)

    i_prime_smaller = np.greater(x,x_prime)
    i_prime_greater = np.greater_equal(x_prime,x)

    
    for j in range(D):
        G[i_prime_smaller] += g[j]*phi[j](x[i_prime_smaller],x_prime[i_prime_smaller]) #Greens function for 0<=x'<x


    for j in range(D,2*D):
        G[i_prime_greater] += g[j]*phi[j-D](x[i_prime_greater],x_prime[i_prime_greater])    #Greens function for x<=x'<=L
    
    return G



###Analytic Greens functions
def green_poisson(x,x_prime,L):
    """calculates analytic Greens function for Poisson equation"""

    return x*(x_prime/L-1)*(x<=x_prime)+x_prime*(x/L-1)*(x>x_prime)

def green_helmholtz(x,x_prime,L,k):
    """calculates analytic Greens function for Helmholtz equation"""

    g1 = np.sin(k*x)*np.sin(k*(x_prime-L))/(k*np.sin(k*L))
    g2 = np.sin(k*x_prime)*np.sin(k*(x-L))/(k*np.sin(k*L))

    return g1*(x<=x_prime)+g2*(x>x_prime)



if __name__ == '__main__':

    ###Generate data
    N = 200 #nuber of datapoints
    M = 20  #number of different source terms
    L = 3   #boundary

    x_0 = np.linspace(0,L,M)    #center of source terms
    sigma = np.ones(M)*0.05     #std of source terms
    x_data = np.linspace(0,L,N)
    x_calc = np.linspace(0,L,5000)   #grid for numeric solving

    bv_fit = [0,0]  #boundary values
    k = 3   #parameter helmholtz

    u_poisson_data = np.zeros((M,N))
    u_helmholtz_data = np.zeros((M,N))

    print('Solve numerically')

    for m in tqdm(range(M)):
        gauss_args_fit = [x_0[m],sigma[m]]
        u_poisson_data[m,:] = poisson_solver(x_calc,bv_fit,gaussian,gauss_args_fit).sol(x_data)[0]
        u_helmholtz_data[m,:] = helmholtz_solver(x_calc,bv_fit,k,gaussian,gauss_args_fit).sol(x_data)[0]


    ###Fit Greens function
    D_poisson = 3  #max order of polynom +1
    D_helmholtz = 10
    n_int = 3000

    #Basis functions
    phi_poisson = generate_phi(D_poisson)
    phi_helmholtz = generate_phi(D_helmholtz)
   
    #Fit
    print('Poisson_fit')
    g_poisson = green_fit(phi_poisson,x_data,u_poisson_data,n_int,L,x_0,sigma)

    print('Helmholtz_fit')
    g_helmholtz = green_fit(phi_helmholtz,x_data,u_helmholtz_data,n_int,L,x_0,sigma)


    ###Validate fitted Greens function
    n_val = 400
    x_val = np.linspace(0,L,n_val)
    x_prime_val = np.linspace(0,L,n_val)

    X_val,X_prime_val = np.meshgrid(x_val,x_prime_val)  #grid of x and x_prime

    #Fitted Greens function
    G_poisson_fit = greens_func(g_poisson,phi_poisson,X_val,X_prime_val)
    G_helmholtz_fit = greens_func(g_helmholtz,phi_helmholtz,X_val,X_prime_val)

    #Analytc Greens function
    G_poisson_val = green_poisson(X_val,X_prime_val,L)
    G_helmholtz_val = green_helmholtz(X_val,X_prime_val,L,k)

    #Plot fitted Greens function for x and x_strich
    fig1,axs = plt.subplots(figsize=(7,5),layout='constrained')
    plt1 = axs.pcolormesh(X_val,X_prime_val,G_poisson_fit)
    axs.set_xlabel('x')
    axs.set_ylabel('x_prime')
    axs.set_title('Greens function of Poisson equation')
    fig1.colorbar(plt1)

    fig2,axs = plt.subplots(figsize=(7,5),layout='constrained')
    plt2 = axs.pcolormesh(X_val,X_prime_val,G_helmholtz_fit)
    axs.set_xlabel('x')
    axs.set_ylabel('x_prime')
    axs.set_title('Greens function of Helmholtz equation')
    fig2.colorbar(plt2)

    #Plot error of Greens functions
    err_green_poisson = np.abs(G_poisson_fit-G_poisson_val)
    err_green_helmholtz = np.abs(G_helmholtz_fit-G_helmholtz_val)
    
    fig3,axs = plt.subplots(figsize=(7,5),layout='constrained')
    plt3 = axs.pcolormesh(X_val,X_prime_val,err_green_poisson)
    axs.set_xlabel('x')
    axs.set_ylabel('x_prime')
    axs.set_title('Error of poisson Greens function')
    fig3.colorbar(plt3)

    fig4,axs = plt.subplots(figsize=(7,5),layout='constrained')
    plt4 = axs.pcolormesh(X_val,X_prime_val,err_green_helmholtz)
    axs.set_xlabel('x')
    axs.set_ylabel('x_prime')
    axs.set_title('Error of helmholtz Greens function')
    fig4.colorbar(plt4)

    #Plot true and fitted Greens function for some fixed x_prime
    i_x_strich_fix = [50,150,300]

    fig,axs = plt.subplots(figsize=(8,8),layout='constrained',ncols=2,nrows=3)
    fig.suptitle('Greens function for fixed x_prime')
    for i,i_fix in enumerate(i_x_strich_fix):
        
        axs[i,0].plot(x_val,G_poisson_fit[i_fix,:],'r')
        axs[i,0].plot(x_val,G_poisson_val[i_fix,:],'k:')
        axs[i,0].set_xlabel('x')
        axs[i,0].set_ylabel(f'G(x,{x_prime_val[i_fix]})')

        axs[i,1].plot(x_val,G_helmholtz_fit[i_fix,:],'r')
        axs[i,1].plot(x_val,G_helmholtz_val[i_fix,:],'k:')
        axs[i,1].set_xlabel('x')
        axs[i,1].set_ylabel(f'G(x,{x_prime_val[i_fix]})')
    
    plt.show()






