import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

dim=4
mu=0.01

#function of the system
def r1(x,y):
    return(np.sqrt((x-mu)**2+y**2))

def r2(x,y):
    return(np.sqrt((x-mu+1)**2+y**2))

def omega_x(x,y):
    return(x-(1-mu)*(x-mu)/(r1(x,y)**3)-mu*(x-mu+1)/(r2(x,y)**3))

def omega_y(x,y):
    return(y-(1-mu)*y/(r1(x,y)**3)-y*mu/(r2(x,y)**3))

def omega_yy(x,y):
    return(1-(1-mu)/(r1(x,y)**3)-mu/(r2(x,y)**3)+y**2*(3*(1-mu)/(r1(x,y)**5)+3*mu/(r2(x,y)**5)))

def omega_xx(x,y):
    return(1-(1-mu)/(r1(x,y)**3)-mu/(r2(x,y)**3)+3*(1-mu)*(x-mu)**2/(r1(x,y)**5)+3*mu*(x-mu+1)**2/(r2(x,y)**5))

def omega_xy(x,y):
    return(y*(3*(1-mu)*(x-mu)/(r1(x,y)**5)+3*mu*(x-mu+1)/(r2(x,y)**5)))

def DG(x):
    DG=np.zeros((4,4))
    DG[2,0]=omega_xx(x[0],x[1])
    DG[2,1]=omega_xy(x[0],x[1])
    DG[3,0]=omega_xy(x[0],x[1])
    DG[3,1]=omega_yy(x[0],x[1])
    DG[1,3]=1
    DG[0,2]=1
    DG[2,3]=2
    DG[3,2]=-2

    return(DG)

def f(x):
    mat_x=np.zeros((4,4))
    c=0
    l1=[]
    l2=[]
    l3=[]
    l4=[]
    for i in range(4):
        l1.append(x[i+4])
        l2.append(x[i+8])
        l3.append(x[i+12])
        l4.append(x[i+16])

    mat_x=np.array([l1,l2,l3,l4])
    mat=np.dot(DG(x),mat_x)
    l=[]
    for i in range(4):
        for j in range(4):
            l.append(mat[i,j])

    var_eq=np.array(l)
    eq=np.array([x[2],x[3],2*x[3]+omega_x(x[0],x[1]),-2*x[2]+omega_y(x[0],x[1])])

    return(np.concatenate((eq,var_eq)))

def g(x):
    return(np.array([x[2],x[3],2*x[3]+omega_x(x[0],x[1]),-2*x[2]+omega_y(x[0],x[1])]))

#define the system of ode
def system_ode_var(t,x,dim,f):
    y=np.zeros(dim) #contain all the values of x_1,x_2...etc
    derivative=np.zeros(dim)

    for i in range (dim): #set all the initial conditions
        y[i]=x[i]

    for i in range(dim):
        derivative[i]=f(x)[i]

    return(derivative)

def system_ode(t,x,dim,g):
    y=np.zeros(dim) #contain all the values of x_1,x_2...etc
    derivative=np.zeros(dim)

    for i in range (dim): #set all the initial conditions
        y[i]=x[i]

    for i in range(dim):
        derivative[i]=g(x)[i]

    return(derivative)

#return the solution with a max_step beetween each time
def solve_system(max_step,args,method,t_max,t_0,x_0):

    #t_eval = np.linspace(t_0, t_max, num_points) si on veut un nombre de point donné, rentrer t_eval à la place de max_step

    solution=solve_ivp(system_ode,[t_0,t_max],x_0,method=method,args=args,max_step=max_step)

    #print(f"x(0)={solution.y[0,0]},y(0)={solution.y[1,0]}")
    #print(f"x(2pi)={solution.y[0,-1]}y(2pi)={solution.y[1,-1]}")

    return(solution)

def solve_system_var(max_step,args,method,t_max,t_0,x_0):

    #t_eval = np.linspace(t_0, t_max, num_points) si on veut un nombre de point donné, rentrer t_eval à la place de max_step

    solution=solve_ivp(system_ode_var,[t_0,t_max],x_0,method=method,args=args,max_step=max_step)

    #print(f"x(0)={solution.y[0,0]},y(0)={solution.y[1,0]}")
    #print(f"x(2pi)={solution.y[0,-1]}y(2pi)={solution.y[1,-1]}")

    return(solution)

#return the solution with np point
def solve_system_NP(NP,args,method,t_max,t_0,x_0):
    time=np.linspace(t_0,t_max,NP)

    solution=solve_ivp(system_ode,[t_0,t_max],x_0,method=method,args=args,t_eval=time,max_step=0.01)

    return(solution)

def solve_system_NP_var(NP,args,method,t_max,t_0,x_0):
    time=np.linspace(t_0,t_max,NP)

    solution=solve_ivp(system_ode_var,[t_0,t_max],x_0,method=method,args=args,t_eval=time,max_step=0.01)

    return(solution)

def omega(x,y):
    return(1/2*(x**2+y**2)+(1-mu)/r1(x,y)+mu/r2(x,y)+(1/2)*mu*(1-mu))

def jacobi_first_int(x):
    return(2*omega(x[0],x[1])-(x[2]**2+x[3]**2))

##Computation of L1

def f2(x):
    return((mu*(1+x)**2/(3-2*mu+x*(3-mu+x)))**(1/3))

def L2(mu):
    xi_0=(mu/3*(1-mu))**(1/3)

    while np.abs(f2(xi_0)-xi_0)>1e-14:
        xi_0=f2(xi_0)
    xi_2=mu-1-xi_0

    return(np.array([xi_2,0,0,0]))
##computation of L2
def f1(x):
    return((mu*(1-x)**2/(3-2*mu-x*(3-mu-x)))**(1/3))

def L1(mu):
    xi_0=(mu/(3*(1-mu)))**(1/3)
    xi_1=f1(xi_0)

    while np.abs(xi_1-xi_0)>1e-14:
        xi_0=xi_1
        xi_1=f1(xi_1)

    xi_1=mu-1+xi_0
    return(np.array([xi_1,0,0,0]))
##computation of L3
def f3(x):
    return(((1-mu)*(1+x)**2/(1+2*mu+x*(2+mu+x)))**(1/3))

def L3(mu):
    xi_0=1-7/12*mu
    xi_3=f3(xi_0)

    while np.abs(xi_3-xi_0)>1e-14:
        xi_0=xi_3
        xi_3=f3(xi_3)

    xi_3=mu+xi_0
    return(np.array([xi_3,0,0,0]))
###Poincaré Map
def h(x):
    return(x[0]-0.2)

def grad_h(x):
    grad=np.zeros(4)
    grad[0]=1
    return(grad)


def delta(t,sol):
    prod=np.dot(grad_h(sol),g(sol))
    return(-h(sol)/prod)

def poincare_map(x0,t0,dir,eps=1e-13):
    plt.scatter(x0[0], x0[1], s=5)
    delta_t=delta(t0,x0)
    x=x0
    t=t0
    if dir==1:
        while delta_t<0:
            print("step1")
            sol=solve_system(0.001,(dim,g),'RK45',t+0.1,t,x)
            x=np.array([sol.y[0,-1],sol.y[1,-1],sol.y[2,-1],sol.y[3,-1]])
            plt.scatter(x[0], x[1], s=5,c='red')
            t=t+0.1
            delta_t=delta(t,x)

        while delta_t>1:
            print("step2")
            sol=solve_system(0.001,(dim,g),'RK45',t+0.1,t,x)
            x=np.array([sol.y[0,-1],sol.y[1,-1],sol.y[2,-1],sol.y[3,-1]])
            plt.scatter(x[0], x[1], s=5,c='blue')
            delta_t=delta(t,x)
            t=t+0.1
    else:
        while delta_t>0:
            print("step3")
            sol=solve_system(0.001,(dim,g),'RK45',t-0.1,t,x)
            x=np.array([sol.y[0,-1],sol.y[1,-1],sol.y[2,-1],sol.y[3,-1]])
            plt.scatter(x[0], x[1], s=5,c='red')
            t=t-0.1
            delta_t=delta(t,x)

        while delta_t<-1:
            print("step4")
            sol=solve_system(0.001,(dim,g),'RK45',t-0.1,t,x)
            x=np.array([sol.y[0,-1],sol.y[1,-1],sol.y[2,-1],sol.y[3,-1]])
            plt.scatter(x[0], x[1], s=5,c='blue')
            delta_t=delta(t,x)
            t=t-0.1

    while np.abs(h(x))>eps:
        print("step5")
        delta_t=delta(t,x)
        #take in acount if it is forward or backward
        sol=solve_system(0.001,(dim,g),'RK45',t+delta_t,t,x)
        #selection of the solution
        x=np.array([sol.y[0,-1],sol.y[1,-1],sol.y[2,-1],sol.y[3,-1]])
        plt.scatter(x[0], x[1], s=5,c='green')
        t=t+delta_t
    plt.show()
    return(x,t)



