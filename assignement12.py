from assignement8 import *
import numpy as np
from matplotlib import pyplot as plt

###Global variable

##Warning: don't put all the booléen variables equals to True, just one at max

plot=False ##True if we want to plot on the same graph all the manifold
separate_plot=True##True if we want the manifolds on differents graph
Poincaré=False ##True if we want the projection of the manifolds on the Poincaré section

mu=0.01
dim=20

x0=[1.033366313746765,0,0,-0.05849376854515592,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
t=3.114802556760205*2

##Génération des conditions initiales sur l'orbite périodique

sol_ini=solve_system_NP_var(20,(dim,f),'RK45',t,0,x0)

plt.plot(sol_ini.y[0],sol_ini.y[1])

for i in range(len(sol_ini.y[0])):
    plt.scatter(sol_ini.y[0][i],sol_ini.y[1][i],color="red")

plt.scatter(x0[0],x0[1],color="red")
plt.scatter(sol_ini.y[0][-1],sol_ini.y[1][-1],color="green")
plt.show()

##Computation of the Monodromy matrix

sol_mon=sol_ini.y[4:20]
monodromy=np.zeros((4,4))

c=0
for i in range(4):
    for j in range(4):
        monodromy[i][j]=sol_mon[i+j+c][-1]
    c=c+3


##Eigen values and eigein vector of the Monodromy matrix

eig_val,eig_vect=np.linalg.eig(monodromy)


lambda1=eig_val[0].real
lambda2=eig_val[1].real
v1=eig_vect[:,0].real
v2=eig_vect[:,1].real
print(lambda1,lambda2)
# print(v1,v2)

###Stable Manifold
##Positive part
s=10**-6 #0.13
x0=[sol_ini.y[0][-1],sol_ini.y[1][-1],sol_ini.y[2][-1],sol_ini.y[3][-1]]
v2=v2/np.linalg.norm(v2)
a=x0+s*v1
G_a=x0+lambda2*v1*s

seg=np.linspace(a,G_a,10)
dim=4
eps=10**-3

print("Computation of the Stable manifold: Positive part ")
for i in range(0,len(seg)):
    t0=0
    lx=[]
    ly=[]
    sol_x=np.array([])
    sol_y=np.array([])
    print(i)

    sol=solve_system(0.01,(dim,g),'RK45',t0+0.01,t0,seg[i])
    lx.append(sol.y[0])
    ly.append(sol.y[1])

    while np.abs(sol.y[0][-1]-0.2)>eps:
        t0=t0+0.01
        x0=[sol.y[0][-1],sol.y[1][-1],sol.y[2][-1],sol.y[3][-1]]
        sol=solve_system(0.01,(dim,g),'RK45',t0+0.01,t0,x0)
        lx.append(sol.y[0])
        ly.append(sol.y[1])
    for k in range(len(lx)):
        sol_x=np.concatenate((sol_x,lx[k]))
        sol_y=np.concatenate((sol_y,ly[k]))

    if plot:
        if i==0:
            plt.plot(sol_x,sol_y,"black",label="Stable Manifold Positive Part")
        else:
            plt.plot(sol_x,sol_y,"black")
        plt.legend()

    if separate_plot:
        plt.plot(sol_x,sol_y)

    if i==0: #for plot scalability
        b=0

    if Poincaré:
        poincar_x=[]
        poincar_y=[]
        for k in range(len(sol_x)):
            if np.abs(sol_x[k]-0.2)<10**-2:
                poincar_x.append(sol_x[k])
                poincar_y.append(sol_y[k])
        if b==0:
            plt.plot(poincar_x,poincar_y,"black",label="Stable Manifold Positive Part")
            b=b+1
        else:
            plt.plot(poincar_x,poincar_y,"black")



if separate_plot:
    plt.title("Stable manifold positive part")
    plt.show()

if Poincaré:
    plt.legend()


##Stable manifold negative Part
s=10**-6
x0=[sol_ini.y[0][-1],sol_ini.y[1][-1],sol_ini.y[2][-1],sol_ini.y[3][-1]]
a=x0-s*v1
G_a=x0-lambda2*v1*s

seg=np.linspace(a,G_a,10)
dim=4
eps=10**-3

print("Computation of the Stable manifold: Negative part ")
for i in range(0,len(seg)):
    t0=0
    lx=[]
    ly=[]
    sol_x=np.array([])
    sol_y=np.array([])
    print(i)
    sol=solve_system(0.01,(dim,g),'RK45',t0+0.01,t0,seg[i])
    lx.append(sol.y[0])
    ly.append(sol.y[1])
    while np.abs(sol.y[0][-1]-0.2)>eps:
        t0=t0+0.01
        x0=[sol.y[0][-1],sol.y[1][-1],sol.y[2][-1],sol.y[3][-1]]
        sol=solve_system(0.01,(dim,g),'RK45',t0+0.01,t0,x0)
        lx.append(sol.y[0])
        ly.append(sol.y[1])
    for k in range(len(lx)):
        sol_x=np.concatenate((sol_x,lx[k]))
        sol_y=np.concatenate((sol_y,ly[k]))

    if plot:
        if i==0:
            plt.plot(sol_x,sol_y,"red",label="Stable Manifold Positive Part")
        else:
            plt.plot(sol_x,sol_y,"red")
        plt.legend()

    if separate_plot:
        plt.plot(sol_x,sol_y)

    if i==0:
        b=0

    if Poincaré:
        poincar_x=[]
        poincar_y=[]
        for k in range(len(sol_x)):
            if np.abs(sol_x[k]-0.2)<10**-2:
                poincar_x.append(sol_x[k])
                poincar_y.append(sol_y[k])
        if b==0:
            plt.plot(poincar_x,poincar_y,"red",label="Stable Manifold Positive Part")
            b=b+1
        else:
            plt.plot(poincar_x,poincar_y,"red")

if separate_plot:
    plt.title("Stable Manifold negative Part")
    plt.show()

if Poincaré:
    plt.title("Projection On the Poincaré section")
    plt.legend()


##Unstable Manifold Positive part
s=1/2*10**-6
x0=[sol_ini.y[0][-1],sol_ini.y[1][-1],sol_ini.y[2][-1],sol_ini.y[3][-1]]
a=x0+s*v2
G_a=x0+lambda1*v2*s

seg=np.linspace(a,G_a,10)
dim=4
eps=10**-3

print("Computation of the Unstable manifold: Positive part ")
for i in range(0,len(seg)):
    t0=0
    lx=[]
    ly=[]
    sol_x=np.array([])
    sol_y=np.array([])
    print(i)

    sol=solve_system(0.01,(dim,g),'RK45',t0-0.01,t0,seg[i])
    lx.append(sol.y[0])
    ly.append(sol.y[1])
    while np.abs(sol.y[0][-1]-0.2)>eps:
        t0=t0-0.01
        x0=[sol.y[0][-1],sol.y[1][-1],sol.y[2][-1],sol.y[3][-1]]
        sol=solve_system(0.01,(dim,g),'RK45',t0-0.01,t0,x0)
        lx.append(sol.y[0])
        ly.append(sol.y[1])
    for k in range(len(lx)):
        sol_x=np.concatenate((sol_x,lx[k]))
        sol_y=np.concatenate((sol_y,ly[k]))

    if plot:
        if i==0:
            plt.plot(sol_x,sol_y,"green",label="Stable Manifold Positive Part")
        else:
            plt.plot(sol_x,sol_y,"green")
        plt.legend()

    if separate_plot:
        plt.plot(sol_x,sol_y)

    if i==0:
        b=0

    if Poincaré:
        poincar_x=[]
        poincar_y=[]
        for k in range(len(sol_x)):
            if np.abs(sol_x[k]-0.2)<10**-2:
                poincar_x.append(sol_x[k])
                poincar_y.append(sol_y[k])
        if b==0:
            plt.plot(poincar_x,poincar_y,"green",label="Stable Manifold Positive Part")
            b=b+1
        else:
            plt.plot(poincar_x,poincar_y,"green")


if separate_plot:
    plt.title(" Unstable Manifold Positive Part")
    plt.show()

if Poincaré:
    plt.legend()

##Negative Part
# #
s=10**-6
x0=[sol_ini.y[0][-1],sol_ini.y[1][-1],sol_ini.y[2][-1],sol_ini.y[3][-1]]
a=x0-s*v2
G_a=x0-lambda1*v2*s

seg=np.linspace(a,G_a,10)
dim=4
eps=10**-3

print("Computation of the Unstable manifold: Negative part ")
for i in range(0,len(seg)):
    t0=0
    lx=[]
    ly=[]
    sol_x=np.array([])
    sol_y=np.array([])
    print(i)

    sol=solve_system(0.01,(dim,g),'RK45',t0-0.01,t0,seg[i])
    lx.append(sol.y[0])
    ly.append(sol.y[1])
    while np.abs(sol.y[0][-1]-0.2)>eps:
        t0=t0-0.01
        x0=[sol.y[0][-1],sol.y[1][-1],sol.y[2][-1],sol.y[3][-1]]
        sol=solve_system(0.01,(dim,g),'RK45',t0-0.01,t0,x0)
        lx.append(sol.y[0])
        ly.append(sol.y[1])

    for k in range(len(lx)):
        sol_x=np.concatenate((sol_x,lx[k]))
        sol_y=np.concatenate((sol_y,ly[k]))

    if plot:
        if i==0:
            plt.plot(sol_x,sol_y,"orange",label="Stable Manifold Positive Part")
        else:
            plt.plot(sol_x,sol_y,"orange")
        plt.legend()

    if separate_plot:
        plt.plot(sol_x,sol_y)

    if i==0:
        b=0

    if Poincaré:
        poincar_x=[]
        poincar_y=[]
        for k in range(len(sol_x)):
            if np.abs(sol_x[k]-0.2)<10**-2:
                poincar_x.append(sol_x[k])
                poincar_y.append(sol_y[k])
        if b==0:
            plt.plot(poincar_x,poincar_y,"orange",label="Stable Manifold Positive Part")
            b=b+1
        else:
            plt.plot(poincar_x,poincar_y,"orange")



if separate_plot:
    plt.title("Unstable Manifold negative Part")
    plt.show()

if Poincaré:
    plt.legend()

plt.show()


