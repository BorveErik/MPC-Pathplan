from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Constants
N = 20
dt = 0.2

# 2D motion
n = 4
m = 2
x = MX.sym('x',n)
u = MX.sym('u',m)

ref = MX.sym('ref',4)

ode = vertcat(dt*x[2],dt*x[3],u[0],u[1])

# Define discrete system based on RK4 integration
intg_opt = {'tf':dt,'simplify':True,'number_of_finite_elements': 4}
dae = {'x':x,'p':u,'ode':ode}

intg = integrator('intg','rk',dae,intg_opt)

res = intg(x0=x,p=u)
x_next = res['xf']

F = Function('F',[x,u],[x_next],['x','u'],['x_next'])


def repulsion(x):
    # Custom Gaussian functions representing obstacles in the potential field
    kx1 = 0.015
    ky1 = 0.7
    funk1 = exp(-kx1*(x[0]+5) ** 2-ky1*(x[1]+3) ** 2)
    kx2 = 1
    ky2 = 0.03
    funk2 = exp(-kx2*(x[0]-8) ** 2-ky2*(x[1]-3) ** 2)
    scale1 = 1
    scale2 = 1
    return scale1*funk1+scale2*funk2

def attractor(x,ref):
    # Global attractor towards final goal position
    scale = 1/500
    return scale*((x[0]-ref[0])*2.2*(x[0]-ref[0])+(x[1]-ref[1])*1*(x[1]-ref[1]))


# Loss function forumlation
L = Function('L',[x,u,ref],[attractor(x,ref) + repulsion(x)+ 1/10*dot(u,u)],['x','u','ref'],['Loss'])

# Formulating optimal control problem
optim = Opti()
x = optim.variable(n,N+1)
u = optim.variable(m,N)
p = optim.parameter(n,1)
ref = optim.parameter(n,N+1)

Cost = 0
for l in range(N):
    Cost += L(x[:,l],u[:,l],ref[:,l])

optim.minimize(Cost)

# # Constraints
for k in range(N):
    optim.subject_to(x[:,k+1] == F(x[:,k],u[:,k]))

ub = 0.5
lb = -0.5
optim.subject_to(optim.bounded(lb,vec(u),ub))
optim.subject_to(x[:,0]==p)

# Solver
s_opt = dict(print_iter = False,print_header =False,print_info = False)
p_opts = dict(qpsol = 'qrqp',print_header=False, print_iteration=False, print_time=False,qpsol_options = s_opt)
optim.solver('sqpmethod',p_opts)

# Compact MPC formulation
MPC = optim.to_function('MPC',[p,ref],[u[:,1]],['p','ref'],['u_opt'])

# System simulation
tend  = 300
T = int(tend/dt)
X = np.zeros((n,T,1))
U = np.zeros((m,T,1))
tspan = np.linspace(0,tend,T)

# Test sys
start = [-10,-9]
x_iter = DM(n,1)
x_iter[0,0] = start[0]
x_iter[1,0] = start[1]
x_iter[2,0] = 0
x_iter[3,0] = 0

goal = [3,5]
cont_ref = DM(n,N+1)
cont_ref[0,:] = goal[0]
cont_ref[1,:] = goal[1]
cont_ref[2,:] = 0
cont_ref[3,:] = 0

for i in range(T):
    u_iter = MPC(x_iter,cont_ref)
    X[:,i] = x_iter
    U[:,i] = u_iter
    x_iter = F(x_iter,u_iter)

    if np.linalg.norm(X[0:2,i]-np.array([goal]).T) < 0.25:
        print('Goal found')
        break
i_crit = i

# Color map for plotting
Nx = 100
Ny = 100
Z = np.zeros((Ny,Nx))

x_sim = np.linspace(-17,17,Nx)
y_sim = np.linspace(-17,17,Ny)
X_sim,Y_sim = np.meshgrid(x_sim,y_sim)

for i in range(Nx):
    for j in range(Ny):
        Z[j,i] = attractor([x_sim[i],y_sim[j]],goal)+repulsion([x_sim[i],y_sim[j]])


# Plotting
fig = plt.figure(constrained_layout=True)
gs = GridSpec(2, 3, figure=fig)
ax1 = fig.add_subplot(gs[0:2,0:2])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[1,2])

N_mid = int(40)
xm = X[0,N_mid,0]
dx = X[0,N_mid+1,0]-X[0,N_mid,0]
ym = X[1,N_mid,0]
dy = X[1,N_mid+1,0] - X[1,N_mid,0]

CS = ax1.contourf(Y_sim,X_sim,Z)
cbar = fig.colorbar(CS,ax=ax1,aspect=75)
cbar.ax.set_ylabel('Cost')
ax1.plot(X[1,0:i_crit,0],X[0,0:i_crit,0])
ax1.arrow(ym,xm,dy,dx,head_width = 0.8)
ax1.scatter(goal[1],goal[0],c='g',marker = 'o',linewidths=2)
ax1.scatter(start[1],start[0],c='g',marker = 'o',linewidths=2)
ax1.set_ylabel('y',fontsize=14)
ax1.set_xlabel('x',fontsize=14)
ax1.set_title('2D Trajectory',fontsize=18)

ax2.plot(tspan[0:i_crit],np.ones(i_crit)*ub,'--',color = 'y')
ax2.plot(tspan[0:i_crit],np.ones(i_crit)*lb,'--',color = 'y')
ax2.step(tspan[0:i_crit],U[1,0:i_crit,0])
ax2.set_title('y-axis motor signals')
ax2.set_ylabel('y')
ax2.set_xlabel('time (s)')

ax3.plot(tspan[0:i_crit],np.ones(i_crit)*ub,'--',color = 'y')
ax3.plot(tspan[0:i_crit],np.ones(i_crit)*lb,'--',color = 'y')
ax3.step(tspan[0:i_crit],U[0,0:i_crit,0])
ax3.set_title('x-axis motor signals')
ax3.set_ylabel('x')
ax3.set_xlabel('time (s)')

plt.show()