from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from decimal import Decimal as D
import time

global G
global timesteps
global h
import time
G = 6.67408/(10**11)
timesteps = 200000
h = 0.001
class particle:
    '''Particula com duas propriedades, uma velocidade e uma posição'''

    # S_0 é um nparray: [[X_0], [Y_0]] (posição inicial)
    # V_0 é um nparray: [[Vx_0], [Vy_0]] (velocidade inicial)
    def __init__(self, S_0, V_0, m): # Construtor do objeto

        self.posicao = np.zeros((3, timesteps), dtype='float64')
        self.velocidade = np.zeros((3, timesteps), dtype='float64')
        self.mass = m
        self.posicao[:, 0] = np.reshape(S_0,(3,))
        self.velocidade[:, 0] = np.reshape(V_0,(3,))
        return

    def update(self, i, a_res, h):
        self.velocidade[:, i+1] = self.velocidade[:, i] + a_res * h
        self.posicao[:, i+1] = self.posicao[:, i] + self.velocidade[:, i] * h

    def calc_a_Self_n(self, Mn, Pn, i):

        # p_n é a posição no formato [[Xn],[Yn]]
        debaxo = sqrt( (Pn[0] - self.posicao[0][i])**2 + (Pn[1] - self.posicao[1][i])**2 + (Pn[2] - self.posicao[2][i])**2)**3
        a = G * Mn * (Pn - self.posicao[:,i]) / debaxo
        return a


# Inicialização das particulas ------------------------------------
particles = list(range(3))
particles[0] = particle(np.array( [[1.189],[0], [0]]), np.array( [[0.123],[0.2], [0]]), 1)
particles[1] = particle(np.array( [[0],[0],[0]]), np.array( [[0],[0],[0.15]]), 10**9)
particles[2] = particle(np.array( [[0],[1],[0]]), np.array( [[0.025],[0.045],[0]]), 10**5)

t1 = time.time()
# Simulação -------------------------------------------------------
for i in range(timesteps - 1):

    for p in particles:
        acelerations = np.zeros((3, len(particles)-1))
        particles_aux = [x for x in particles if x != p] # Redundante!!
        for j in range(len(particles_aux)):
            acelerations[:,j] = p.calc_a_Self_n(particles_aux[j].mass, particles_aux[j].posicao[:,i], i)
        p.update(i, np.sum(acelerations, axis=1), h)

t2 = time.time()

print("Concluido em " + str(t2-t1) + ' segundos.')
P0 = particles[0].posicao
P1 = particles[1].posicao
P2 = particles[2].posicao


fig = plt.figure()
ax = plt.axes(projection='3d')
plt.title('Em tres dimensões.')
ax.plot3D(P0[0],P0[1],P0[2], color='r')
ax.plot3D(P1[0],P1[1],P1[2], color='k')
ax.plot3D(P2[0],P2[1],P2[2], color='b')
plt.show()

'''
for i in range(80):
    plt.plot(P0[0][i*500], P0[1][i*500], color='m')
    plt.plot(P1[0][i*500], P0[1][i*500] color='r')
    plt.show()
    time.sleep(0.5)
    
'''