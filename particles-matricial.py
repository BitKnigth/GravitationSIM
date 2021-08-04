import random
import numpy as np
import math
import random as rd
global G
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

G=-6.67408/(10**11)

IS = dict()
IS['Dx'] = np.array([[1.189],[0],[0]])
IS['Dy'] = np.array([[0],[0],[1]])
IS['Dz'] = np.array([[0],[0],[0]])

IS['Vx'] = np.array([[0.123],[0],[0.025]])
IS['Vy'] = np.array([[0.2],[0],[0.045]])
IS['Vz'] = np.array([[0],[0.15],[0]])

IS['m'] = np.array([[1],[10**9],[10**5]])
class trajectories:

    def __init__(self, D, h, m):
        self.positions = D
        self.masses = m
        self.opt_positions = list()
        for d in D:
            self.opt_positions.append(np.transpose(np.array([d[:,i] for i in range(0, len(self.positions[0][0]), int(1/h))])))

        self.n = len(self.opt_positions[0][0])
        print('n: ', self.n)
        self.colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'b', 'r', 'g', 'c', 'm', 'y', 'k']
        print('\nNumero de pontos antes da otimização:', len(self.positions[0][0]))
        print('Numero de pontos depois da otimização:', len(self.opt_positions[0][0]))
        return

    def free_trajectories(self):

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.title('Trajetórias Livres.')

        for i in range(len(self.opt_positions[0])):
            ax.plot3D(self.opt_positions[0][i], self.opt_positions[1][i], self.opt_positions[2][i], color=random.choice(self.colors))
        plt.show()

    def referential_trajectories(self, i=None, heavier=True):

        referencial = i
        if heavier:
            m = np.amax(self.masses)
            for j in range(len(self.opt_positions[0])):
                aux = self.masses[j][0]
                if aux == m:
                    referencial = j

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.title('Trajetórias no referencial.')

        for i in range(len(self.opt_positions[0])):
            ax.plot3D(self.opt_positions[0][i] - self.opt_positions[0][referencial],
                      self.opt_positions[1][i] - self.opt_positions[1][referencial],
                      self.opt_positions[2][i] - self.opt_positions[2][referencial],
                      color=random.choice(self.colors))
        plt.show()

    def evolution(self):

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(11, 7))
        fig.suptitle("Evolução", fontsize=16)
        ax = plt.axes(projection='3d')
        ax2 = plt.axes(projection='3d')
        ax3 = plt.axes(projection='3d')

        ax = plt.subplot(1,3,1)
        ax.set_title('Inicio')
        ax.scatter(self.opt_positions[0][:, 0], self.opt_positions[1][:, 0], self.opt_positions[2][:, 0],
                      color=random.choice(self.colors))
        ax2 = plt.subplot(1,3,2)
        ax2.set_title('Meio')
        ax2.scatter(self.opt_positions[0][:, self.n//2], self.opt_positions[1][:, self.n//2], self.opt_positions[2][:, self.n//2],
                  color=random.choice(self.colors))
        ax3 = plt.subplot(1,3,3)
        ax3.set_title('Fim')
        ax3.scatter(self.opt_positions[0][:, self.n - 1], self.opt_positions[1][:, self.n - 1], self.opt_positions[2][:, self.n - 1],)

        plt.show()

def initialize_particles(n, timesteps, random, initialState,  k=5000, M=10**9):

    Dx = np.zeros((n, timesteps))
    Dy = np.zeros((n, timesteps))
    Dz = np.zeros((n, timesteps))

    Vx = np.zeros((n, 1))
    Vy = np.zeros((n, 1))
    Vz = np.zeros((n, 1))

    #colocar para inicializar nas iterações
    ax = np.zeros((1, n))
    ay = np.zeros((1, n))
    az = np.zeros((1, n))

    m = np.zeros((n,1))

    if random:
        for i in range(n):
            Dx[i,0] = rd.randint(-1000000,1000000)
            Dy[i,0] = rd.randint(-1000000,1000000)
            Dz[i,0] = rd.randint(-1000000,1000000)

            Vx[i][0] = rd.randint(-25**4//math.sqrt(3), 25**4//math.sqrt(3))
            Vy[i][0] = rd.randint(-25**4//math.sqrt(3), 25**4//math.sqrt(3))
            Vz[i][0] = rd.randint(-25**4//math.sqrt(3), 25**4//math.sqrt(3))

            coef = rd.randint(1, k)
            m[i][0] = coef*M

    else:
        Dx[:, 0] = np.reshape(initialState['Dx'], (n,))
        Dy[:, 0] = np.reshape(initialState['Dy'], (n,))
        Dz[:, 0] = np.reshape(initialState['Dz'], (n,))

        Vx[:, 0] = np.reshape(initialState['Vx'], (n,))
        Vy[:, 0] = np.reshape(initialState['Vy'], (n,))
        Vz[:, 0] = np.reshape(initialState['Vz'], (n,))

        m = initialState['m']

    return [Dx, Dy, Dz], [Vx, Vy, Vz], m

def calc_a(D, m, n):
    D[0] = D[0] - np.transpose(D[0])
    D[1] = D[1] - np.transpose(D[1])
    D[2] = D[2] - np.transpose(D[2])
    module_d = np.sqrt(D[0]**2 + D[1]**2 + D[2]**2)

    radius = [d/np.transpose(module_d) for d in D]
    for r in radius:
        for i in range(n):
            r[i][i] = 0
    a_res = [np.dot(r, m*G) for r in radius]
    return a_res

def update(D, V, i, m, h, n):

    a_res = calc_a([np.reshape(d[:, i],(n,1)) for d in D], m, n)
    V = [(v+a*h) for a, v in zip(a_res,V)]

    Dx = D[0]
    Dy = D[1]
    Dz = D[2]

    Vx = V[0]
    Vy = V[1]
    Vz = V[2]

    Dx[:, i + 1] = np.reshape(np.reshape(Dx[:, i], (n, 1)) + Vx * h, (n,))
    Dy[:, i + 1] = np.reshape(np.reshape(Dy[:, i], (n, 1)) + Vy * h, (n,))
    Dz[:, i + 1] = np.reshape(np.reshape(Dz[:, i], (n, 1)) + Vz * h, (n,))

    return [Dx, Dy, Dz], V

def simulate(n, timesteps, h, random=True, initialState=None):

    t1 = time.time()

    D, V, m = initialize_particles(n, timesteps, random, initialState)

    for i in range(timesteps-1):
        D, V = update(D, V, i, m, h, n)

    t2 = time.time()
    print("\nConcluido em " + str(t2 - t1) + 'segundos.')
    return D, trajectories(D, h, m)

D, trajetorias = simulate(3, 700000, 0.01)
#trajetorias.evolution()
trajetorias.free_trajectories()
trajetorias.referential_trajectories()









