#!/usr/bin/env python3
# -------------------------------
# Barotropic Vorticity Equation
# -------------------------------
# @author Qiyu Song (sqy2017@pku.edu.cn)
# Nov 22, 2020

import numpy as np
import calc_ode as ode
import matplotlib.pyplot as plt


def generate_2pi(N):
    # Generate N points within 2*PI
    d = 2 * np.pi / N
    dhalf = d / 2
    gen_x = np.linspace(dhalf, 2 * np.pi - dhalf, N)
    return gen_x


def init(nx, ny):
    x = generate_2pi(nx)
    y = generate_2pi(NY)

    zeta_i = np.zeros((nx, ny))

    width = 10
    nx_half = int(nx/2)
    ny_half = int(ny/2)
    zeta_i[:, ny_half - width:ny_half] = -1
    zeta_i[:, ny_half:ny_half + width] = +1

    # perturbation
    zeta_i[nx_half - 1:nx_half + 1, ny_half - 1] -= 0.01
    zeta_i[nx_half - 1:nx_half + 1, ny_half] += 0.01
    return x, y, zeta_i

def laplace(z):
    c = np.fft.fft2(z)
    c = c * k2
    lap = np.fft.ifft2(c)
    return lap.real


def inv_laplace(z):
    c = np.fft.fft2(z)
    c = c * ik2
    i_lap = np.fft.ifft2(c)
    return i_lap.real


def pzpx_2d(z):
    c = np.fft.fft2(z)
    c = kx * c
    pzpx = np.fft.ifft2(c)
    return pzpx.real


def pzpy_2d(z):
    c = np.fft.fft2(z)
    c = ky * c
    pzpy = np.fft.ifft2(c)
    return pzpy.real


def tend_zeta(z):
    phi = inv_laplace(z)
    u = -1 * pzpy_2d(phi)
    v = pzpx_2d(phi)
    return -u * pzpx_2d(z) - v*pzpy_2d(z) + 1.e-4 * laplace(z)


##########
## Main ##
##########
if __name__=='__main__':
    # initialization
    NX = 128
    NY = 128
    kmax = int(NX/2)
    lmax = int(NY/2)
    jj = (0 + 1j)  # unit of imaginary number
    x, y, zeta = init(NX, NY)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # preparation for PSM
    kx1d = np.zeros(NX)
    kx1d[:kmax] = np.arange(kmax)
    kx1d[-1:-kmax:-1] = -1 * kx1d[1:kmax]
    kx1d[kmax] = kmax
    kx1d = kx1d * jj
    ky1d = kx1d  # x, y axises are the same in this project
    kx = kx1d[:, np.newaxis]
    ky = ky1d[np.newaxis, :]
    k2 = kx ** 2 + ky ** 2
    k2_temp = k2
    k2_temp[0, 0] = 1.
    ik2 = 1 / k2_temp
    ik2[0, 0] = 0.

    # for model run
    dt = 0.1  # second
    steps = 5000
    scheme = 'rk4'

    print("\n=== Pseudo Spectral Method")
    BVE = ode.ode(zeta, tend_zeta, dt, steps, 'rk4', debug=0)
    BVE.Integrate()

    for number in range(0, 5001, 10):
        fig = plt.figure()
        plt.contourf(x, y, np.transpose(BVE.traj[number, :, :]), np.linspace(-1.05, 1.05, 22), cmap='bwr')
        cbar = plt.colorbar()
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        plt.xticks(np.pi * np.array([0, 0.5, 1, 1.5, 2]), ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        plt.yticks(np.pi * np.array([0, 0.5, 1, 1.5, 2]), ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('t = ' + format(number/10, '0.1f') + ' s')
        plt.savefig('./figure/'+str(number).zfill(4)+'_vort.png')
        plt.close()

