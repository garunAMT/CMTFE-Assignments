import numpy as np

def de_fault():
    global XL, YL, NDX, NDY, POWERX, POWERY, PI, MODE, KSOLVE, KIMP, RELAX, BIG, SMALL
    XL = 1.0
    YL = 1.0
    NDX = 20
    NDY = 20
    POWERX = 1.0
    POWERY = 1.0
    PI = 4.0 * np.arctan(1.0)
    MODE = 1
    KSOLVE = [1] * 1  # Assuming 1 element in the array
    KIMP = 1
    RELAX = [1.0 for _ in range(NFMAX)]
    BIG = 1e16
    SMALL = 1e-15

def update():
    global FOLD, F, RHOLD, RHO
    for i in range(1, L1 + 1):
        for j in range(1, M1 + 1):
            for nf in range(1, NFMAX + 1):
                FOLD[i, j, nf] = F[i, j, nf]
                RHOLD[i, j, nf] = RHO[i, j, nf]

def setup1():
    global L1, L2, L3, FDX, X, XDIF, M1, M2, M3, FDY, Y, YDIF
    L1 = NDX + 1
    L2 = L1 - 1
    L3 = L2 - 1
    FDX = float(NDX)

    X[0] = 0.0
    for i in range(2, L2 + 1):
        dd = (i - 1) / FDX
        if POWERX > 0.0:
            X[i] = XL * dd ** POWERX
        else:
            X[i] = XL * (1.0 - (1.0 - dd) ** (-POWERX))
    X[L1] = XL

    for i in range(2, L1 + 1):
        XDIF[i] = X[i] - X[i - 1]

    M1 = NDY + 1
    M2 = M1 - 1
    M3 = M2 - 1
    FDY = float(NDY)

    Y[0] = 0.0
    for j in range(2, M2 + 1):
        dd = (j - 1) / FDY
        if POWERY > 0.0:
            Y[j] = YL * dd ** POWERY
        else:
            Y[j] = YL * (1.0 - (1.0 - dd) ** (-POWERY))
    Y[M1] = YL

    for j in range(2, M1 + 1):
        YDIF[j] = Y[j] - Y[j - 1]

def setup2():
    global AP, SC, AIP, AIM, AJP, AJM
    for nf in range(1, NFMAX + 1):
        rel = 1.0 - RELAX[nf]
        if KSOLVE[nf] == 1:
            dense()
            gamsor()
            for i in range(2, L2 + 1):
                for j in range(2, M2 + 1):
                    apo = RHO[i, j, nf] / DT
                    apt = RHOLD[i, j, nf] / DT
                    rfn, rfs, pfe, pfw = 1.0, 1.0, 1.0, 1.0
                    if MODE == 2:
                        rfn = 0.5 * (1.0 + Y[j + 1] / Y[j])
                        rfs = 0.5 * (1.0 + Y[j - 1] / Y[j])
                    elif MODE == 3:
                        rfn = 0.5 * (1.0 + Y[j + 1] / Y[j])
                        rfs = 0.5 * (1.0 + Y[j - 1] / Y[j])
                        pfe = 1.0 / (Y[j] * Y[j])
                        pfw = 1.0 / (Y[j] * Y[j])
                    
                    game = 2.0 * GAM[i + 1, j] * GAM[i, j] / (GAM[i, j] + GAM[i + 1, j])
                    gamw = 2.0 * GAM[i - 1, j] * GAM[i, j] / (GAM[i, j] + GAM[i - 1, j])
                    gamn = 2.0 * GAM[i, j + 1] * GAM[i, j] / (GAM[i, j] + GAM[i, j + 1])
                    gams = 2.0 * GAM[i, j - 1] * GAM[i, j] / (GAM[i, j] + GAM[i, j - 1])

                    denox = 0.5 * (XDIF[i] + XDIF[i + 1])
                    denoy = 0.5 * (YDIF[j] + YDIF[j + 1])

                    ajp = rfn * gamn / (denoy * YDIF[j + 1])
                    ajm = rfs * gams / (denoy * YDIF[j])
                    aip = pfe * game / (denox * XDIF[i + 1])
                    aim = pfw * gamw / (denox * XDIF[i])

                    AP[i, j] = AP[i, j] - apo + aip + aim + ajp + ajm
                    AP[i, j] /= RELAX[nf]
                    SC[i, j] += apt * FOLD[i, j, nf]
                    SC[i, j] += AP[i, j] * rel * F[i, j, nf]

            lc()

            if KIMP == 1:
                tdma()
            elif KIMP == 0:
                explicit()

def tdma():
    global F
    ll2 = 2 * L2
    ll = ll2 - 2
    mm2 = 2 * M2
    mm = mm2 - 2

    # I-direction TDMA
    for jj in range(2, mm + 1):
        j = min(jj, mm2 - jj)
        P[0], Q[0] = 0.0, F[0, j, nf]
        for i in range(2, L2 + 1):
            a, b, c = AP[i, j], AIP[i, j], AIM[i, j]
            sc_i = AJP[i, j] * F[i, j + 1, nf] + AJM[i, j] * F[i, j - 1, nf]
            d = SC[i, j] + sc_i
            deno = a - c * P[i - 1]
            tnum = d + c * Q[i - 1]
            P[i] = b / deno
            Q[i] = tnum / deno

        for i in range(L2, 1, -1):
            F[i, j, nf] = P[i] * F[i + 1, j, nf] + Q[i]

    # J-direction TDMA
    for ii in range(2, ll + 1):
        i = min(ii, ll2 - ii)
        P[0], Q[0] = 0.0, F[i, 0, nf]
        for j in range(2, M2 + 1):
            a, b, c = AP[i, j], AJP[i, j], AJM[i, j]
            sc_j = AIP[i, j] * F[i + 1, j, nf] + AIM[i, j] * F[i - 1, j, nf]
            d = SC[i, j] + sc_j
            deno = a - c * P[j - 1]
            tnum = d + c * Q[j - 1]
            P[j] = b / deno
            Q[j] = tnum / deno

        for j in range(M2, 1, -1):
            F[i, j, nf] = P[j] * F[i, j + 1, nf] + Q[j]

def explicit():
    global F
    for i in range(2, L2 + 1):
        for j in range(2, M2 + 1):
            sum_ = AIP[i, j] * FOLD[i + 1, j, nf] + AIM[i, j] * FOLD[i - 1, j, nf] + AJP[i, j] * FOLD[i, j + 1, nf] + AJM[i, j] * FOLD[i, j - 1, nf] + SC[i, j]
            F[i, j, nf] = sum_ / AP[i, j]

def reset():
    global AP, SC
    for j in range(2, M2 + 1):
        for i in range(2, L2 + 1):
            AP[i, j] = 0.0
            SC[i, j] = 0.0
