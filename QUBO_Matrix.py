import numpy as np
from dimod import BinaryQuadraticModel


def create_QUBO(T, K, P_k, A, B_a, Q_a, R_vend,R_acqu, Epf, Ecf, E_max, E0, Ek_min, g1, g2, g4,etac, etas, PaT, Pa0, Pvend, Pacqu, Pinc, costk, Eas_max,Eac_max,Ek_max, const=8):
    '''
    T       int,    number of time intervals.
    K       int,    number of generators.
    P_k     matrix, index: id_generator, timeslot,  value: number of binary variable for its production.
    A       int,    number of accaumulators.
    B_a     array,  index: id accumulator,  value: number of binary variables for charding/discharging.
    Q_a     array,  index: id accumulator,  value: number of binary variables for total charge.
    R_vend  array,  index: timeslot,        value: number of binary varibles that models the sold energy.
    R_acqu  array,  index: timeslot,        value: number of binary varibles that models the bought energy.
    Epf     array,  index: timeslot,        value: fixed production of energy for the timeslot.
    Ecf     array,  index: timeslot,        value: fixed consumption of energy for the timeslot.
    E_max   array,  index: id accumulator,  value: maximum energy in the accumulator
    E0      array,  index: id accumulator,  value: energy stored at time t.
    Ek_min  array,  index: id generator,    value: minimum energy produced by the generator.
    g1      int,    granularity for the generators, 
    g2      int,    granulartiy for the accumulators,
    g4      int,    granulartiy for the vend/acqu variables,
    etac    array,  index: id accumulator,  value: charging efficency coefficient.
    etas    array,  index: id accumulator,  value: discharging efficiency coefficient.
    PaT     int,    estimated price of energy at time T.
    Pa0     int,    estimated price of energy at time 0.
    Pvend   array,  index: timeslot,        value: Selling price for the timeslot
    Pacqu   array,  index: timeslot,        value: Buying price for the timeslot
    Pinc    int,    Price of the incentive 
    costk   array,  index: id generator,    value: Cost of utilizing the generator
    const   int,    number of constraints to consider, debugging option.
    Eas_max array,  index: id accumulator   value: maximum energy per unit of time in charge
    Eac_max array,  index: id accumulator   value: maximum energy per unit of time in discharge
    Ek_max  array,  index: id generator, timeslot,  value: maximum energy produced by k at time t
    RETURNS:
    bqm     BinaryQuadraticModel,   contains the binary quadratic model as Dwave requires
    Q       np.Array,               contains the upper diagonal matrix representing the problem
    Q1      dict,                   contains a sparse matrix in the form of a dictionary
    Offset  float,                  contains the offset of the computation.


    Description:
    le variabili che sono state identificate sono:
    xpk    -> produzione generatori,
    xcab   -> carica accumulatori,
    xsab   -> scarica accumulatori,
    xicaq  -> supporto carica accumulatori,
    xisaq  -> supporto scarica accumulatori,
    xrvend -> energia venduta,
    xracqu -> energia acquistata,
    xiince -> supporto per incentivo.
    '''
    totP        = sum(sum(P_k))
    totB        = sum(B_a)
    totQ        = sum(Q_a)
    totR_vend   = sum(R_vend)
    totR_acqu   = sum(R_acqu)
    totInc      = T
    totPsik     = sum(sum(P_k))
    #pk and r depend on the timestep so they dont need to be multiplied by T
    n_var = int(2*totP + T*totB*4 + T*totQ*2 + totR_vend + totR_acqu+T)
    xc_zero     = totP   
    xs_zero     = xc_zero + T*totB
    xic_zero    = xs_zero + T*totB
    xis_zero    = xic_zero + T*totQ
    xvend_zero  = xis_zero + T*totQ
    xacqu_zero  = xvend_zero + totR_vend
    xiinc_zero  = xacqu_zero + totR_acqu
    psipk_zero  = xiinc_zero + totInc
    psis_zero = psipk_zero + totP
    psic_zero = psis_zero + T*totB


    def xpk_point(t,k,p):
        if k==0:
            return sum(P_k[:k]) + sum(P_k[k,:t]) + p
        return sum(sum(P_k[:k])) + sum(P_k[k,:t]) + p
    def xcab_point(t,a,b):
        return xc_zero + t*totB + sum(B_a[:a])+b
    def xsab_point(t,a,b):
        return xs_zero + t*totB + sum(B_a[:a])+b
    def xicaq_point(t,a,q):
        return xic_zero + t*totQ + sum(Q_a[:a])+q
    def xisaq_point(t,a,q):
        return xis_zero + t*totQ + sum(Q_a[:a])+q
    def xrvend_point(t,r):
        return xvend_zero + sum(R_vend[:t]) +r
    def xracqu_point(t,r):
        return xacqu_zero + sum(R_acqu[:t]) +r
    def xiinc_point(t):
        return int(xiinc_zero) + t
    def psipk_point(t,k,p):
        if k==0:
            return psipk_zero + sum(P_k[:k]) + sum(P_k[k,:t]) + p
        return psipk_zero + sum(sum(P_k[:k])) + sum(P_k[k,:t]) + p
    def psisab_point(t,a,b):
        return psis_zero + t*totB + sum(B_a[:a])+b
    def psicab_point(t,a,b):
        return psic_zero + t*totB + sum(B_a[:a])+b


    g3 = g2
    E_MIN_K =sum(Ek_min)
    offset = 0
    Q = np.zeros((n_var,n_var))

    ro_cs = -1
    ro_va = -10
    ro_inc = -1
    ro_ac = -1
    ro_as = -1
    ro_e0 = -0.005
    ro_max = -0.01
    #obj function
    offset -= (PaT -Pa0 )*sum(E0)
    for t in range(T):
        xiinc = xiinc_point(t)
        for r in range(R_vend[t]):
            Q[xrvend_point(t,r),xrvend_point(t,r)]-= Pvend[t]*pow(2,r+g4)
        for r in range(R_acqu[t]):
            Q[xracqu_point(t,r),xracqu_point(t,r)]+= Pacqu[t]*pow(2,r+g4)
        Q[xiinc,xiinc] -= Pinc*E_MIN_K
        for k in range(K):
            for p in range(P_k[k][t]):
                Q[xpk_point(t,k,p),xiinc] -= Pinc*pow(2,p+g1)
        for a in range(A):
            for b in range(B_a[a]):
                Q[xsab_point(t,a,b),xiinc] -= Pinc*pow(2,b+g2)
        Q[xiinc,xiinc] -= Pinc*Ecf[t]
        Q[xiinc,xiinc] -= Pinc*Epf[t]
        offset += Pinc*Ecf[t]
        for a in range(A):
            for b in range(B_a[a]):
                Q[xcab_point(t,a,b),xcab_point(t,a,b)] -= Pinc*pow(2,b+g2)
                Q[xcab_point(t,a,b),xiinc] += Pinc*pow(2,b+g2)
        for k in range(K):
             for p in range(P_k[k][t]):
                Q[xpk_point(t,k,p),xpk_point(t,k,p)] += costk[k]*pow(2,p+g1)
        for a in range(A):
            for b in range(B_a[a]):
                Q[xcab_point(t,a,b),xcab_point(t,a,b)] -= PaT*etac[a]*pow(2,b+g2)
                Q[xsab_point(t,a,b),xsab_point(t,a,b)] += PaT*(1/etas[a])*pow(2,b+g2)
    if const == 0: 

        Q1= {(a,b):-1*Q[a,b] for a in range(len(Q)) for b in range(len(Q)) if Q[a,b] != 0}
        bqm = BinaryQuadraticModel.from_qubo(Q1, offset=-1*offset)
        Q = np.array(-1*Q, dtype=float)

        return bqm, Q, offset, Q1



    # rocs constraint
    for t in range(T):
        for a in range(A):
            for b1 in range(B_a[a]):
                for b2 in range(B_a[a]):
                    Q[xcab_point(t,a,b1),xsab_point(t,a,b2)] +=ro_cs

    if const == 1: 

        Q1= {(a,b):-1*Q[a,b] for a in range(len(Q)) for b in range(len(Q)) if Q[a,b] != 0}
        bqm = BinaryQuadraticModel.from_qubo(Q1, offset=-1*offset)
        Q = np.array(-1*Q, dtype=float)

        return bqm, Q, offset, Q1

    # rova constraint
    for t in range(T):
        for r1 in range(R_vend[t]):
            for r2 in range(R_acqu[t]):
                Q[xrvend_point(t,r1),xracqu_point(t,r2)] += ro_va
    if const == 2: 

        Q1= {(a,b):-1*Q[a,b] for a in range(len(Q)) for b in range(len(Q)) if Q[a,b] != 0}
        bqm = BinaryQuadraticModel.from_qubo(Q1, offset=-1*offset)
        Q = np.array(-1*Q, dtype=float)

        return bqm, Q, offset, Q1

    #roinc constraint
    for t in range(T):
        xiinc = xiinc_point(t)
        Q[xiinc,xiinc] += Epf[t]*ro_inc
        Q[xiinc,xiinc] += E_MIN_K*ro_inc
        for k in range(K):
            for p in range(P_k[k][t]):
                Q[xpk_point(t,k,p),xiinc] +=pow(2,p+g1)*ro_inc
        for a in range(A):
            for b in range(B_a[a]):
                Q[xsab_point(t,a,b),xiinc] += pow(2,b+g2)*ro_inc
        
        Q[xiinc,xiinc] -= Ecf[t]*ro_inc
        offset += Ecf[t]*ro_inc
        for a in range(A):
            for b in range(B_a[a]):
                Q[xcab_point(t,a,b),xcab_point(t,a,b)] -= pow(2,b+g2)*ro_inc
                Q[xcab_point(t,a,b),xiinc] -= pow(2,b+g2)*ro_inc
    
    if const == 3: 

        Q1= {(a,b):-1*Q[a,b] for a in range(len(Q)) for b in range(len(Q)) if Q[a,b] != 0}
        bqm = BinaryQuadraticModel.from_qubo(Q1, offset=-1*offset)
        Q = np.array(-1*Q, dtype=float)

        return bqm, Q, offset, Q1



    #romx constraint
    for t in range(T):
        #max for x_k(t)
        for k in range(K):
            E_delta=Ek_min-Ek_max[k][t]
            for p_1 in range(P_k[k][t]):
                for p_2 in range(P_k[k][t]):
                    Q[xpk_point(t,k,p_1),xpk_point(t,k,p_2)] +=pow(2,p_1+p_2+2*g1)*ro_max
                    Q[psipk_point(t,k,p_1),psipk_point(t,k,p_2)] +=pow(2,p_1+p_2+2*g1)*ro_max
                    Q[xpk_point(t,k,p_1),psipk_point(t,k,p_2)] +=pow(2,p_1+p_2+2*g1)*ro_max
                Q[xpk_point(t,k,p_1),xpk_point(t,k,p_1)] += pow(2,p_1+g1)*E_delta*ro_max
                Q[psipk_point(t,k,p_1),psipk_point(t,k,p_1)] +=pow(2,p_1+g1)*E_delta*ro_max
            offset+=E_delta**2*ro_max

        #max for x_ab^s
        for a in range(A):
            for b_1 in range(B_a[a]):
                for b_2 in range(B_a[a]):
                    Q[xsab_point(t,a,b_1),xsab_point(t,a,b_2)] +=pow(2,b_1+b_2+2*g2)*ro_max
                    Q[psisab_point(t,a,b_1),psisab_point(t,a,b_2)] +=pow(2,b_1+b_2+2*g2)*ro_max
                    Q[xsab_point(t,a,b_1),psisab_point(t,a,b_2)] +=pow(2,b_1+b_2+2*g2)*ro_max
                Q[xsab_point(t,a,b_1),xsab_point(t,a,b_1)] += pow(2,b_1+g2)*Eas_max*ro_max
                Q[psisab_point(t,a,b_1),psisab_point(t,a,b_1)] +=pow(2,b_1+g2)*Eas_max[a]*ro_max
            offset+=Eas_max**2*ro_max

        for a in range(A):
            for b_1 in range(B_a[a]):
                for b_2 in range(B_a[a]):
                    Q[xcab_point(t,a,b_1),xcab_point(t,a,b_2)] +=pow(2,b_1+b_2+2*g2)*ro_max
                    Q[psicab_point(t,a,b_1),psicab_point(t,a,b_2)] +=pow(2,b_1+b_2+2*g2)*ro_max
                    Q[xcab_point(t,a,b_1),psicab_point(t,a,b_2)] +=pow(2,b_1+b_2+2*g2)*ro_max
                Q[xcab_point(t,a,b_1),xcab_point(t,a,b_1)] += pow(2,b_1+g2)*Eac_max*ro_max
                Q[psicab_point(t,a,b_1),psicab_point(t,a,b_1)] +=pow(2,b_1+g2)*Eac_max[a]*ro_max
            offset+=Eac_max**2*ro_max


    if const == 4: 

        Q1= {(a,b):-1*Q[a,b] for a in range(len(Q)) for b in range(len(Q)) if Q[a,b] != 0}
        bqm = BinaryQuadraticModel.from_qubo(Q1, offset=-1*offset)
        Q = np.array(-1*Q, dtype=float)

        return bqm, Q, offset, Q1



    #roac constraint
    for t in range(T):
        for a in range(A):
            offset += E0[a]**2*ro_ac
            offset += E_max[a]**2*ro_ac
            offset -= E_max[a]*2*E0[a]*ro_ac
            for tao1 in range(t):
                for b1 in range(B_a[a]):
                    Q[xcab_point(tao1,a,b1), xcab_point(tao1,a,b1)] += E0[a] *etac[a]*pow(2,b1+g1+1)*ro_ac
                    Q[xsab_point(tao1,a,b1), xsab_point(tao1,a,b1)] -= E0[a] *(1/(etas[a]+1))*pow(2,b1+g1)*ro_ac
                    for tao2 in range(t):
                        for b2 in range(B_a[a]):
                            Q[xcab_point(tao1,a,b1),xcab_point(tao2,a,b2)] += etac[a]**2 *pow(2,b1+b2+2*g2) *ro_ac
                            Q[xcab_point(tao1,a,b1),xsab_point(tao2,a,b2)] -= etac[a]/etas[a] *pow(2,b1+b2+2*g2) *ro_ac
                            Q[xsab_point(tao1,a,b1),xcab_point(tao2,a,b2)] -= etac[a]/etas[a]*pow(2,b1+b2+2*g2) *ro_ac
                            Q[xsab_point(tao1,a,b1),xsab_point(tao2,a,b2)] += (1/etas[a]**2) *pow(2,b1+b2+2*g2) *ro_ac
            for b1 in range(B_a[a]):
                for b2 in range(B_a[a]):
                    Q[xcab_point(t,a,b1),xcab_point(t,a,b2)] += etac[a]**2 *pow(2,b1+b2+2*g2) *ro_ac
            for q1 in range(Q_a[a]):
                for q2 in range(Q_a[a]):
                    Q[xicaq_point(t,a,q1),xicaq_point(t,a,q2)] += pow(2,q1+q2+2*g3)*ro_ac
            for b in range(B_a[a]):
                Q[xcab_point(t,a,b),xcab_point(t,a,b)] += etac[a]*E0[a]*pow(2,b+g2+1) * ro_ac
            for tao in range(t):
                for b1 in range(B_a[a]):
                    for b2 in range(B_a[a]):
                        Q[xcab_point(tao,a,b1),xcab_point(t,a,b2)] += etac[a]**2* pow(2,b1+b2+2*g2+1)*ro_ac
                        Q[xsab_point(tao,a,b1),xcab_point(t,a,b2)] -= etac[a]/etas[a]* pow(2,b1+b2+2*g2+1)*ro_ac
            for tao in range(t):
                for b in range(B_a[a]):
                    Q[xcab_point(tao,a,b),xcab_point(tao,a,b)] -= E_max[a]*etac[a]*pow(2,b+g2+1)*ro_ac
                    Q[xsab_point(tao,a,b),xsab_point(tao,a,b)] -= E_max[a]*(1/etas[a])*pow(2,b+g2+1)*ro_ac
            for q in range(Q_a[a]):
                Q[xicaq_point(t,a,q),xicaq_point(t,a,q)] +=E0[a]*pow(2, q+g3+1)*ro_ac
            for tao in range(t):
                for b in range(B_a[a]):
                    for q in range(Q_a[a]):
                        Q[xcab_point(tao,a,b),xicaq_point(t,a,q)] += etac[a]*pow(2,q+b+g2+g3+1)*ro_ac
                        Q[xsab_point(tao,a,b),xicaq_point(t,a,q)] -= (1/etas[a])*pow(2,q+b+g2+g3+1)*ro_ac
            for b in range(B_a[a]):
                Q[xcab_point(t,a,b),xcab_point(t,a,b)] -= etac[a]*E_max[a]*pow(2,b+g2+1) * ro_ac

            for q in range(Q_a[a]):
                for b in range(B_a[a]):
                    Q[xcab_point(t,a,b),xicaq_point(t,a,q)] +=etac[a]*pow(2,q+b+2*g3+1)*ro_ac

            for q in range(Q_a[a]):
                Q[xicaq_point(t,a,q),xicaq_point(t,a,q)] -= E_max[a]+pow(2,q+g3+1)*ro_ac
    
    if const == 5: 

        Q1= {(a,b):-1*Q[a,b] for a in range(len(Q)) for b in range(len(Q)) if Q[a,b] != 0}
        bqm = BinaryQuadraticModel.from_qubo(Q1, offset=-1*offset)
        Q = np.array(-1*Q, dtype=float)
   
        return bqm, Q, offset, Q1

    #roas constraint
    for t in range(T):
        for a in range(A):
            offset += E0[a]**2*ro_as
            for tao1 in range(t):
                for b1 in range(B_a[a]):
                    Q[xcab_point(tao1,a,b), xcab_point(tao1,a,b)] += E0[a]*etac[a]*pow(2,b+g2+1)*ro_as
                    Q[xsab_point(tao1,a,b), xsab_point(tao1,a,b)] -= E0[a]*(1/etas[a])*pow(2,b+g2+1)*ro_as
                    for tao2 in range(t):
                        for b2 in range(B_a[a]):
                            Q[xcab_point(tao1,a,b1),xcab_point(tao2,a,b2)] += etac[a]**2 *pow(2,b1+b2+2*g2) *ro_as
                            Q[xcab_point(tao1,a,b1),xsab_point(tao2,a,b2)] -= etac[a]/etas[a] *pow(2,b1+b2+2*g2) *ro_as
                            Q[xsab_point(tao1,a,b1),xcab_point(tao2,a,b2)] -= etac[a]/etas[a]*pow(2,b1+b2+2*g2) *ro_as
                            Q[xsab_point(tao1,a,b1),xsab_point(tao2,a,b2)] += (1/etas[a]**2) *pow(2,b1+b2+2*g2) *ro_as
            for b1 in range(B_a[a]):
                for b2 in range(B_a[a]):
                    Q[xsab_point(t,a,b1),xsab_point(t,a,b2)] += (1/etac[a]**2) *pow(2,b1+b2+2*g2) *ro_as
            for q1 in range(Q_a[a]):
                for q2 in range(Q_a[a]):
                    Q[xisaq_point(t,a,q1),xisaq_point(t,a,q2)] += pow(2,q1+q2+2*g3)*ro_ac
            for b in range(B_a[a]):
                Q[xsab_point(t,a,b),xsab_point(t,a,b)] += (1/etac[a])*E0[a]*pow(2,b+g2+1) * ro_as
            for tao in range(t):
                for b1 in range(B_a[a]):
                    for b2 in range(B_a[a]):
                        Q[xsab_point(tao,a,b1),xsab_point(t,a,b2)] += (1/etac[a])**2* pow(2,b1+b2+2*g2+1)*ro_as
                        Q[xcab_point(tao,a,b1),xsab_point(t,a,b2)] -= etac[a]/etas[a]* pow(2,b1+b2+2*g2+1)*ro_as
            for q in range(Q_a[a]):
                Q[xisaq_point(t,a,q),xisaq_point(t,a,q)] -= E0[a]*pow(2,q+g3+1)*ro_as
            for tao in range(t):
                for b in range(B_a[a]):
                    for q in range(Q_a[a]):
                        Q[xcab_point(tao,a,b), xisaq_point(t,a,q)] += pow(2,b+g2+q+g3+1)*etac[a]*ro_as 
                        Q[xsab_point(tao,a,b), xisaq_point(t,a,q)] += pow(2,b+g2+q+g3+1)*(1/etac[a])*ro_as 
            for q in range(Q_a[a]):
                for b in range(B_a[a]):
                    Q[xsab_point(t,a,b), xisaq_point(t,a,q)] += pow(2, b+2*g2+q+1)*(1/etac[a])*ro_as 

    if const == 6: 

        Q1= {(a,b):-1*Q[a,b] for a in range(len(Q)) for b in range(len(Q)) if Q[a,b] != 0}
        bqm = BinaryQuadraticModel.from_qubo(Q1, offset=-1*offset)
        Q = np.array(-1*Q, dtype=float)

        return bqm, Q, offset, Q1

    #roe0 constraint
    for t in range(T):
        offset += E_MIN_K**2*ro_e0
        offset += Ecf[t]**2*ro_e0
        offset += Epf[t]**2*ro_e0
        for k1 in range(K):
            for p1 in range(P_k[k1][t]):
                for k2 in range(K):
                    for p2 in range(P_k[k2][t]):
                        Q[xpk_point(t,k1,p1),xpk_point(t,k2,p2)] += pow(2, p1+p2+2*g1) *ro_e0
                Q[xpk_point(t,k1,p1),xpk_point(t,k1,p1)] += E_MIN_K*pow(2,p+g1+1) * ro_e0

        for a1 in range(A):
            for a2 in range(A):
                for b1 in range(B_a[a1]):
                    for b2 in range(B_a[a2]):
                        Q[xsab_point(t,a1,b1),xsab_point(t,a2,b2)] += pow(2, b1+b2+2*g2)* ro_e0

        for r1 in range(R_acqu[t]):
            for r2 in range(R_acqu[t]):
                Q[xracqu_point(t,r1), xracqu_point(t,r2)] += pow(2,r1+r2+2*g4) * ro_e0
        for r1 in range(R_vend[t]):
            for r2 in range(R_vend[t]):
                Q[xrvend_point(t,r1), xrvend_point(t,r2)] += pow(2,r1+r2+2*g4) * ro_e0

        for a1 in range(A):
            for a2 in range(A):
                for b1 in range(B_a[a1]):
                    for b2 in range(B_a[a2]):
                        Q[xcab_point(t,a1,b1),xcab_point(t,a2,b2)] += pow(2, b1+b2+2*g2)* ro_e0


        for a in range(A):
            for b in range(B_a[a]):
                Q[xsab_point(t,a,b), xsab_point(t,a,b)] += pow(2, b+g2+1) * E_MIN_K *ro_e0
    
        for k in range(K):
            for p in range(P_k[k][t]):
                for a in range(A):
                    for b in range(B_a[a]):
                        Q[xpk_point(t,k,p),xsab_point(t,a,b)] += pow(2, p+b+g2+g1+1)* ro_e0

        for r in range(R_acqu[t]):
            Q[xracqu_point(t,r),xracqu_point(t,r)] += E_MIN_K*pow(2,r+g4+1)*ro_e0

        for k in range(K):
            for p in range(P_k[k][t]):
                for r in range(R_acqu[t]):
                     Q[xpk_point(t,k,p), xracqu_point(t,r)] += pow(2,p+g1+r+g4+1) *ro_e0

        for r in range(R_vend[t]):
            Q[xrvend_point(t,r),xrvend_point(t,r)] += E_MIN_K*pow(2,r+g4+1)*ro_e0

        for k in range(K):
            for p in range(P_k[k][t]):
                for r in range(R_vend[t]):
                     Q[xpk_point(t,k,p), xrvend_point(t,r)] += pow(2,p+g1+r+g4+1) *ro_e0

        offset -= 2*Ecf[t]*E_MIN_K*ro_e0
        
        for k in range(K):
            for p in range(P_k[k][t]):
                Q[xpk_point(t,k,p),xpk_point(t,k,p)] -= Ecf[t]*pow(2,p+g1+1)*ro_e0

        for a in range(A):
            for b in range(B_a[a]):
                Q[xcab_point(t,a,b),xcab_point(t,a,b)] -= E_MIN_K*pow(2, b+g2+1)*ro_e0

        for k in range(K):
            for p in range(P_k[k][t]):
                for a in range(A):
                    for b in range(B_a[a]):
                        Q[xpk_point(t,k,p), xcab_point(t,a,b)] -= pow(2, p+g1+b+g2+1)*ro_e0

        offset += 2*Epf[t]*E_MIN_K*ro_e0

        for k in range(K):
            for p in range(P_k[k][t]):
                Q[xpk_point(t,k,p),xpk_point(t,k,p)] += Epf[t]*pow(2,p+g1+1)*ro_e0

        for a in range(A):
            for b in range(B_a[a]):
                for r in range(R_acqu[t]):
                    Q[xsab_point(t,a,b), xracqu_point(t,r)] += pow(2, r+g4+b+g2+1)*ro_e0
                for r in range(R_vend[t]):
                    Q[xsab_point(t,a,b), xrvend_point(t,r)] -= pow(2, r+g4+b+g2+1)*ro_e0

        for a in range(A):
            for b in range(B_a[a]):
                Q[xsab_point(t,a,b), xsab_point(t,a,b)] -= Ecf[t]*pow(2,b+g2+1)*ro_e0
            
        for a1 in range(A):
            for b1 in range(B_a[a1]):
                for a2 in range(A):
                    for b2 in range(B_a[a2]):
                        Q[xsab_point(t,a1,b1),xcab_point(t,a2,b2)] -= pow(2, b1+b2+2*g2+1)*ro_e0

        for a in range(A):
            for b in range(B_a[a]):
                Q[xsab_point(t,a,b), xsab_point(t,a,b)] += Epf[t]*pow(2,b+g2+1)*ro_e0
            

        for r1 in range(R_acqu[t]):
            for r2 in range(R_vend[t]):
                Q[xrvend_point(t,r2),xracqu_point(t,r1)] -= pow(2,r1+r2+2*g4+1)*ro_e0

        for r in range(R_acqu[t]):
            Q[xracqu_point(t,r),xracqu_point(t,r)] -= Ecf[t]*pow(2,r+g4+1)*ro_e0

        for r in range(R_acqu[t]):
            for a in range(A):
                for b in range(B_a[a]):
                    Q[xracqu_point(t,r),xcab_point(t,a,b)] -= pow(2,r+b+g2+g4)*ro_e0

        for r in range(R_acqu[t]):
            Q[xracqu_point(t,r),xracqu_point(t,r)] += Epf[t]*pow(2,r+g4+1)*ro_e0
        for r in range(R_vend[t]):
            Q[xrvend_point(t,r),xrvend_point(t,r)] += Epf[t]*pow(2,r+g4+1)*ro_e0

        for r in range(R_vend[t]):
            for a in range(A):
                for b in range(B_a[a]):
                    Q[xrvend_point(t,r),xcab_point(t,a,b)] += pow(2,r+b+g2+g4)*ro_e0

        for r in range(R_vend[t]):
            Q[xrvend_point(t,r),xrvend_point(t,r)] -= Epf[t]*pow(2,r+g4+1)*ro_e0

        for a in range(A):
            for b in range(B_a[a]):
                Q[xcab_point(t,a,b), xcab_point(t,a,b)] += Ecf[t]*pow(2,b+g2+1)*ro_e0
                Q[xcab_point(t,a,b), xcab_point(t,a,b)] -= Epf[t]*pow(2,b+g2+1)*ro_e0

        offset *= 2*Ecf[t]*Epf[t]*ro_e0




    Q1= {(a,b):-1*Q[a,b] for a in range(len(Q)) for b in range(len(Q)) if Q[a,b] != 0}
    bqm = BinaryQuadraticModel.from_qubo(Q1, offset=-1*offset)
    Q = np.array(-1*Q, dtype=float)

    return bqm, Q, offset, Q1