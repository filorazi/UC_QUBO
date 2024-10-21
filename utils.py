import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from sklearn.metrics import  ConfusionMatrixDisplay, confusion_matrix


import xml.etree.ElementTree as ET
import os 
import numpy as np
import sys

import pandas as pd

def plot_qubo(bqm):
    import plotly.express as px

    qubo_matrix = np.array(bqm.to_numpy_matrix(), dtype=float)

    fig = px.imshow(qubo_matrix, 
                    color_continuous_scale=[[0, 'blue'], [0.5, 'white'], [1.0,'red']],
                    color_continuous_midpoint=0,
                    width=600, height=600,
                    title='QUBO matrix:')
    return fig

def highlight_qubo(samplevalues, Q):
    highlighted_matrix = np.where(np.outer(samplevalues,samplevalues) == 1, Q, np.nan)

    highlighted_matrix = np.where(highlighted_matrix!= 0, highlighted_matrix, np.nan)

    # Create a mask for the heatmap
    mask = np.isnan(highlighted_matrix)

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(Q, annot=True, cmap='viridis', mask=mask, cbar_kws={'label': 'Values'})

    # Highlight the values in the second matrix corresponding to 1s in the binary matrix

    plt.title('Highlighted Values in Second Matrix')
    plt.show()

def interactive_heatmap(n_step, maps, P_k, B_a, Q_a, R_vend, R_acqu, T, fun='step'):
    '''
    function to generate an heatmap with slider to see the evolution of an heatmap
    parameter:
    -   n_step: number of step for the slider
    -   
    '''
    import seaborn as sns
    from ipywidgets import interactive
    


    totP        = sum(sum(P_k))
    totB        = sum(B_a)
    totQ        = sum(Q_a)
    totR_vend   = sum(R_vend)
    totR_acqu   = sum(R_acqu)
    totInc      = T
    totPsik     = sum(sum(P_k))
    #pk and r depend on the timestep so they dont need to be multiplied by T
    n_var = int(totP + T*totB*2 + T*totQ*2 + totR_vend + totR_acqu+T)
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


    def difference(a):
        if a == 0:
            return maps[a]
        else:
            return maps[a]-maps[a-1]
    
    def highlight(a):
        if a == 0:
            m = maps[a]
        else:
            m = maps[a]-maps[a-1]
        highlighted_matrix = m
        highlighted_matrix = np.where(highlighted_matrix!= 0, highlighted_matrix, np.nan)

        # Create a mask for the heatmap
        mask = np.isnan(highlighted_matrix)
        return m,mask

    def step(a):    
        return maps[a]

    f = {'dif':difference,
         'step':step,
         'high':highlight}
    
    def plot(a):

        z = f[fun](a)
        if fun == 'high':
            plt.figure(figsize=(30,26), dpi=80)
            ax=sns.heatmap(z[0], linewidth=0, mask=z[1])
        else:
            plt.figure(figsize=(15,12), dpi=80)
            ax=sns.heatmap(z, linewidth=0)
        ax.hlines([xc_zero, xs_zero, xic_zero, xis_zero, xvend_zero, xacqu_zero, xiinc_zero, psipk_zero, psis_zero, psic_zero] ,*ax.get_xlim()  )
        ax.vlines([xc_zero, xs_zero, xic_zero, xis_zero, xvend_zero ,xacqu_zero ,xiinc_zero, psipk_zero, psis_zero, psic_zero] ,*ax.get_ylim()  )
    return interactive(plot, a=(0,n_step,1))

def read_sample(sample,xml,norm_factor=None,verbose=False, explicit_errors=False):
    '''
    sample  array,  index: variable number,     values: value found for each variable.
    '''

    T, K, P_k, A, B_a, Q_a, R_vend,R_acqu, Epf, Ecf, E_max, E0, Ek_min, g1, g2, g4,etac, etas, PaT, Pa0, Pvend, Pacqu, Pinc, costk =parseXml(xml,norm_factor)
    print(f"Interpratation of sample.\n\nsample in question {sample}\n\n")
    print(f'Found {len(P_k)} generators\nfound {len(B_a)} accumulators')
    totP = sum(sum(P_k))
    totB = sum(B_a)
    totQ = sum(Q_a)
    totR_vend = sum(R_vend)
    totR_acqu = sum(R_acqu)
    xc_zero     = totP   

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

    #pk and r depend on the timestep so they dont need to be multiplied by T
    n_var = int(totP + T*totB*2 + T*totQ*2 + totR_vend + totR_acqu+T)
    xc_zero     = totP   
    xs_zero     = xc_zero + T*totB
    xic_zero    = xs_zero + T*totB
    xis_zero    = xic_zero + T*totQ
    xvend_zero  = xis_zero + T*totQ
    xacqu_zero  = xvend_zero + totR_vend
    xiinc_zero  = xacqu_zero + totR_acqu
    print('Check result\n')
    fulldf =pd.DataFrame()

    if len(P_k)>0:
        E_generated={}
        for k in range(len(P_k)):
            E_generated[f'gen{k}']=[]
            for t in range(T):
                tmp=0
                for i,e in enumerate(sample[xpk_point(t,k,0):xpk_point(t,k,P_k[k,t])]):
                    print(e)

                    tmp+=pow(2,i+g1)*e
                E_generated[f'gen{k}'].append(tmp+Ek_min[k])

        for k,v in E_generated.items():
            fulldf[k]=v

    if len(B_a) > 0:
        E_scarica={}
        E_carica={}
        for a,_ in enumerate(B_a):   
            E_carica[f'c_acc{a}']=[]
            E_scarica[f's_acc{a}']=[]
            for t in range(T):
                tmp=0
                for i,e in enumerate(sample[xcab_point(t,a,0):xcab_point(t,a,B_a[a])]):
                    tmp+=pow(2,i+g2)*e
                E_carica[f'c_acc{a}'].append(tmp)
                
                
                tmp=0
                for i,e in enumerate(sample[xsab_point(t,a,0):xsab_point(t,a,B_a[a])]):
                    tmp+=pow(2,i+g2)*e
                E_scarica[f's_acc{a}'].append(tmp)
        for a,v in E_carica.items():
            fulldf[a]=v
        for a,v in E_scarica.items():
            fulldf[a]=v


    E_va={'E_vend':[],
          'E_acqu':[]}
    for t in range(T):
        tmp=0
        for i,e in enumerate(sample[xrvend_point(t,0):xrvend_point(t,R_vend[t])]):
            tmp+=pow(2,i+g4)*e
        E_va['E_vend'].append(tmp)

        tmp=0
        for i,e in enumerate(sample[xracqu_point(t,0):xracqu_point(t,R_acqu[t])]):
            tmp+=pow(2,i+g4)*e
        E_va['E_acqu'].append(tmp)
    for a,v in E_va.items():
        fulldf[a]=v

    print(f'Check constraints\n')

    # rocs constraint
    for a in range(A): 
        c=0
        for t in range(T):
            sat = E_carica[f'c_acc{a}'][t] ==0 or E_scarica[f's_acc{a}'][t] ==0
            if not sat:
                c-=-1
        print(f"Il vincolo di carica-scarica e' stato infranto {c} volte nell' accumulatore {a}\n")

    # rova constraint
    c=0
    for t in range(T):
        sat = E_va['E_vend'][t] ==0 or E_va['E_acqu'][t] ==0
        if not sat:
            c-=-1
    print(f"Il vincolo di acquisto vendita e' stato infranto {c} volte\n")

    #roinc constraint

    c=0
    einc =[]
    for t in range(T):
        eprod = Epf[t] + sum([E_generated[f'gen{k}'][t] + Ek_min[k] for k in range(K)]) + sum([E_scarica[f's_acc{a}'][t] for a in range(A)])
        econs = Ecf[t] + sum([E_carica[f'c_acc{a}'][t] for a in range(A)])
        einc.append(sample[xiinc_point(t)]*eprod+(1-sample[xiinc_point(t)])*econs)
        sat = (eprod<econs)==sample[xiinc_point(t)]
        if not sat:
            c-=-1

    print(f"Il calcolo dell incentivo e' stato sbagliato {c} volte\n")

    #roac roas constraint
    EaT=[]
    for a in range(A):
        cc=0
        cs=0
        Ea=E0[a]
        for t in range(T):
            Ea += etac[a]*E_carica[f'c_acc{a}'][t]
            Ea -= 1/etas[a]*E_scarica[f's_acc{a}'][t]
            satc = Ea<E_max[a]
            if not satc:
                cc-=-1
            sats = Ea>0
            if not sats:
                cs-=-1
        EaT.append(Ea)
        print(f"Il vincolo di carica massima e' stato infranto {cc} volte\n")
        print(f"Il vincolo di scarica minima e' stato infranto {cs} volte\n")


    #roe0 constraint

    c=0

    for t in range(T):
        eprod = Epf[t] + sum([E_generated[f'gen{k}'][t] + Ek_min[k] for k in range(K)]) + sum([E_scarica[f's_acc{a}'][t] for a in range(A)])
        econs = Ecf[t] + sum([E_carica[f'c_acc{a}'][t] for a in range(A)])
        sat = (eprod - E_va['E_vend'][t] + E_va['E_acqu'][t] - econs) == 0
        if not sat:
            c-=-1

    print(f"Il vincolo di enrgia nulla e' stato infranto {c} volte\n")
    ricavot=[]
    costot=[]
    for t in range(T):
        ricavot.append(E_va['E_vend'][t]*Pvend[t]+Pinc*einc[t])
        costot.append(E_va['E_acqu'][t]*Pacqu[t]+sum([costk[k]*E_generated[f'gen{k}'][t] for k in range(K)]))

    fulldf['costo(t)']=costot
    fulldf['ricavo(t)']=ricavot

    ricavo = sum(ricavot)+ sum([EaT[a]*PaT for a in range(A)])
    costo = sum(costot) +sum([E0[a]*Pa0 for a in range(A)])
    print(f"Il bilancio finale e' pari a:{ricavo-costo}")
    return fulldf

def matrix_to_txt(matrix,file_name):
    '''
    matrix      2D array (or np.array)      matrix to transcribe
    file_name   string                      path and name of the output file (note that the file will be overritten if it already exists)'''
    with open(file_name,'w') as file:
        for row in matrix:
            for col in row:
                file.write(str(col)+' ')
            file.write('\n')

def parseXml(source, norm_factor=None):
    tree = ET.parse(source)
    prosumer = tree.getroot().find('prosumer')
    T = int(prosumer.find('prosumer_header').find('number_of_periods').text)


    g1 = 0
    g2 = 0 
    g4 = 0
    # g1 = int(prosumer.find('prosumer_header').find('number_of_periods').text)
    # g2 = int(prosumer.find('prosumer_header').find('number_of_periods').text)
    # g4 = int(prosumer.find('prosumer_header').find('number_of_periods').text)
    PaT = 0
    Pa0 = 0
    Ecf =[]
    Epf=[]
    Pvend=[]
    Pacqu=[]
    if norm_factor is not None:
        Pinc = float(prosumer.find('prosumer_header/price_incentives').text)/norm_factor        
        for t in prosumer.find('prosumer_time_series').findall('time_block'):
            Ecf.append(-1*float(t.find('quantity_fixed_consumption').text)*norm_factor)
            Epf.append(float(t.find('quantity_fixed_production').text)*norm_factor)
            Pvend.append(float(t.find('price_sell').text)/norm_factor)
            Pacqu.append (float(t.find('price_buy').text)/norm_factor)
    
        generatori=[]
        for gen in prosumer.find('prosumer_units').find('prosumer_units_up').findall('prosumer_unit_up'):
            g={}
            g['id']=gen.find('prosumer_unit_up_header/prosumer_unit_id').text
            g['prod_nom']=float(gen.find('prosumer_unit_up_header/prosumer_up_power_quantity_nominal').text)*norm_factor
            g['prod_min']=float(gen.find('prosumer_unit_up_header/prosumer_up_power_quantity_min').text)*norm_factor
            g['cost']=float(gen.find('prosumer_unit_up_header/prosumer_up_power_price_generation_cost').text)/norm_factor
            g['prod_max']= [float(a.find('quantity_production').text)*norm_factor for a in gen.findall('prosumer_time_series_up/time_block')]
            generatori.append(g)

        accumulatori=[]
        for acc in prosumer.find('prosumer_units').find('prosumer_units_ua').findall('prosumer_unit_ua'):
            a={}
            a['id']=acc.find('prosumer_unit_ua_header/prosumer_unit_id').text
            a['E_max']=float(acc.find('prosumer_unit_ua_header/prosumer_unit_energy_max').text)*norm_factor
            a['E_max_s']=float(acc.find('prosumer_unit_ua_header/prosumer_unit_energy_discharge_max').text)*norm_factor
            a['E_max_c']=float(acc.find('prosumer_unit_ua_header/prosumer_unit_energy_charge_max').text)*norm_factor
            a['E0']=float(acc.find('prosumer_unit_ua_header/prosumer_unit_energy_t0').text)*norm_factor
            a['eta_car']=float(acc.find('prosumer_unit_ua_header/prosumer_unit_performance_charge').text)
            a['eta_sca']=float(acc.find('prosumer_unit_ua_header/prosumer_unit_performance_discharge').text)
            accumulatori.append(a)
    else:
        Pinc = float(prosumer.find('prosumer_header/price_incentives').text)
        for t in prosumer.find('prosumer_time_series').findall('time_block'):
            Ecf.append(float(t.find('quantity_fixed_consumption').text))
            Epf.append(float(t.find('quantity_fixed_production').text))
            Pvend.append(float(t.find('price_sell').text))
            Pacqu.append (float(t.find('price_buy').text))
        
        generatori=[]
        for gen in prosumer.find('prosumer_units').find('prosumer_units_up').findall('prosumer_unit_up'):
            g={}
            g['id']=gen.find('prosumer_unit_up_header/prosumer_unit_id').text
            g['prod_nom']=float(gen.find('prosumer_unit_up_header/prosumer_up_power_quantity_nominal').text)
            g['prod_min']=float(gen.find('prosumer_unit_up_header/prosumer_up_power_quantity_min').text)
            g['cost']=float(gen.find('prosumer_unit_up_header/prosumer_up_power_price_generation_cost').text)
            g['prod_max']= [float(a.find('quantity_production').text) for a in gen.findall('prosumer_time_series_up/time_block')]
            generatori.append(g)

        accumulatori=[]
        for acc in prosumer.find('prosumer_units').find('prosumer_units_ua').findall('prosumer_unit_ua'):
            a={}
            a['id']=acc.find('prosumer_unit_ua_header/prosumer_unit_id').text
            a['E_max']=float(acc.find('prosumer_unit_ua_header/prosumer_unit_energy_max').text)
            a['E_max_s']=float(acc.find('prosumer_unit_ua_header/prosumer_unit_energy_discharge_max').text)
            a['E_max_c']=float(acc.find('prosumer_unit_ua_header/prosumer_unit_energy_charge_max').text)
            a['E0']=float(acc.find('prosumer_unit_ua_header/prosumer_unit_energy_t0').text)
            a['eta_car']=float(acc.find('prosumer_unit_ua_header/prosumer_unit_performance_charge').text)
            a['eta_sca']=float(acc.find('prosumer_unit_ua_header/prosumer_unit_performance_discharge').text)
            accumulatori.append(a)

    K = len(generatori)
    A = len(accumulatori)


    # Eas_max,
    Ek_max = []
    P_k = []
    for g in generatori:
        argv=list(map(int,np.floor(np.log2(np.array(g['prod_max']) - g['prod_min'] +0.0001))-g1+1))
        argv = [a if a > 0 else 0 for a in argv]
        P_k.append(argv)       
        Ek_max.append(np.array(g['prod_max']) - g['prod_min']) 

    P_k=np.array(P_k)
    Ek_max=np.array(Ek_max)

    Eac_max = []
    Eas_max = []
    B_a = []
    for a in accumulatori:
        argv=int(np.floor(np.log2((a['E_max_c'])+0.0001))-g1+1)
        argv = argv if argv > 0 else 0
        B_a.append(argv)      
        Eac_max.append(a['E_max_c'])
        Eas_max.append(a['E_max_s'])



    B_a=np.array(B_a)
    Eac_max=np.array(Eac_max)
    Eas_max=np.array(Eas_max)


    Q_a = []
    for a in accumulatori:
        argv=int(np.floor(np.log2((a['E_max'])+0.0001))-g1+1)
        argv = argv if argv > 0 else 0
        Q_a.append(argv)      

 
    Q_a=np.array(Q_a)

    R_acqu=[]
    for t in range(T):
        argv = -Epf[t]+Ecf[t]+sum([a['E_max_c'] for a in accumulatori])
        argv = argv+0.0001 if argv > 0 else 0.0001
        argv = int(np.floor(np.log2(argv))-g4+1)
        argv = argv if argv > 0 else 1
        R_acqu.append(argv)

    R_vend=[]
    for t in range(T):
        argv = Epf[t]-Ecf[t]+sum([g['prod_max'][t] for g in generatori])+sum([a['E_max_s'] for a in accumulatori])
        argv = argv+0.0001 if argv > 0 else 0.0001
        argv = int(np.floor(np.log2(argv))-g4+1)
        argv = argv if argv > 0 else 0
        R_vend.append(argv)

    E_max =[a['E_max'] for a in accumulatori]
    E0 =[a['E0'] for a in accumulatori]
    etac =[a['eta_car'] for a in accumulatori]
    etas =[a['eta_sca'] for a in accumulatori]

    Ek_min =[g['prod_min'] for g in generatori]
    costk =[g['cost'] for g in generatori]

    return T, K, P_k, A, B_a, Q_a,R_vend,R_acqu, Epf, Ecf, E_max, E0, Ek_min, g1, g2, g4,etac, etas, PaT, Pa0, Pvend, Pacqu, Pinc, costk,Eas_max,Eac_max,Ek_max

def to_up_triang(m):
    rows,col= m.shape
    for a in range(rows):
        for b in range(a):
            m[b,a] += m[a,b]
            m[a,b] = 0
    return m