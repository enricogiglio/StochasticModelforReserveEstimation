# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:10:44 2023

@author: user
"""


import pypsa
import os
os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from math import e
import time
import pandas as pd
from itertools import product
from parfor import parfor
from multiprocessing import Process
from multiprocessing import Pool
import time
import sys, traceback
import cProfile
import pstats
import io
import json
from itertools import chain
import gzip
import gc

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)
    
    
###############################################
"""
Input files:
file_path: The Excel file containing the combinations for which the stochastic reserve estimation model is to be run.
network_name: This is the file that contains the characteristics of the network, such as its components and their parameters.
P_ref: This is the reference nominal power of the network components affected by the forecast error. P_ref is the value of the nominal power to which the assumed forecast error refers. 
"""
file_path = 'CombToStudy.xlsx'
comb_to_study = pd.read_excel(file_path, skiprows=[1], index_col=None).iloc[:, 1:]
network_name='network_SA2050'
network=pd.read_csv(network_name)
P_ref=100
################################################

def generate_saw_sequence(tMC, VRD):
    """
    Dynamic ramp effect: it generate a sawtooth sequence for the given time horizon and hourly Variation of Residual Demand (VRD).

    Parameters
    ----------
    tMC : int
        Monte Carlo time horizon.
    DDM : float
        Desired variation demand.

    Returns
    -------
    np.ndarray
        The generated sawtooth sequence scaled by the ramp value.
    """
    saw_length = 15
    repetitions = tMC // saw_length
    remainder = tMC % saw_length
    
    single_sequence = np.linspace(-1, 1, saw_length)
    
    saw_sequence = np.tile(single_sequence, repetitions)
    
    if remainder > 0:
        saw_sequence = np.concatenate((saw_sequence, np.linspace(-1, 1, remainder)))
    
    ramp_value = VRD/8
    return ramp_value * saw_sequence
####################################################################


def tripping(yr,network):
    """
    Evaluate tripping events for network components over the given years.

    Parameters
    ----------
    yr : int
        Number of years to simulate.
    network : DataFrame
        Network data containing information about generators and their characteristics.

    Returns
    -------
    dict
        A dictionary with indices of tripping events for each element.
    """
    
    #Montecarlo Time horizon
    tMC=yr*365*24*60
    #Trip imbalance
    
    #element that can trip
    elem_trip=network['prob_trip']>0
    #number of elements that can trip
    n_elem_trip=(elem_trip*network['n_mod']).sum()
    yr_limit=0.05
    tMC_par=int(yr_limit*365*24*60)
    
    repeated_trip=np.repeat(network['prob_trip'][elem_trip].values, network['n_mod'][elem_trip].values)
    trip_val_matrix=np.transpose(np.tile(repeated_trip, (tMC_par,1)))
    trip_index = {i: [] for i in range(n_elem_trip+1)}
    for yr_indx in range(int(tMC/tMC_par)):
        #print(str(yr_indx)+' su '+ str(int(tMC/tMC_par)))
        tMC_past=yr_indx*tMC_par
        trip_prob=np.random.rand(n_elem_trip,tMC_par)
        trip_binary=trip_prob<trip_val_matrix
        #trip_binary=trip_binary.ravel()
        trip_index_current=np.where(trip_binary>0)
        
        for idx0, idx1 in zip(trip_index_current[0], trip_index_current[1]):
            trip_index[idx0].append(idx1+tMC_past)
    
    return trip_index

def reserve_dimensioning(yr,VRD, network,quantile_fcr_perc, quantile_afrr_perc, quantile_mfrr_perc, quantile_rr_perc, sigma_network_selected, ramp_val, tripping_event):
    """
    Calculate reserve requirements (FCR, aFRR, mFRR, RR) based on specified quantiles using a stochastic approach.

    Parameters
    ----------
    yr : int
        The number of years for which to perform the Monte Carlo simulation.
    VRD : int
        The hourly Variation of the Residual Demand (VRD) for which to perform the Monte Carlo simulation.
    network : DataFrame
        The network object containing information about the power system.
    quantile_fcr_perc : float
        The desired quantile for FCR (e.g., 0.997 for 99.7th percentile).
    quantile_afrr_perc : float
        The desired quantile for aFRR (e.g., 0.95 for 95th percentile).
    quantile_mfrr_perc : float
        The desired quantile for mFRR (e.g., 0.99 for 99th percentile).
    quantile_rr_perc : float
        The desired quantile for RR (e.g., 0.997 for 99.7th percentile).
    sigma_network_selected : Series
        The standard deviation of the vRES tecs and loads of the network selected for evaluation, per each of the combinations under study.
    ramp_val : Series
        The ramp values for the network characterising the ramp dynamics.
    tripping_event : dict
        The tripping events for the network components.

    Returns
    -------
    tuple
        A tuple containing the calculated reserves: (FCR, aFRR, mFRR, RR, quartile_trip, EA_trip).
    """
    
    
    gc.collect()
    
    #Montecarlo Time horizon
    tMC=yr*365*24*60
    
    
    #VDR
    ramp_sequence = generate_saw_sequence(tMC, VRD)

    ################################################################################################
    #Trip Imbalance: Evaluation of the maximum number of elements per each network component that can trip
   
    #element that can trip
    elem_trip=network['prob_trip']>0
    #number of elements that can trip
    n_elem_trip=(elem_trip*network['n_mod']).sum()
    ################################################################################################
    #Application of Monte Carlo method for the evaluation of tripping events
    trip_imb_t=np.zeros((tMC))
    
    if n_elem_trip>0:
        #matrix with uniform distribution
        
        #If the probability is lower than the threshold associated with the
        #individual generator (specific trip probability),
        #the generator is considered to have tripped (=1).
        
        p_nom=network['P_nom-avlb'][elem_trip*(network['n_mod']>0)]
        n_mod=network['n_mod'][elem_trip*(network['n_mod']>0)]
        
        trip_element_status=[]
        elem_trip_online=network['n_mod'][elem_trip].values
        elem_trip_offline= network['n_mod_max'][elem_trip].values-network['n_mod'][elem_trip].values
        for a, b in zip(elem_trip_online, elem_trip_offline):
            trip_element_status.extend([1] * a)
            trip_element_status.extend([0] * b)
        
        online_elem_index=np.where(np.array(trip_element_status) == 1)[0]
        trip_index=[tripping_event[i] for i in online_elem_index]
        #t_trip is the time a generator remains off after being tripped.
        t_trip=30
        g_start=0
        
        p_nom.loc[p_nom==0]=100
        for g in range(len(p_nom)):
            g_end=g_start+n_mod.iloc[g]
            trip_index_list=list(chain.from_iterable(trip_index[g_start:g_end]))
            if len(trip_index_list)>0:
                shift_vec = np.tile(list(range(0,t_trip)), len(trip_index_list))
                all_idxs_tripped=np.repeat(trip_index_list, t_trip) + shift_vec
                
                vector_counts = np.bincount(all_idxs_tripped, minlength=tMC)[0:tMC]
                trip_imb_t += vector_counts * p_nom.iloc[g]
            g_start=g_end
    
    quartile_trip=np.percentile(trip_imb_t, 99.7)
    EA_trip=trip_imb_t.sum()/yr/60
    
    i_proccess=10
    RAM_avlb=30*1024**3/i_proccess
    n_elem_lim = int(RAM_avlb/8/tMC) 
    t_frr = 15
    n_quart = int(tMC / t_frr)
    
    FCR = np.array([])
    aFRR = np.array([])
    mFRR = np.array([])
    RR = np.array([])
    
    ###################################################################
    #Convolution between the different sources of unbalance (tripping events, load and RES generation variation, ramp dynamics)
    for i in range(0, len(sigma_network_selected), n_elem_lim):
        end_i=min(i + n_elem_lim, len(sigma_network_selected))
        current_block = sigma_network_selected.iloc[i:end_i]
        current_ramp_val=ramp_val.iloc[i:end_i].values
        
        network_t = np.random.normal(np.zeros((len(current_block), 1)), current_block.values.reshape((len(current_block), 1)), (len(current_block), tMC))
        tot_imb_t = network_t + trip_imb_t + ramp_sequence[0:tMC]
        
        #####################################################################
        
        FCR_block = np.percentile(tot_imb_t, quantile_fcr_perc * 100 * np.ones(len(current_block)),axis=1)[0]
    
        mean_imb_t = tot_imb_t.reshape(-1, t_frr).sum(1) / t_frr
        mean_imb_t = mean_imb_t.reshape(len(current_block), int(n_quart))
        tot_imb_frr_t = tot_imb_t - np.repeat(mean_imb_t, t_frr, axis=1)
    
        aFRR_block = np.percentile(tot_imb_frr_t, quantile_afrr_perc * 100 * np.ones(len(current_block)),axis=1)[0]
        mFRR_block = -aFRR_block + np.percentile(tot_imb_frr_t, quantile_mfrr_perc * 100 * np.ones(len(current_block)),axis=1)[0]
        RR_block = -aFRR_block - mFRR_block + np.percentile(tot_imb_frr_t, quantile_rr_perc * 100 * np.ones(len(current_block)),axis=1)[0]
    
        FCR = np.concatenate((FCR, FCR_block))
        aFRR = np.concatenate((aFRR, aFRR_block))
        mFRR = np.concatenate((mFRR, mFRR_block))
        RR = np.concatenate((RR, RR_block))

    return (FCR,aFRR,mFRR,RR, quartile_trip*np.ones(len(FCR)), EA_trip*np.ones(len(FCR)))


def reserve_function_builder(network, yr, quantile_fcr_perc, quantile_afrr_perc, quantile_mfrr_perc, quantile_rr_perc):
    """
    Build and evaluate reserve allocation strategies based on different generator statuses and available/demanded power levels.

    This function generates and evaluates reserve allocation strategies for the given network and specified parameters.
    The strategies are evaluated at various points within the specified domain defined by generator statuses and power levels.

    Parameters
    ----------
    network : DataFrame
        Network data containing information about generators and their characteristics.
    yr : int
        Number of years for Monte Carlo simulation.
    quantile_fcr_perc : float
        The desired quantile for FCR (e.g., 0.997 for 99.7th percentile).
    quantile_afrr_perc : float
        The desired quantile for aFRR (e.g., 0.95 for 95th percentile).
    quantile_mfrr_perc : float
        The desired quantile for mFRR (e.g., 0.99 for 99th percentile).
    quantile_rr_perc : float
        The desired quantile for RR (e.g., 0.997 for 99.7th percentile).

    Returns
    -------
    f_reserve : DataFrame
        Dataframe containing reserve allocation strategies for each combination of generator statuses and available power levels.

    """
    
    elem_committable_name=network.Name[network['Committable']>0]
    elem_p_var_name=network.Name[network['resol_explor_P_avlb']>0]
    elem_p_var=network['resol_explor_P_avlb']>0
    
    combinations_network = comb_to_study.copy()  

    mod_columns = [col for col in comb_to_study.columns if '-mod' in col]
    
    # Get unique values for each column that contains '-mod'
    range_tripping = {col: comb_to_study[col].unique() for col in mod_columns}      
    
    f_reserve= pd.DataFrame(np.zeros((len(combinations_network),6)), columns=['f_FCR','f_aFRR','f_mFRR','f_RR', 'quartile_trip', 'EA_trip'])
    
    n_mod_p_var_commit=combinations_network[(elem_committable_name[elem_committable_name.isin(elem_p_var_name)]+'-mod').to_list()]
    
    sigma_network=sigma_network_eval(n_mod_p_var_commit, network, combinations_network.loc[:,list(elem_p_var_name)])
    
    trip_comb_indx = find_new_combination_index(combinations_network, len(range_tripping))
    
    file_path = 'TE'+str(yr)+'yr_'+network_name+'.json'
    ##################################################################################
    #Create a matrix for the tripping events being considered
    tripping_event=tripping(yr,network)
    with open(file_path, 'w') as json_file:
        json.dump(tripping_event, json_file, cls=NumpyEncoder)

    
    network.loc[:,'n_mod_max']=network.loc[:,'n_mod'].copy()
    
    print('tripping matrix done')
    ##################################################################################
    
    for i in range(len(trip_comb_indx)):
        idx=trip_comb_indx[i]
        if i==len(trip_comb_indx)-1:
            idx_end=len(combinations_network)
        else: 
            idx_end=trip_comb_indx[i+1]
        f_reserve.iloc[idx:idx_end,:]=evaluate_reserve(idx, trip_comb_indx, tripping_event, combinations_network, yr, network, quantile_fcr_perc, quantile_afrr_perc, quantile_mfrr_perc, quantile_rr_perc, sigma_network.iloc[idx:idx_end], combinations_network.Load.iloc[idx:idx_end])
    
   
    for i in elem_committable_name[elem_committable_name.isin(elem_p_var_name)].tolist():
        combinations_network[i]=combinations_network[i+'-mod']*combinations_network[i]
    
    return f_reserve, combinations_network


def evaluate_reserve_single_case(args):
    """
    Wrapper function for calling the evaluate_reserve function with the provided arguments.
    
    Parameters
    ----------
    args : tuple
        A tuple containing the arguments needed for evaluate_reserve function.
    
    Returns
    -------
    DataFrame
        The reserve allocation strategies for the given combination.
    """
    i, trip_comb_indx, tripping_event, combinations_network, yr, network, quantile_fcr_perc, quantile_afrr_perc, quantile_mfrr_perc, quantile_rr_perc, sigma_network_selected, ramp_val = args
    return evaluate_reserve(i, trip_comb_indx, tripping_event, combinations_network, yr, network, quantile_fcr_perc, quantile_afrr_perc, quantile_mfrr_perc, quantile_rr_perc, sigma_network_selected, ramp_val)


def sigma_network_eval(combinations_n_mod, network, p_var):
    """
    Evaluate the standard deviation of vRES tecs and the load of the network elements based on their availability and combination.
    
    Parameters
    ----------
    combinations_n_mod : DataFrame
        DataFrame containing combinations of generator statuses.
    network : DataFrame
        Network data containing information about generators and their characteristics.
    p_var : DataFrame
        DataFrame containing information about power variability.
    
    Returns
    -------
    Series
        The evaluated standard deviation for the network combinations.
    """
    #Delta_f
    #element which cause frequency imbalance
    elem_network=network['std_dev']>0
    n_elem_network=(elem_network*network['n_mod']).sum()
    
    elem_n_mod_fix=set(network.Name[elem_network]) - set(combinations_n_mod.columns.str.replace('-mod', ''))
    
    n_mod_fix=network.loc[network.Name.isin(elem_n_mod_fix), 'n_mod']
    n_mod_fix_matrix=np.tile(n_mod_fix.values, (len(combinations_n_mod), 1))
    
    n_mod=combinations_n_mod.copy()
    n_mod.columns=list(combinations_n_mod.columns.str.replace('-mod', ''))
    n_mod[list(elem_n_mod_fix)]=n_mod_fix_matrix
    
    std_dev_matrix_network=np.tile(network['std_dev'][elem_network].values, (len(combinations_n_mod), 1))
    
    intersection_indx=p_var.columns.intersection(n_mod.columns)
    
    std_dev_matrix_tot=std_dev_matrix_network**2*p_var.loc[:,intersection_indx]*n_mod*P_ref
    std_dev_vect=np.sqrt(((std_dev_matrix_tot)).sum(1))
    
    return std_dev_vect

def find_new_combination_index(dataframe, len_combination):
    """
    Find indices of new combinations in the dataframe based on the first few columns.

    Parameters
    ----------
    dataframe : DataFrame
        DataFrame containing combinations of parameters.
    len_combination : int
        The number of columns to consider for finding new combinations.

    Returns
    -------
    list
        A list of indices representing the start of new combinations.
    """
    first_four_columns = dataframe.iloc[:, :len_combination]
    indices = first_four_columns.apply(lambda x: x != x.shift(1)).any(axis=1)
    change_indices = indices.loc[indices].index.tolist()
    return change_indices

def evaluate_reserve_wrapper(args):
    """
    Wrapper function to call evaluate_reserve with the provided arguments.
    
    Parameters
    ----------
    args : tuple
        A tuple containing the arguments needed for evaluate_reserve function.
    
    Returns
    -------
    tuple
        The result of the evaluate_reserve function.
    """
    return evaluate_reserve(*args)

def evaluate_reserve(i, trip_comb_indx, tripping_event, combinations_network, yr, network, quantile_fcr_perc, quantile_afrr_perc, quantile_mfrr_perc, quantile_rr_perc, sigma_network_selected, ramp_val):
    """
    Evaluate reserve allocation strategies for a specific combination of parameters
    that affect its sizing.
    
    Parameters
    ----------
    i : int
        Index representing the combination.
    combinations_network : DataFrame
        DataFrame containing combinations of generator statuses and available power levels.
    yr : int
        Number of years for Monte Carlo simulation.
    network : DataFrame
        Network data containing information about generators and their characteristics.
    quantile_fcr_perc : float
        The desired quantile for FCR (e.g., 0.997 for 99.7th percentile).
    quantile_afrr_perc : float
        The desired quantile for aFRR (e.g., 0.95 for 95th percentile).
    quantile_mfrr_perc : float
        The desired quantile for mFRR (e.g., 0.99 for 99th percentile).
    quantile_rr_perc : float
        The desired quantile for RR (e.g., 0.997 for 99.7th percentile).
    
    Returns
    -------
    tuple
        A tuple containing the reserve allocation strategies for the given combination.
    """
    
    elem_committable_name = network.Name[network['Committable'] > 0]
    elem_p_var_name = network.Name[network['resol_explor_P_avlb'] > 0]
    elem_committable = network['Committable'] > 0
    elem_p_var = network['resol_explor_P_avlb'] > 0

    network_actual = network.copy()
    network_actual.loc[elem_committable, 'n_mod'] = np.array(combinations_network[elem_committable_name+'-mod'].iloc[i], dtype=int)
    network_actual.loc[elem_p_var, 'P_nom-avlb'] = np.array(combinations_network[elem_p_var_name].iloc[i])
    
    tMC=yr*365*24*60
    
    result_fcr, result_afrr, result_mfrr, result_rr,  quartile_trip, EA_trip = reserve_dimensioning(yr, network_actual, quantile_fcr_perc, quantile_afrr_perc, quantile_mfrr_perc, quantile_rr_perc, sigma_network_selected, ramp_val, tripping_event)
    
    f_reserve_current = pd.DataFrame([result_fcr, result_afrr, result_mfrr, result_rr,  quartile_trip, EA_trip]).T
    f_reserve_current.columns = ['f_FCR', 'f_aFRR', 'f_mFRR', 'f_RR', 'quartile_trip', 'EA_trip']
    
    return f_reserve_current


#######################################################################################à
#Input parameters for case study characterisation
yr_MC=192
VRD=0
FCR_perc=0.997
aFRR_perc= 0.95
FRR_perc= 0.99
RR_perc=0.997
#######################################################################################à
#Run of the stochastic model for the reserve estimation
(f_reserve, combinations_network)=reserve_function_builder(network, VRD, yr_MC, FCR_perc, aFRR_perc, FRR_perc, RR_perc)

