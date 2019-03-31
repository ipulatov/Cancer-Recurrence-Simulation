# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:57:05 2019

@author: David
"""

import time
import numpy as np
import multiprocessing
from multiprocessing import Process, freeze_support
import os
import matplotlib.pyplot as plt
import datetime
import random
from numba import jit
from numba import vectorize, cuda, jit, njit, prange, float64, float32, int64
from numba.numpy_support import from_dtype
import numba
import pandas as pd

equilibrium_size = 242853
starting_size = equilibrium_size
number_of_tumors = 100
time_interval = 50


trial_folder = ('/Users/David/Documents/isaac_trials/' + str(datetime.date.today()) + str(time.time()))
os.mkdir(trial_folder)


@jit(nopython=True)
def make_array(number_of_rows, row_size, starting_size):
    q = np.zeros((number_of_rows,row_size))
    q[:,0]=starting_size
    return(q)

delta_t = 1

@jit(nopython=True)
def row_size(array):
    return(array.shape[1])
@jit(nopython=True)
def number_of_rows(array):
    return(array.shape[0])

q = make_array(number_of_tumors,time_interval,starting_size)

@jit(nopython=True)   
def foo(array):
    
    global delta_t
    result = np.zeros(array.size).reshape(1,array.shape[1])
    result[:] = array[:]
    shedding_row = np.zeros(array.size).reshape(1,array.shape[1])
    birth_row = np.zeros(array.size).reshape(1,array.shape[1])
    for i in range((array.shape[0])):
        for j in range((array.shape[1])-1):
            if  result[i,j] !=0:

                shedding = 0
                if result[i,j]!=1:
                    shedding_param = delta_t*( 1.22 * 10**-4 ) * (result[i,j])**.91
                                    
                    shedding = (np.random.poisson( (shedding_param), 1))[0]
                
                birth_param =(delta_t) * (4.02) * (result[i,j] ) ** .81
                birth = (np.random.poisson( (birth_param), 1))[0]

                #This is the regular case
                #death_param =(delta_t) * (20.15) * (result[i,j] ) ** .68
                
                #This is case 1
                #death_param =(delta_t) * (4.02) * (result[i,j] ) ** .68
                
                #This is case 2
                #death_param =(delta_t) * (5.42) * (result[i,j] ) ** .68
                
                #This is case 3
                #death_param =(delta_t) * (7.32) * (result[i,j] ) ** .68
                
                #This is case 4
                #death_param =(delta_t) * (9.87) * (result[i,j] ) ** .68
                
                death_param = 0
                death = (np.random.poisson( (death_param), 1))[0]    
                

                result[i,j+1] = result[i,j] - shedding + birth - death
                if result[i,j+1]<0:
                    result[i,j+1] = 0
                shedding_row[i,j+1] = shedding
                birth_row[i,j+1] = birth
                
            if result[i,j] == 0:
                result[i,j] = result[i,j]

    return(result, shedding_row)


@jit(nopython=True)   
def foo_two(array):

    result = np.zeros(array.size).reshape(array.shape[0],array.shape[1])
    result_two = np.zeros(array.size).reshape(array.shape[0],array.shape[1])       
    i = 0

    while i != (result.shape[0]):

        fill_in_row=  0*np.arange(1 * result.shape[1]).reshape(1, result.shape[1])
        fill_in_row[0] = array[i]
        result[i], shedding_row = foo(fill_in_row)
        result_two[i] = shedding_row
        i+=1            
    
    return(result, result_two)

@jit(nopython=True)    
def foo_three(array):
    
    global delta_t
    array_sum = np.sum(array, axis = 0)
    
    array_sum = array_sum.reshape(1,array_sum.size)

    result = np.zeros(array_sum.size).reshape(1,array_sum.size)

    for i in range((result.shape[0])):
        for j in range((result.shape[1])):
            
            shed_death_param = 1428.6
            shed_metastasis_param = 1.4286
            
            shed_death_param = 6.094
            shed_metastasis_param = 6.094 * 10**(-3)
            
            prob_nothing_happens = (np.exp((-1) * delta_t * (shed_death_param+shed_metastasis_param)))
            prob_something_happens = 1 - prob_nothing_happens
            times_something_happens = np.random.binomial( (int(array_sum[i,j])), prob_something_happens )

            prob_metastasis_happens = (shed_metastasis_param/shed_death_param)
            times_metastasis_happens = np.random.binomial( times_something_happens, prob_metastasis_happens )

            result[i,j] = int(times_metastasis_happens)
            number_to_add = (int(array_sum[i,j])) - (int(times_something_happens))            
            
            if j < row_size(array_sum) - 1:
            	(array_sum[i,j+1]) += number_to_add

    return(result)

@jit(nopython=True)    
def foo_four(array):
    result = np.zeros(array.size).reshape(1,array.size)
    for i in range((result.shape[0])):
        for j in range((result.shape[1])):
            if int(array[i,j])!= 0:
                for q in range(int(array[i,j])):
                     addition = np.zeros((1,result.shape[1]))
                     addition[0][j] = 1
                     result = np.concatenate((result, addition), axis=0)
    if result.shape[0]!=1:
        result = result[1:]
    return(result)

@jit(nopython=True) 
def the_process(array):

    array, master_shedding_array = (foo_two(array))
    
    
    master_metastasis_array = foo_three(master_shedding_array)

    new_array = (foo_four(master_metastasis_array))

    return(array,new_array)

 
def the_bigger_process(array):
    big_array = make_array(1,row_size(array),0)
    big_metastasis_array = make_array(1,row_size(array),0)
    counter =0
    i = 0

    #while counter < row_size(array)-1:
    while counter < row_size(array)-1:
  
        updated_array,metastasis_array = the_process(array)     
        big_array = np.concatenate((big_array, updated_array), axis=0)
        
        if sum( metastasis_array[0] ) != 0:
            big_metastasis_array = np.concatenate((big_metastasis_array, metastasis_array), axis=0)
        
        i+=1
           
        third_big_metastasis_array = big_metastasis_array[np.where(big_metastasis_array[:,i] == 1)]
#        if i ==row_size(array):
#            continue
        
        array = third_big_metastasis_array

        counter+=1
      
    big_array = big_array[1:]
    big_array = np.concatenate((big_array, third_big_metastasis_array), axis=0)

    big_metastasis_array = big_metastasis_array[1:]
    
    np.savetxt(trial_folder +'_Big_Array_CSV.csv', big_array, delimiter = ",")
    np.savetxt(trial_folder +'_Big_Metastasis_Array.csv', big_metastasis_array, delimiter = ",")
    df = pd.DataFrame(big_array)
    df.to_excel(trial_folder +'_Big_Array_Excel.xlsx', index=False)

    return(big_array,big_metastasis_array)   

start = time.time()
something, big_metastasis_array = the_bigger_process(q)
end = time.time()


print("something is\n",something)
print("big_metastasis_array is\n",big_metastasis_array)
print(end-start)


import pandas as pd


def surgery():
    global trial_folder
    df = pd.read_excel(trial_folder +'_Big_Array_Excel.xlsx', sheet_name='Sheet1')
    something = df.values
    last_column = (something[:,-1])
    filtered_last_column = last_column[last_column[:] <equilibrium_size]
    filtered_last_column = filtered_last_column[filtered_last_column[:] > 0]
    print("filtered last column is")
    print(filtered_last_column)

    trial_folder = ('/Users/David/Documents/isaac_trials/' + str(datetime.date.today()) + str(time.time()))
    q_anon = make_array(number_of_rows(filtered_last_column),time_interval,0)
    q_anon[:,0]=filtered_last_column

    start = time.time()
    something, big_metastasis_array = the_bigger_process(q_anon)
    end = time.time()
    print("something is\n",something)
    print("big_metastasis_array is\n",big_metastasis_array)
    print(end-start)
    
surgery()