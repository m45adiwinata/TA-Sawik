# -*- coding: utf-8 -*-
"""
Created on Thu May 30 06:37:14 2019

@author: tatas
"""

import numpy as np
from Backpro import run
from Backpro2 import testing
from openpyxl import load_workbook
from numpy import random

def genetic_to_real(x):
    return x * 0.1 + -0.5

def neural_networknya(member, member2):
    akurasi = run(member, member2)
    return akurasi

def fitness(member, member2):
    num_member = member.shape[0]
    member_baru = np.copy(member)
    akurasi = np.empty(num_member)
    for i in range(num_member):
       akurasi[i] = neural_networknya(member_baru[i, :, :], member2)
       print(akurasi[i])
    temp_sum = np.sum(akurasi)
    for i in range(num_member):
        akurasi[i] = akurasi[i] / temp_sum
    for i in range(num_member):
        if i > 0:
            akurasi[i] = akurasi[i] + akurasi[i-1]
    return akurasi

def selection_parent(member, member2):
    member_fitnes = fitness(member, member2)
    #Roullete Wheel Here
    max_iteration = int(population * crossover_rate * 2)
    parent = np.empty(max_iteration)
    for i in range(max_iteration):
        point_selected = np.random.uniform(0, 1)
        j = 0
        while(j < member_fitnes.shape[0]):
            if(j == 0 and member_fitnes[j] >= point_selected and member_fitnes[j+1] >= point_selected ):
                parent[i] = j
                break
            elif(member_fitnes[j] <= point_selected and member_fitnes[j+1] > point_selected and parent[i-1] != j):
                parent[i] = j
                break
            j += 1
    
    parent = parent.astype(int)
    return parent

def crossover(parent, member):
    num_child = int(len(parent) / 2)
    child = np.empty([num_child, weight_row, weight_column], dtype=int)
#   half_weight_row itu int, pastiin tu int
    k = 0
    for i in range(num_child):
        index = np.random.randint(weight_row)
        child[i, 0:index, :] = member[parent[k], 0:index, :]
        k = k + 1
        child[i, index:, :] = member[parent[k], index:, :]
        k = k + 1
    return child

def mutation(child):
    for i in range(child.shape[0]):
        index_first_row = np.random.randint(weight_row)
        index_first_column = np.random.randint(weight_column)
        
        index_second_row = np.random.randint(weight_row)
        index_second_column = np.random.randint(weight_column)
        
        first_value = child[i, index_first_row, index_first_column]
        second_value = child[i, index_second_row, index_second_column]
        
        child[i, index_first_row, index_first_column] = second_value
        child[i, index_second_row, index_second_column] = first_value
    return child

def selection_death(member, member2, child):
    all_member = np.concatenate((member, child), axis=0)
    member_fitnes = fitness(all_member, member2)
    new_member = np.empty([population,weight_row, weight_column], dtype=int)
    for i in range(population):
        temp_value = 0
        for j in range(len(all_member)):
            if(member_fitnes[j] > temp_value):
              temp_value = member_fitnes[j]
              temp_index = j
        
        new_member[i] = all_member[temp_index]
        all_member = np.delete(all_member, temp_index, 0) 
    return new_member

def ranking_the_best(member, member2):
    member_fitnes = fitness(member, member2)
    temp_value = 0
    for j in range(len(member)):
        if(member_fitnes[j] > temp_value):
            temp_value = member_fitnes[j]
            temp_index = j
            break
        
    return member[temp_index]

population = 5
weight_column = 2
weight_row = 8
generation = 100
mutation_rate = 0.1
crossover_rate = 0.2

#HARUSNYA DATA TRAINING, NANTI DIBAWAH BARU JE DATA TESTING

inputs = 8
hidden = 2

member = random.rand(population, weight_row, weight_column) - 0.5
member2 =  random.rand(2, 1) - 0.5

for i in range(generation):
    print "generation ", i+1
    parent = selection_parent(member, member2)
    child = crossover(parent, member)
    print(child)
    child = mutation(child)
    member = selection_death(member, member2, child)
    
member_the_best = ranking_the_best(member, member2)

#DISINI BARU JE BIKIN DATA TESTING, DI TESTING
#INISIASI CLASSNYA PAKE VARIABLE member_the_bes

print("bobot 1", member_the_best)
print("bobot 2", member2)

akurasi = testing(np.matrix(member_the_best), member2)
print("akurasi terbaik ",akurasi)
