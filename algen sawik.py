import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))

def normalize(x):
    nmax = np.max(x)
    nmin = np.min(x)
    newx = np.copy(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            newx[i,j] = (newx[i,j] - nmin) / (nmax - nmin)
    return newx

def feedforward(X, w0, w1):
    f1 = nonlin(np.dot(X, w0))
    f2 = nonlin(np.dot(f1, w1))
    f2_error = prediction - f2
    loss = np.mean(np.abs(f2_error))
    return loss

def calculate_fitness(p):
    fitness = []
    for i in p:
        w0 = np.reshape(i[:inputlayer.flatten().size], inputlayer.shape)
        w1 = np.reshape(i[inputlayer.flatten().size:], hiddenlayer.shape)
        fitness.append(feedforward(data, w0, w1))
    return np.array(fitness)
    

def sort_population(p, f):
    for i in range(f.size):
        for j in range(i+1, f.size):
            if f[j] < f[i]:
                temp = f[i]
                f[i] = f[j]
                f[j] = temp
                temp = p[i]
                p[i] = p[j]
                p[j] = temp
    return p, f

def crossover(p, r):
    jml_child = int(r*8)
    parents = p[random.sample(range(8), jml_child)]
    childs = []
    for i in range(jml_child//2):
        child = parents[i*2]
        child2 = parents[i*2+1]
        #PENENTUAN SATU TITIK CROSSOVER
        cridx = random.sample(range(parents.shape[1]), 1)
        '''
        if cridx[0] > cridx[1]:
            temp = cridx[0]
            cridx[0] = cridx[1]
            cridx[1] = temp
        #PROSES CROSSOVER
        temp = child[cridx[0]]
        child[cridx[0] : cridx[1]] = child2[cridx[0] : cridx[1]]
        child2[cridx[0] : cridx[1]] = temp
        '''
        if cridx[0] != 0 and cridx[0] <= len(child)//2:
            temp = child[:cridx[0]]
            child[:cridx[0]] = child2[:cridx[0]]
            child2[:cridx[0]] = temp
        elif cridx[0] != len(child)-1 and cridx[0] > len(child)//2:
            temp = child[cridx[0]:]
            child[cridx[0]:] = child2[cridx[0]:]
            child2[cridx[0]:] = temp
        elif cridx[0] == 0:
            temp = child[:cridx[0]]
            child[:cridx[0]] = child2[:cridx[0]]
            child2[:cridx[0]] = temp
        else:
            temp = child[cridx[0]:]
            child[cridx[0]:] = child2[cridx[0]:]
            child2[cridx[0]:] = temp
        childs.append(child)
        childs.append(child2)
    p = np.append(p, np.array(childs), axis=0)
    return p

def mutation(p, r):
    jml_child = int(r*8)
    parents = p[random.sample(range(8), jml_child)]
    childs = []
    for i in range(jml_child):
        child = parents[i]
        cridx = random.sample(range(parents.shape[1]), np.random.randint(parents.shape[1]))
        for idx in cridx:
            child[idx] = np.random.random()
        childs.append(child)
    p = np.append(p, np.array(childs), axis=0)
    return p

#READ DATA DAN NORMALISASI
data = np.array(pd.read_excel("data.xlsx"))
prediction = np.array(pd.read_excel("prediction.xlsx"))
data = normalize(data)
prediction = normalize(prediction)

#ATUR JUMLAH NEURON
jml_neuron = 15

#INISIASI JARINGAN
inputlayer = 2*np.random.random((data.shape[1], jml_neuron)) - 1
hiddenlayer = 2*np.random.random((jml_neuron, prediction.shape[1])) - 1

#JUMLAH GENERASI MAKSIMAL
max_generation = 100
population = 2*np.random.random((8, inputlayer.flatten().size+hiddenlayer.flatten().size)) - 1
best_fitness = []
for i in range(max_generation):
    fitness = calculate_fitness(population)
    r = 0.5
    while r < 0.25:
        r = np.random.rand()
    population = crossover(population, r)
    mr = 0.25
    population = mutation(population, mr)
    fitness = calculate_fitness(population)
    #SELEKSI POPULASI MENGGUNAKAN RANK SELECTION
    sortedidx = np.argsort(fitness)                 #SORTIR INDEX ARRAY FITNESS
    fitness = fitness[sortedidx]                    #URUTKAN FITNESS KECIL KE BESAR
    best_fitness.append(fitness[0])                 #SIMPAN BEST FITNESS UNTUK GRAFIK
    population = population[sortedidx]              #URUTKAN POPULASI BERDASARKAN FITNESS
    population = population[:8]                     #PANGKAS ARRAY POPULASI/ RANK SELECTION
    fitness = fitness[:8]                           #PANGKAS ARRAY FITNESS

#TESTING BACKPRO, LANGKAH ALGEN BACKPRO
i = population[0]
inputlayer = np.reshape(i[:inputlayer.flatten().size], inputlayer.shape)
hiddenlayer = np.reshape(i[inputlayer.flatten().size:], hiddenlayer.shape)
max_epoh = 100
learning_rate = 1
for i in range(max_epoh):
    for j in range(data.shape[0]):
        #ALUR MAJU BACKPROPAGATION
        X = np.array([data[j]])
        f1 = nonlin(np.dot(X, inputlayer))
        f2 = nonlin(np.dot(f1, hiddenlayer))
        
        #PERHITUNGAN LOSS PREDICTION
        f2_error = [prediction[j]] - f2
        loss = np.mean(np.abs(f2_error))
        if j == 0:
            print("loss:" + str(loss))
        
        #PERHITUNGAN NILAI PENGUBAH BOBOT
        f2_delta = f2_error*nonlin(f2,deriv=True)
        f1_error = f2_delta.dot(hiddenlayer.T)
        f1_delta = f1_error * nonlin(f1,deriv=True)
        
        #PENGUBAHAN BOBOT
        hiddenlayer += learning_rate * f1.T.dot(f2_delta)
        inputlayer += learning_rate * X.T.dot(f1_delta)
#TESTING BOBOT
data = np.array(pd.read_excel("data testing.xlsx"))
data = normalize(data)
X = data
prediction = np.array(pd.read_excel("prediction testing.xlsx"))
prediction = normalize(prediction)
f1 = nonlin(np.dot(X, inputlayer))
f2 = nonlin(np.dot(f1, hiddenlayer))
f2_error = prediction - f2
loss = np.mean(np.abs(f2_error))
akurasi = (1 - loss) * 100
print("Akurasi testing : %.2f" % akurasi)
layers = {"inputlayer": inputlayer, "hiddenlayer": hiddenlayer}
pickle.dump(layers, open('bobot_algen.p', 'wb'))
#MEMBUAT DAN MENAMPILKAN GRAFIK
plt.title("Grafik Fitness")
plt.xlabel("Generasi")
plt.ylabel("Fitness")
plt.plot(best_fitness)