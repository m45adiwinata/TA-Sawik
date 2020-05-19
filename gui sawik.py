import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Frame, Label, Entry, StringVar, Canvas, Scrollbar, Button, OptionMenu, filedialog, messagebox
import random
       
root = Tk()
 
class Main():
    def __init__(self, master, title):
        self.master = master
        self.master.title(title)
        self.mainFrame()
        self.data = np.array(pd.read_excel('data normalisasi.xlsx'))
        self.prediction = np.array(pd.read_excel('prediction normalisasi.xlsx'))
    
    def mainFrame(self):
        frLeft = Frame(self.master)
        frLeft.grid(row=0, column=0, padx=5)
        frRight = Frame(self.master)
        frRight.grid(row=0, column=1, padx=5)
        lblNeuron = Label(frLeft, text="Neuron")
        lblNeuron.grid(row=0, column=0, sticky="W")
        self.neuron = StringVar(value="")
        entryNeuron = Entry(frLeft, textvariable=self.neuron)
        entryNeuron.grid(row=0, column=1)
        lblBulan = Label(frLeft, text="Bulan")
        lblBulan.grid(row=1, column=0, sticky="W")
        self.bulans = [
                "Januari", "Februari", "Maret", "April",
                "Mei", "Juni", "Juli", "Agustus",
                "September", "Oktober", "November", "Desember"
                ]
        self.bulan = StringVar(value=self.bulans[0])
        optBulan = OptionMenu(frLeft, self.bulan, *self.bulans)
        optBulan.grid(row=1, column=1, sticky="W")
        lblMetode = Label(frLeft, text="Metode")
        lblMetode.grid(row=2, column=0, sticky="W")
        self.metode = StringVar(value="Backpro")
        optMetode = OptionMenu(frLeft, self.metode, "Backpro", "Backpro-Algen")
        optMetode.grid(row=2, column=1, sticky="W")
        btnRun = Button(frLeft, text="Run", command=self.run)
        btnRun.grid(row=3, column=0, columnspan=2)
        lblHasil = Label(frRight, text="Hasil")
        lblHasil.grid(row=0, column=0, sticky="W")
        self.hasil = StringVar(value="")
        entryHasil = Entry(frRight, textvariable=self.hasil)
        entryHasil.grid(row=0, column=1)
        lblAkurasi = Label(frRight, text="Akurasi")
        lblAkurasi.grid(row=1, column=0, sticky="W")
        self.akurasi = StringVar(value="")
        entryAkurasi = Entry(frRight, textvariable=self.akurasi)
        entryAkurasi.grid(row=1, column=1)
        lblGrafik = Label(frRight, text="Grafik : ")
        lblGrafik.grid(row=2, column=0, sticky="W")
        self.frGrafik = Frame(self.master)
        self.frGrafik.grid(row=1, column=0, columnspan=10)
        
    def nonlin(self, x,deriv=False):
    	if(deriv==True):
    	    return x*(1-x)
    
    	return 1/(1+np.exp(-x))
    
    def normalize(self, x):
        nmax = np.max(x)
        nmin = np.min(x)
        newx = np.copy(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                newx[i,j] = (newx[i,j] - nmin) / (nmax - nmin)
        return newx
    
    def feedforward(self, X, w0, w1):
        f1 = self.nonlin(np.dot(X, w0))
        f2 = self.nonlin(np.dot(f1, w1))
        f2_error = self.prediction - f2
        loss = np.mean(np.abs(f2_error))
        return loss
    
    def calculate_fitness(self, p, inputlayer, hiddenlayer, data):
        fitness = []
        for i in p:
            w0 = np.reshape(i[:inputlayer.flatten().size], inputlayer.shape)
            w1 = np.reshape(i[inputlayer.flatten().size:], hiddenlayer.shape)
            fitness.append(self.feedforward(data, w0, w1))
        return np.array(fitness)
        
    
    def sort_population(self, p, f):
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
    
    def crossover(self, p, r):
        jml_child = int(r*8)
        parents = p[random.sample(range(8), jml_child)]
        childs = []
        for i in range(jml_child//2):
            child = parents[i*2]
            child2 = parents[i*2+1]
            #PENENTUAN DUA TITIK CROSSOVER
            cridx = random.sample(range(parents.shape[1]), 1)
            '''
            if cridx[0] > cridx[1]:
                temp = cridx[0]
                cridx[0] = cridx[1]
                cridx[1] = temp
            '''
            #PROSES CROSSOVER
            '''
            temp = child[cridx[0] : cridx[1]]
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
    
    def mutation(self, p, r):
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
    
    def run(self):
        print(self.metode.get())
        print(self.neuron.get())
        if(int(self.neuron.get()) > 0):
            print(self.data.shape)
            inputlayer = 2*np.random.random((self.data.shape[1], int(self.neuron.get()))) - 1
            print(inputlayer.shape)
            hiddenlayer = 2*np.random.random((int(self.neuron.get()), self.prediction.shape[1])) - 1
            if(self.metode.get() == "Backpro"):
                max_epoh = 100
                learning_rate = 1
                losses = []
                for i in range(max_epoh):
                    for j in range(self.data.shape[0]):
                        #ALUR MAJU BACKPROPAGATION
                        X = np.array([self.data[j]])
                        f1 = self.nonlin(np.dot(X, inputlayer))
                        f2 = self.nonlin(np.dot(f1, hiddenlayer))
                        
                        #PERHITUNGAN LOSS PREDICTION
                        f2_error = [self.prediction[j]] - f2
                        loss = np.mean(np.abs(f2_error))
                        if j == 0:
                            print("loss:" + str(loss))
                            losses.append(loss)
                        
                        #PERHITUNGAN NILAI PENGUBAH BOBOT
                        f2_delta = f2_error*self.nonlin(f2,deriv=True)
                        f1_error = f2_delta.dot(hiddenlayer.T)
                        f1_delta = f1_error * self.nonlin(f1,deriv=True)
                        
                        #PENGUBAHAN BOBOT
                        hiddenlayer += learning_rate * f1.T.dot(f2_delta)
                        inputlayer += learning_rate * X.T.dot(f1_delta)
                #data = np.array([self.data[-1]])
                data = np.array(pd.read_excel("data testing.xlsx"))
                #NORMALISASI DATA TESTING
                data = self.normalize(data)
                X = data
                prediction = np.array(pd.read_excel("prediction testing.xlsx"))
                prediction = self.normalize(prediction)
                f1 = self.nonlin(np.dot(X, inputlayer))
                f2 = self.nonlin(np.dot(f1, hiddenlayer))
                f2_error = prediction - f2
                loss = np.mean(np.abs(f2_error))
                akurasi = (1 - loss) * 100
                print("Akurasi testing : %.2f" % akurasi)
                f = Figure(figsize=(6, 3.5), dpi=100)
                self.canvas = FigureCanvasTkAgg(f, master=self.frGrafik)
                self.canvas.get_tk_widget().grid(row=3, column=0)
                p = f.gca()
                p.plot(losses)
                p.set_xlabel("Epoh")
                p.set_ylabel("Loss")
                self.canvas.draw()
            else:
                population = 2*np.random.random((8, inputlayer.flatten().size+hiddenlayer.flatten().size)) - 1
                max_generation = 100
                best_fitness = []
                for i in range(max_generation):
                    fitness = self.calculate_fitness(population, inputlayer, hiddenlayer, self.data)
                    r = 0.5                 #CROSSOVER RATE
                    while r < 0.25:
                        r = np.random.rand()
                    population = self.crossover(population, r)
                    mr = 0.25               #MUTATION RATE
                    population = self.mutation(population, mr)
                    fitness = self.calculate_fitness(population, inputlayer, hiddenlayer, self.data)
                    #SELEKSI POPULASI MENGGUNAKAN RANK SELECTION
                    sortedidx = np.argsort(fitness)
                    fitness = fitness[sortedidx]
                    population = population[sortedidx]
                    population = population[:8]
                    fitness = fitness[:8]
                    best_fitness.append(fitness[0])
                i = population[0]
                inputlayer = np.reshape(i[:inputlayer.flatten().size], inputlayer.shape)
                hiddenlayer = np.reshape(i[inputlayer.flatten().size:], hiddenlayer.shape)
                max_epoh = 100
                learning_rate = 1
                data = self.data
                prediction = self.prediction
                for i in range(max_epoh):
                    for j in range(data.shape[0]):
                        #ALUR MAJU BACKPROPAGATION
                        X = np.array([data[j]])
                        f1 = self.nonlin(np.dot(X, inputlayer))
                        f2 = self.nonlin(np.dot(f1, hiddenlayer))
                        
                        #PERHITUNGAN LOSS PREDICTION
                        f2_error = [prediction[j]] - f2
                        loss = np.mean(np.abs(f2_error))
                        if j == 0:
                            print("loss:" + str(loss))
                        
                        #PERHITUNGAN NILAI PENGUBAH BOBOT
                        f2_delta = f2_error*self.nonlin(f2,deriv=True)
                        f1_error = f2_delta.dot(hiddenlayer.T)
                        f1_delta = f1_error * self.nonlin(f1,deriv=True)
                        
                        #PENGUBAHAN BOBOT
                        hiddenlayer += learning_rate * f1.T.dot(f2_delta)
                        inputlayer += learning_rate * X.T.dot(f1_delta)
                data = np.array(pd.read_excel("data testing.xlsx"))
                data = self.normalize(data)
                X = data
                prediction = np.array(pd.read_excel("prediction testing.xlsx"))
                prediction = self.normalize(prediction)
                print(inputlayer.shape)
                f1 = self.nonlin(np.dot(X, inputlayer))
                f2 = self.nonlin(np.dot(f1, hiddenlayer))
                f2_error = prediction - f2
                loss = np.mean(np.abs(f2_error))
                akurasi = (1 - loss) * 100
                print("Akurasi testing : %.2f" % akurasi)
                f = Figure(figsize=(6, 3.5), dpi=100)
                self.canvas = FigureCanvasTkAgg(f, master=self.frGrafik)
                self.canvas.get_tk_widget().grid(row=3, column=0)
                p = f.gca()
                p.plot(best_fitness)
                p.set_xlabel("Epoh / Generasi")
                p.set_ylabel("Loss")
                self.canvas.draw()
            self.akurasi.set(value=akurasi)
            bulan = np.argwhere(np.array(self.bulans) == self.bulan.get())[0,0]
            data = self.data[bulan]
            f1 = self.nonlin(np.dot(data, inputlayer))
            f2 = self.nonlin(np.dot(f1, hiddenlayer))
            self.hasil.set(value=np.mean(f2))
        
Main(root, "Prediksi Curah Hujan")
root.mainloop()