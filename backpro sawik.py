import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

#READ DATA DAN NORMALISASI
data = np.array(pd.read_excel("data.xlsx"))
prediction = np.array(pd.read_excel("prediction.xlsx"))
data = normalize(data)
df = pd.DataFrame(data)
df.to_excel('data normalisasi.xlsx')
prediction = normalize(prediction)
df = pd.DataFrame(prediction)
df.to_excel('prediction normalisasi.xlsx')

#ATUR JUMLAH NEURON
jml_neuron = 15

#INISIASI JARINGAN
inputlayer = 2*np.random.random((data.shape[1], jml_neuron)) - 1
hiddenlayer = 2*np.random.random((jml_neuron, prediction.shape[1])) - 1

#ATUR BATAS EPOH
max_epoh = 100

#ATUR LEARNING RATE
learning_rate = 1

losses = []
for i in range(max_epoh):
    for j in range(data.shape[0]):
        #ALUR MAJU BACKPROPAGATION
        X = np.array([data[j]])
        f1 = nonlin(np.dot(X, inputlayer))
        f2 = nonlin(np.dot(f1, hiddenlayer))
        
        #PERHITUNGAN LOSS PREDICTION
        f2_error = [prediction[j]] - f2[0]
        loss = np.mean(np.abs(f2_error))
        if j == 0:
            print("loss:" + str(loss))
            losses.append(loss)
        
        #PERHITUNGAN NILAI PENGUBAH BOBOT
        f2_delta = f2_error*nonlin(f2,deriv=True)
        f1_error = f2_delta.dot(hiddenlayer.T)
        f1_delta = f1_error * nonlin(f1,deriv=True)
        
        #PENGUBAHAN BOBOT
        hiddenlayer += learning_rate * f1.T.dot(f2_delta)
        inputlayer += learning_rate * X.T.dot(f1_delta)
#TESTING
#AMBIL DATA TESTING
data = np.array(pd.read_excel("data testing.xlsx"))
#NORMALISASI DATA TESTING
data = normalize(data)
X = data
#AMBIL DATA PREDIKSI TESTING
prediction = np.array(pd.read_excel("prediction testing.xlsx"))
#NORMALISASI DATA PREDIKSI TESTING
prediction = normalize(prediction)
#ALUR MAJU BACKPRO DENGAN BOBOT YANG SUDAH DITRAINING
f1 = nonlin(np.dot(X, inputlayer))
f2 = nonlin(np.dot(f1, hiddenlayer))
f2_error = prediction - f2
loss = np.mean(np.abs(f2_error))
akurasi = (1 - loss) * 100
print("Akurasi testing : %.2f" % akurasi)
#MEMBUAT DAN MENAMPILKAN GRAFIK
plt.title("Grafik Loss")
plt.xlabel("Epoh")
plt.ylabel("Loss")
plt.plot(losses)
