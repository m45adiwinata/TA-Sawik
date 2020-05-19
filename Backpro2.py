import math
from numpy import zeros
from numpy import random

def sigmoid(x):
    return math.tanh(x)

def deltaSigmoid(y):
    return 1.0 - y**2

class network:
    def __init__(self, inputs, hidden, bobot, bobot2, outputs):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.input_activations = [1.0] * self.inputs
        self.hidden_activations = [1.0] * self.hidden
        self.outputs_activations = [1.0] * self.outputs
        self.input_weights = bobot
        self.output_weights = bobot2
        self.alpha = 0.5
        self.iterations = 10

    def update(self, inputs):
        for input in range(self.inputs - 1):
            self.input_activations[input] = inputs[input]
        for hidden in range(self.hidden):
            sum = 0.0
        for input in range(self.inputs):
            sum += self.input_activations[input] * self.input_weights[input, hidden]
        self.hidden_activations[hidden] = sigmoid(sum)
        for output in range(self.outputs):
            sum = 0.0
            for hidden in range(self.hidden):
                sum += self.hidden_activations[hidden] * self.output_weights[hidden, output]
            self.outputs_activations[output] = sigmoid(sum)
        return self.outputs_activations

    def backpropagate(self, targets):
        output_deltas = [0.0] * self.outputs
        for output in range(self.outputs):
            error = targets - self.outputs_activations[output]
            output_deltas[output] = deltaSigmoid(self.outputs_activations[output]) * error
        hidden_deltas = [0.0] * self.hidden
        
        for hidden in range(self.hidden):
            error = 0.0
            for output in range(self.outputs):
                error += output_deltas[output] * self.output_weights[hidden, output]
            hidden_deltas[hidden] = deltaSigmoid(self.hidden_activations[hidden]) * error

        for hidden in range(self.hidden):
            for output in range(self.outputs):
                update = output_deltas[output] * self.hidden_activations[hidden]
                self.output_weights[hidden, output] = self.output_weights[hidden, output] + self.alpha * update
  
        for i in range(self.inputs):
            for hidden in range(self.hidden):
                update = hidden_deltas[hidden] * self.input_activations[i]
                self.input_weights[i, hidden] = self.input_weights[i, hidden] + self.alpha * update
   
        error = 0.5 * (targets - self.outputs_activations[0]) ** 2
        return error

    def test(self, pattern):
        return self.update(pattern)[0]


    def train(self, patterns):
        for iteration in range(self.iterations):
            error = 0.0
            for pattern in patterns:
                inputs = pattern[0 : 8]
                targets = pattern[8]
                self.update(inputs)
                error += self.backpropagate(targets)
            if iteration % 100 == 0:
                '''print('error %-.5f' % error)'''

def average(data):
    sum = 0.0
    for datum in data:
        sum += datum
    return sum / len(data)

def maximum(data):
    return max(data)

def minimum(data):
    return min(data)

def normalize(data, minimum, maximum):
    newmax = 0.9
    newmin = 0.1
    return (data-minimum)*(newmax-newmin)/(maximum-minimum)+newmin
  

def denormalize(normalized, minimum, maximum):
    return (((normalized * (maximum-minimum)) / 2) + (maximum + minimum)) / 2

def slidingWindow(data):
    windows = []
    beginning = 0
    end = 9
    i = 0
    while i < len(data)-8:
        windows.append(data[beginning:end])
        beginning += 1
        end += 1
        i += 1
    return windows

def run(bobot, bobot2):
    akurasi = []
    data = [[9.7, 14, 6.9, 10.3, 17.1, 8.4, 12.1, 17.9, 21.4, 17.9, 10, 4.3, 9.9],
          [3.7, 10.4, 5.9, 15.6, 12.9, 6.8, 10, 7.2, 6.1, 9.9, 6.4, 19.3, 12.1],
          [5.9, 12.7, 11.6, 8.6, 8.1, 2.1, 8.8, 17, 3.8, 1.8, 9.8, 3, 6.3],
          [5.9, 7.4, 11.5, 3.8, 2, 10.9, 9.4, 1.4, 2.4, 1, 1.7, 0.9, 3.7],
          [0.5, 1.7, 0.6, 2.1, 2.5, 2.8, 3.9, 3.6, 4.4, 0.9, 2, 3.5, 1.8],
          [0.9, 0.6, 1, 0.8, 0.1, 4.6, 0.6, 0.1, 6.3, 0.3, 0.1, 8.8, 6.5],
          [0.4, 0.1, 0.1, 0.3, 0.8, 3.7, 0.6, 1.6, 3.6, 1.5, 0, 3.1, 1.5],
          [2, 0.2, 1.3, 0, 0, 3.7, 0, 0, 0.2, 0.2, 0, 2.4, 0.1],
          [0.2, 0.3, 0, 1.6, 1.5, 9.9, 0, 0.1, 0, 0, 9, 13, 0.3],
          [6.5, 0.6, 2.5, 3.9, 0.3, 8.1, 1.1, 0.4, 0.3, 0, 0, 2.2, 1.7],
          [4, 0.3, 2.5, 3, 0.9, 5.6, 4.2, 3.1, 6.6, 5.1, 1.1, 9.2, 11.5],
          [10.8, 4.2, 21, 9, 10.2, 16.2, 10.8, 10.8, 7.6, 14.2, 15.6, 6.2, 12.9]]
    
    prediction = [[8.4, 12.1, 17.9, 21.4, 17.9, 10, 4.3, 9.9, 19.3],
                    [6.8, 10, 7.2, 6.1, 9.9, 6.4, 19.3, 12.1, 12.4],
                    [2.1, 8.8, 17, 3.8, 1.8, 9.8, 3, 6.3, 2.9],
                    [10.9, 9.4, 1.4, 2.4, 1, 1.7, 0.9, 3.7, 0.4],
                    [2.8, 3.9, 3.6, 4.4, 0.9, 2, 3.5, 1.8, 0.1],
                    [4.6, 0.6, 0.1, 6.3, 0.3, 0.1, 8.8, 6.5, 0.5],
                    [3.7, 0.6, 1.6, 3.6, 1.5, 0, 3.1, 1.5, 0.6],
                    [3.7, 0, 0, 0.2, 0.2, 0, 2.4, 0.1, 3.1],
                    [9.9, 0, 0.1, 0, 0, 9, 13, 0.3, 0.2],
                    [8.1, 1.1, 0.4, 0.3, 0, 0, 2.2, 1.7, 0.2],
                    [5.6, 4.2, 3.1, 6.6, 5.1, 1.1, 9.2, 11.5, 9.7],
                    [16.2, 10.8, 10.8, 7.6, 14.2, 15.6, 6.2, 12.9, 19.5]]

    for sawix in range(12):
        dataset = data[sawix]
        print dataset
        nmin = min(dataset)
        nmax = max(dataset)
        computed_data = []
        for data in dataset:
            computed_data.append(normalize(data, nmin, nmax))
        training_data = slidingWindow(computed_data)
        n = network(8, 2, bobot, bobot2, 1)
        n.train(training_data)
        prediction_dataset = prediction[sawix]
        nmin = minimum(prediction_dataset)
        nmax = maximum(prediction_dataset)
        computed_data = []
        for data in prediction_dataset:
            computed_data.append(normalize(data, nmin, nmax))
        test_data = slidingWindow(computed_data)
        
        predictions = []
        for i in range(len(test_data)):
            value = n.test(test_data[i][0: 8])
            if test_data[i][8] > value:
                predictions.append((test_data[i][8] - value) )
            else:
                predictions.append((value - test_data[i][8]) )
        akurasi.append((1 - sum(predictions)/len(predictions)) * 100)
    return (sum(akurasi)/12)

def testing(bobot, bobot2):
    akurasi = []
    n = network(8, 2, bobot, bobot2, 1)
    prediction = [[8.4, 12.1, 17.9, 21.4, 17.9, 10, 4.3, 9.9, 19.3],
                    [6.8, 10, 7.2, 6.1, 9.9, 6.4, 19.3, 12.1, 12.4],
                    [2.1, 8.8, 17, 3.8, 1.8, 9.8, 3, 6.3, 2.9],
                    [10.9, 9.4, 1.4, 2.4, 1, 1.7, 0.9, 3.7, 0.4],
                    [2.8, 3.9, 3.6, 4.4, 0.9, 2, 3.5, 1.8, 0.1],
                    [4.6, 0.6, 0.1, 6.3, 0.3, 0.1, 8.8, 6.5, 0.5],
                    [3.7, 0.6, 1.6, 3.6, 1.5, 0, 3.1, 1.5, 0.6],
                    [3.7, 0, 0, 0.2, 0.2, 0, 2.4, 0.1, 3.1],
                    [9.9, 0, 0.1, 0, 0, 9, 13, 0.3, 0.2],
                    [8.1, 1.1, 0.4, 0.3, 0, 0, 2.2, 1.7, 0.2],
                    [5.6, 4.2, 3.1, 6.6, 5.1, 1.1, 9.2, 11.5, 9.7],
                    [16.2, 10.8, 10.8, 7.6, 14.2, 15.6, 6.2, 12.9, 19.5]]
    
    for sawix in range(12):
        prediction_dataset = prediction[sawix]
        nmin = minimum(prediction_dataset)
        nmax = maximum(prediction_dataset)
        computed_data = []
        for data in prediction_dataset:
            computed_data.append(normalize(data, nmin, nmax))
        test_data = slidingWindow(computed_data)
        predictions = []
        for i in range(len(test_data)):
            value = n.test(test_data[i][0: 8])
            '''print(value, '->', test_data[i][8])'''
            if test_data[i][8] > value:
                predictions.append((test_data[i][8] - value) )
            else:
                predictions.append((value - test_data[i][8]) )
        akurasi.append((1 - sum(predictions)/len(predictions)) * 100)
    
    for akur in akurasi:
        print ('Akurasi bulan', akur)
    print ('Akurasi Total', sum(akurasi)/12)
    return (sum(akurasi)/12)