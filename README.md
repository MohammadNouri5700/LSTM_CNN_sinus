# LSTM & CNN Sinus
Prediction of Sinus 
It is well known that artificial neural networks are good at modeling any function. I wonder whether they can go one step further, and learn the generalized model of a function. For simplicity let’s try to learn a sine function with just one parameter A, which controls the frequency:
y = sin(A*x)
For us humans, once we understand the sine function, we know how it behaves under any parameter A. If we are presented with a partial sine wave, we can figure out what A should be, and we can extrapolate the wave out to infinity.
[![1](https://raw.githubusercontent.com/MohammadNouri5700/RNN_sinus/main/download%20(1).png "1")](https://raw.githubusercontent.com/MohammadNouri5700/RNN_sinus/main/download%20(1).png "1")

We used Conv1D layers, since our data is one dimensional.
`model.add(Dense(1))`
LSTM network retains memory of the data it has seen in the past
`model.add(LSTM(length, input_shape=(length, 1)))`


and it's a train result :
[![train](https://raw.githubusercontent.com/MohammadNouri5700/RNN_sinus/main/2.png "train")](https://raw.githubusercontent.com/MohammadNouri5700/RNN_sinus/main/2.png "train")


IN CONCLUSION, THE PREDICTION IS : 
[![PREDICTIONS](https://raw.githubusercontent.com/MohammadNouri5700/RNN_sinus/main/3.png "PREDICTIONS")](https://raw.githubusercontent.com/MohammadNouri5700/RNN_sinus/main/3.png "PREDICTIONS")

