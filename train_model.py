import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping

x = np.linspace(0, 50, 501)
y = np.sin(x)
plt.figure(figsize=(25, 20))
plt.plot(x, y)

df = pandas.DataFrame(index=x, data=y, columns=['Sine'])
index_train = len(df) - int(len(x) * 0.1)
print(index_train)

x_test = df.iloc[index_train:]
x_train = df.iloc[:index_train]

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

length = int(len(x_test)-1)
generator = TimeseriesGenerator(x_train, x_train, length=length, batch_size=1)
generator_validation = TimeseriesGenerator(x_test, x_test, length=length, batch_size=1)

model = Sequential()

model.add(SimpleRNN(length, input_shape=(length, 1)))

model.add(Dense(1))


early = EarlyStopping(patience=2)
model.compile(optimizer='adam', loss='mse')
model.fit_generator(generator, epochs=120,validation_data=generator_validation,callbacks=[early])

losedf = pandas.DataFrame(model.history.history)
losedf.plot(figsize=(18, 12))
plt.show()

last_batch = x_train[-length:]
current_batch = last_batch.reshape((1, length, 1))
test_predictions = []
for i in range(len(x_test)):
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]

    # store prediction
    test_predictions.append(current_pred)

    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

true_predictions = scaler.inverse_transform(test_predictions)

dfn = pandas.DataFrame(index=x, data=y, columns=["sine"]).iloc[451:]
dfn['SimpleRNN'] = true_predictions

# Retrain model to add ltsm

generator = TimeseriesGenerator(x_train, x_train, length=length, batch_size=1)
generator_validation = TimeseriesGenerator(x_test, x_test, length=length, batch_size=1)

model = Sequential()

model.add(LSTM(length, input_shape=(length, 1)))

model.add(Dense(1))


early = EarlyStopping(patience=2)
model.compile(optimizer='adam', loss='mse')
model.fit_generator(generator, epochs=120,validation_data=generator_validation,callbacks=[early])




last_batch = x_train[-length:]
current_batch = last_batch.reshape((1, length, 1))
test_predictions = []
for i in range(len(x_test)):
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]

    # store prediction
    test_predictions.append(current_pred)

    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

true_predictions = scaler.inverse_transform(test_predictions)


dfn['LSTM'] = true_predictions
dfn.plot(figsize=(18, 12))
plt.show()



# Step there

newscaler = MinMaxScaler()

full_data = newscaler.fit_transform(df)

length = int(len(x_test)-1)
generator = TimeseriesGenerator(full_data, full_data, length=length, batch_size=1)


model = Sequential()

model.add(LSTM(length, input_shape=(length, 1)))

model.add(Dense(1))

early = EarlyStopping(patience=2,monitor='loss')
model.compile(optimizer='adam', loss='mse')
model.fit_generator(generator, epochs=120,callbacks=[early])


forecast = []
last_batch = x_train[-length:]
current_batch = last_batch.reshape((1, length, 1))
for i in range(25):
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]

    # store prediction
    forecast.append(current_pred)

    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

forecast = scaler.inverse_transform(forecast)

stop = (25*0.1) + 50.1
index = np.arange(50.1,stop,0.1)

