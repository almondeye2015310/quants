pred = list()
training = list()

def lstm(var, attr, day):
    train_int = df[[var] + attr].dropna()
    scaler = StandardScaler().fit(train_int[attr])
    train_scal = pd.DataFrame(scaler.transform(train_int[attr]), columns=attr, index=train_int.index)
    train_int.update(train_scal)
    train_data = np.array(train_int)

    x_train = []
    y_train = []

    for i in range(day, len(train_data)):
        xset = []
        for j in range(train_data.shape[1]):
            a = train_data[i - day:i, j]
            xset.append(a)
        x_train.append(xset)
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train_3D = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    
    n1_hidden = 4
    n2_hidden = 4
    tf.random.set_seed(1)

    model = Sequential()
    model.add(LSTM(n1_hidden, activation='relu', return_sequences=True, input_shape=(x_train_3D.shape[1], day)))
    model.add(Dropout(0.2))
    model.add(LSTM(n2_hidden,return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    batch_size = 1
    epochs = 8
    model.fit(x_train_3D, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

    test_int = pd.concat([train_int.iloc[train_int.shape[0]-day:,:], test_df[[var] + attr]])
    test_scal = pd.DataFrame(scaler.transform(test_int[attr]), columns=attr, index=test_int.index)
    test_int.update(test_scal)
    test_data = np.array(test_int)

    x_test = []
    y_test = []

    for i in range(day, len(test_data)): 
        xset = []
        for j in range(test_data.shape[1]):
            a = test_data[i - day:i, j]
            xset.append(a)
        x_test.append(xset)
        y_test.append(test_data[i, 0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    predictions = model.predict(x_test)
    pred.append(predictions)
    
    pre_train = model.predict(x_train)
    training.append(pre_train)

    from sklearn import metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(np.corrcoef(pre_train[:,0], y_train))
    print(np.corrcoef(predictions[:,0], y_test))
    print(f'rmse: {rmse:.4f}')
    
    plt.subplot(121)
    plt.plot(pre_train, color="orange")
    plt.plot(y_train)
    plt.title(var)
    
    plt.subplot(122)
    plt.plot(predictions, color="orange")
    plt.plot(y_test)
    plt.title(var)
    plt.show()
