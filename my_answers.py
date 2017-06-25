import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    import numpy as np
    # containers for input/output pairs
    X = np.zeros((len(series) - window_size, window_size))
    y = np.zeros((len(series) - window_size, 1))
    for i in range(X.shape[0]):
        X[i, :] = series[i:i+window_size]
        y[i, 0] = series[i+window_size]              
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    ### create required RNN model
    # import keras network libraries
    from keras.layers import Input
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.models import Model
    import keras

    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)

    # build an RNN to perform regression on our time series input/output data
    inputs = Input(X_train.shape[1:], dtype='float32', name='main_input')
    hidden = LSTM(5)(inputs)
    outputs = Dense(1, activation='tanh')(hidden)
    model = Model(inputs=inputs, outputs=outputs)
    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    chars = sorted(list(set(text)))
    print (chars)
    # remove as many non-english characters and character sequences as you can 
    import re
    
    # shorten any extra dead space created above
    text = re.sub(r'[^a-z,.;:?]', ' ', text)
    
    # shorten any extra dead space created above
    text = re.sub('\s+',' ', text)
    chars = sorted(list(set(text)))
    print (chars)
    
    return (text, chars)


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    start = 0
    end = window_size
    while end < len(text):
        inputs.append(text[start:end])
        outputs.append(text[end])
        start += step_size
        end += step_size
       
    return inputs,outputs
