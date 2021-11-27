import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow import keras
from sklearn.metrics import mean_squared_error

from generator_utils import create_dataset
from generator_utils import split_dataset
from generator_utils import min_max_dataset
from generator_utils import DataGenerator_Conv1D_class

'''Parameters'''
#dataset parameters
items=50
dim=10
dataset_folder=os.getcwd() +'\\dummie_data\\'

#generator parameters
params = {'dim': dim,
          'batch_size': 2,
          'shuffle': False}

'''Dataset''' 

#create a dummie dataset
create_dataset(dataset_folder,items,dim)

#split into traning and validation sets
X_files_train, X_files_val, y_dict=split_dataset(dataset_folder,dim,extension='.npy',
                                                 test_size=0.5, val_size=0.25, random_state=50)

#dataset generator
min_input,max_input=min_max_dataset(X_files_train,y_dict,dataset_folder,**params)
train_generator = DataGenerator_Conv1D_class(X_files_train, y_dict,dataset_folder,min_input,max_input, **params)
val_generator = DataGenerator_Conv1D_class(X_files_val,y_dict,dataset_folder,min_input,max_input,**params)


'''Model'''

#reset model
keras.backend.clear_session()
# define model
model = Sequential()
model.add(keras.Input(shape=(params['dim'],1)))
model.add(Conv1D(filters=32, kernel_size=4,strides=1,padding='same', input_shape=(params['dim'],1), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
#
model.add(Conv1D(filters=32, kernel_size=2,strides=1,padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# compile model
model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00000001)

# train model
model_history = model.fit_generator(generator=train_generator,
                                    validation_data=val_generator,
                                    epochs=50,callbacks=[reduce_lr],shuffle=True)


'''Weights and biases'''

#model architecture
with open('modelsummary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    
#history of errors and learning rates
hist_df = pd.DataFrame(model_history.history)

#model weights
layer_weights=[]
layer_biases=[]
for i in range(len(model.layers)):
    try:
        layer_weights.append(model.layers[i].get_weights()[0].ravel())
        layer_biases.append(model.layers[i].get_weights()[1].ravel())
    except:
        pass
    
'''Evaluation'''

#evaluate model
loss_acc_training=model.evaluate_generator(generator=train_generator)
loss_acc_validation=model.evaluate_generator(generator=val_generator)

#predict y value
y_pred=model.predict_generator(generator=val_generator)
y_pred=y_pred.flatten()
y_pred_val=np.ravel(y_pred).tolist()
qty_y_val=len(X_files_val)-(len(X_files_val)%params['batch_size'])
y_val=[y_dict[x] for x in X_files_val][:qty_y_val]

y_pred=model.predict_generator(generator=train_generator)
y_pred_train=np.ravel(y_pred).tolist()
qty_y_train=len(X_files_train)-(len(X_files_train)%params['batch_size'])
y_train=[y_dict[x] for x in X_files_train][:qty_y_train]

#mean squared error
mse_train=mean_squared_error(y_train,y_pred)
mse_val=mean_squared_error(y_val,y_pred_val)

        
# Plots:
metric='loss'
plt.figure()
plt.plot(model_history.history[metric])
plt.plot(model_history.history['val_' + metric])
plt.title('model ' + metric)
plt.ylabel(metric, fontsize='large')
plt.xlabel('epoch', fontsize='large')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.figure()
plt.plot(y_pred_train,y_train,'.')
plt.title('prediction vs training data')
plt.ylabel('y_train', fontsize='large')
plt.xlabel('y_pred_train', fontsize='large')
plt.show()