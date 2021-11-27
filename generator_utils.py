import shutil
import os
import numpy as np

import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

def create_dataset(dataset_folder,items=100,dim=100):
    try:
        shutil.rmtree(dataset_folder)
    except:
        pass
    os.mkdir(dataset_folder)
    os.mkdir(dataset_folder+'\\y')
    
    y_values=[]
    for i in range(items):
        file_name='p_'+'{:04d}'.format(i)+'.npy'
        np.random.seed(i)
        x=np.random.rand(dim)*10
        y=(x[0]+x[1]**2+x[2]**3)/1110
        np.save(dataset_folder+'\\' + file_name,x)
        y_values.append((file_name,y))    
    np.save(dataset_folder+'\\y\\y_values.npy',np.array(y_values))

def split_dataset(dataset_folder,dim,extension='.npy',test_size=0.5, val_size=0.25, random_state=50):
    # Get all input filenames. Each file has one item.
    fn=[f for f in os.listdir(dataset_folder) if f.endswith(extension)]
    X_files_train,X_files_val_test=train_test_split(fn,test_size=test_size, random_state=random_state)    
    X_files_val=X_files_val_test[0:round(val_size*dim)]         
    y_values_array=np.load(dataset_folder +'y\\y_values.npy')
    y_dict={a : float(b) for a,b in y_values_array}
    return X_files_train, X_files_val, y_dict

def min_max_dataset(X_files_train,y_dict,dataset_folder,**params):
    training_generator_class = DataGenerator_Conv1D_class(X_files_train, y_dict,dataset_folder,0,1,**params)
    #number of samples per batch
    len_training_generator_class=training_generator_class.__len__()
    #maximum and minimum in training dataset
    max_input_list= [np.amax(training_generator_class[n][0]) for n in range(len_training_generator_class) ]
    max_input=max(max_input_list)    
    min_input_list= [np.amin(training_generator_class[n][0]) for n in range(len_training_generator_class) ]
    min_input=min(min_input_list)
    return min_input,max_input


'''This generator is 1D and it pads 
each item so all of them have the same size'''
class DataGenerator_Conv1D_class(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels,folder,min_input=0,max_input=1, batch_size=32, dim=100, shuffle=False):
        'Initialization'
        self.max_input = max_input
        self.min_input = min_input
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.folder=folder
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, self.dim,1))
        y = np.empty((self.batch_size))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store item
            X_raw=np.load(self.folder + ID)
            # pad item for all to have the same maximum dimension=dim
            X_padded=np.pad(X_raw,(0,self.dim-X_raw.shape[0]),constant_values=(0,0))
            X[i,] = X_padded.reshape(-1,1)/self.max_input
            # Store output
            y[i] = self.labels[ID]

        return X, y    
    

        
