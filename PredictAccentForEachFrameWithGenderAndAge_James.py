#!/usr/bin/env python
# coding: utf-8

# In[2]:


from utils_3_updated import *
import pandas as pd
import numpy as np
import glob
from keras.utils import np_utils
import multiprocessing
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from numpy import save


# In[3]:


path_to_midway = 'C:/Users/wanyi/Desktop/Deep Learning/Group Project/Fixed_Window/testing/' #!!!!!!change your scratch/home directory to save and load npy files


# ### Loading metadata:
# * Loading filtered mp3 audio speech files (52433, 8) - Refer to notebook 1

# In[4]:


print('Loading metadata')
filtered_audios = pd.read_csv('C:/Users/wanyi/Desktop/Deep Learning/Group Project/Fixed_Window/Fix_window_continent_22k_globorder.csv')
filtered_audios['filename'] = filtered_audios['filename'].apply(lambda x: x.replace('.mp3','.wav'))
filtered_audios.shape


# ### Create dummies for Gender and Age:
# * Age dummies : 'seventies', 'thirties', 'twenties', 'sixties', 'fourties', 'fifties', 'teens', 'eighties'
# * Gender dummies : 'male', 'female'

# In[5]:


filtered_audios_dummies= pd.concat([filtered_audios['filename'],filtered_audios['accent'], pd.get_dummies(filtered_audios['gender']), pd.get_dummies(filtered_audios['age'])], axis=1)


# In[6]:


print('Checking stats of accent')
filtered_audios_dummies['accent'].value_counts(normalize=True)


# ### Filtered accent:
# * Selected top 5 accents by looking at the distribution; 'US', 'England', 'Canada', 'Indian', 'Australia'
# * Shape of the filtered file is 41366, 8
# * Mapped accent category to number 

# In[7]:


'''
accent_dict = {'us':0, 'australia':1, 'england':2, 'indian':3, 'canada':4, 'scotland':5,
       'african':6, 'newzealand':7, 'ireland':8, 'bermuda':9, 'wales':10, 'malaysia':11,
       'philippines':12, 'hongkong':13, 'singapore':14, 'southatlandtic':15}
filtered_audios_dummies['accent'] = filtered_audios_dummies['accent'].map(accent_dict)
filtered_audios_dummies_accent    = filtered_audios_dummies[filtered_audios_dummies['accent'].isin([0, 1, 2, 3, 4])]
filtered_audios_dummies_accent.shape
'''


# In[8]:


filtered_audios_dummies= filtered_audios_dummies[:100] #-- for testing purpose


# ### Seperate and split train, validation and test set:
# * X = array of filenames, Y = array of top 5 accents
# * Split train, val, test into 50%, 20% and 30%
# * Stratified accent for all 3 sets to have equal distribution of y

# In[9]:


X = filtered_audios_dummies.drop('accent', axis=1)#['filename'].values
Y = filtered_audios_dummies['accent'].values


# In[10]:


X_, X_test, Y_, y_test = train_test_split(X, Y, test_size=0.30, random_state=99, stratify=Y)
X_train, X_val, y_train, y_val = train_test_split(X_, Y_, test_size=0.2855, random_state=99, stratify=Y_)


# In[11]:


print ('shape of train file:', X_train.shape)
print ('shape of validation file:', X_val.shape)
print ('shape of test file:', X_test.shape)


# ### Read wav file for train and test:
# * Multiprocessed both train and test data files
# * Used librosa library to load wav files

# In[12]:


print('Loading wav files....')
start_time  = time.time()
pool        = multiprocessing.Pool(processes=multiprocessing.cpu_count())
train_wav   = pool.map(get_wav, X_train['filename'].values)
val_wav     = pool.map(get_wav, X_val['filename'].values)
test_wav    = pool.map(get_wav, X_test['filename'].values)
print('total time for processing is:', (time.time()- start_time)/60.0)


# ### Save train, test and validation wav file:
#  * Saving y and X for train, test and validation

# In[ ]:


print("Starting saving files ....")
np.save(path_to_midway + 'train_wav.npy', train_wav)
np.save(path_to_midway + 'val_wav.npy', val_wav)
np.save(path_to_midway + 'test_wav.npy', test_wav)

np.save(path_to_midway + 'y_train.npy', y_train)
np.save(path_to_midway + 'y_test.npy', y_test)
np.save(path_to_midway + 'y_val.npy', y_val)

X_train.to_csv(path_to_midway+'X_train.csv', index=False)
X_test.to_csv(path_to_midway+'X_test.csv', index=False)
X_val.to_csv(path_to_midway+'X_val.csv', index=False)

print("Files saved!")


# In[13]:


print("Starting loading files ....")

train_wav = np.load(path_to_midway + "train_wav.npy", allow_pickle=True)
test_wav  = np.load(path_to_midway + 'test_wav.npy', allow_pickle=True)
val_wav   = np.load(path_to_midway + 'val_wav.npy', allow_pickle=True)

y_train   = np.load(path_to_midway + 'y_train.npy', allow_pickle=True)
y_test    = np.load(path_to_midway + 'y_test.npy', allow_pickle=True)
y_val    = np.load(path_to_midway + 'y_val.npy', allow_pickle=True)


X_train   = pd.read_csv(path_to_midway+'X_train.csv')
X_test    = pd.read_csv(path_to_midway+'X_test.csv')
X_val     = pd.read_csv(path_to_midway+'X_val.csv')


# ### Trim log silences/pauses in train and test:

# In[14]:


print('Trimming wav files....')
start_time  = time.time()
train_trim  = pool.map(trim_long_silences, train_wav)
val_trim    = pool.map(trim_long_silences, val_wav)
test_trim   = pool.map(trim_long_silences, test_wav)
print('total time for processing is:', (time.time()- start_time)/60.0)


# ### Feature extraction using Kaldi-toolkit:
# * mfcc - 13 (12 cfpc coef, 1 energy)
# * filterbank - 64
# * delta_1, delta_2 - 64 each
# * Frame level accent vector

# In[15]:


print('Get Kaldi features....')
start_time      = time.time()
train_features, y_train_, train_len_  = get_kaldi_features(train_trim, y_train, X_train)
val_features, y_val_, val_len_        = get_kaldi_features(val_trim, y_val, X_val)
test_features, y_test_, test_len_     = get_kaldi_features(test_trim, y_test, X_test)
print('total time for processing is:', (time.time()- start_time)/60.0)


# ### Training a Model:
# * conv1D

# In[16]:


batch_size  = 100 
epochs      = 20 
nb_classes  = 4
input_shape = (train_features.shape[1],1) 


# In[17]:


input_shape


# In[18]:


Y_train    = np_utils.to_categorical(y_train_, nb_classes)
Y_val      = np_utils.to_categorical(y_val_, nb_classes)
xts        = train_features.shape
train_X    = np.reshape(train_features, (xts[0], xts[1], 1))
xtv        = val_features.shape
val_X      = np.reshape(val_features, (xtv[0], xtv[1], 1))
xtt        = test_features.shape
test_X      = np.reshape(test_features, (xtt[0], xtt[1], 1))


# In[19]:


print("Start modeling.....")


# In[21]:


nn_1d = train_conv1D_model(train_X, Y_train, val_X, Y_val, input_shape = (train_features.shape[1],1), nb_classes = 4, batch_size = 20, epochs = 2)


# ### Save conv1D model

# In[23]:


print("Model completed and now saving modeling........")
nn_1d.save("nn_conv1d_fixwindow.h5")
print("Saved model to RCC")
#save_model1 = open(path_to_midway+ 'nn_conv1d', 'wb') 
#pickle.dump(nn_1d, save_model1)
#save_model1.close()

print("Model saved...")
print("Start predicting test set....")


# ### Prediction for test set:
# * Calculate frame level accuracy

# In[24]:


y_pred_test     = nn_1d.predict(test_X)
y_pred_test_cls = np.argmax(y_pred_test, axis=1)


# In[25]:


y_pred_val     = nn_1d.predict(val_X)
y_pred_val_cls = np.argmax(y_pred_val, axis=1)


# In[26]:


y_pred_train     = nn_1d.predict(train_X) 
y_pred_train_cls = np.argmax(y_pred_train, axis=1)


# In[34]:


print ('Frame level F1_score for train set:', f1_score(y_train_, y_pred_train_cls, average = "micro"))


# In[33]:


print ('Frame level F1_score for val set:', f1_score(y_val_, y_pred_val_cls, average = "micro"))


# In[32]:


print ('Frame level F1_score for test set:', f1_score(y_test_, y_pred_test_cls, average = "micro"))


# ### Get speaker level accuracy:
# * Combine frame level accent ouput to find a major accent label for a speaker

# In[35]:


y_pred_train_cls_sp = get_speaker_pred(train_len_['frame_len'].values, y_pred_train_cls)
y_pred_test_cls_sp  = get_speaker_pred(test_len_['frame_len'].values, y_pred_test_cls)
y_pred_val_cls_sp   = get_speaker_pred(val_len_['frame_len'].values, y_pred_val_cls)


# In[36]:


print ('Speaker level F1_score for train:', f1_score(train_len_['accent'], y_pred_train_cls_sp,average = "micro"))


# In[37]:


print ('Speaker level F1_score for val:',   f1_score(val_len_['accent'], y_pred_val_cls_sp, average = "micro"))


# In[38]:


print ('Speaker level F1_score for test:',  f1_score(test_len_['accent'], y_pred_test_cls_sp, average = "micro"))

