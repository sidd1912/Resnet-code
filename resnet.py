"""
A simple toy resnet model and its implementation

Requirements
============
python
keras
tensorflow
sckit-learn

"Added a mlp model also to this code for comparison study with resnet"
"""
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets
from keras.utils import to_categorical
from keras.models import Sequential,Model
from keras.layers import Dense,Input,Add,Activation
import tensorflow as tf
import numpy as np

#Set the seeds
seed=111
tf.random.set_seed(seed)
np.random.seed(seed)

#Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
y=to_categorical(y)
input_shape=X.shape[1]
output_shape=y.shape[1]



"""
        Training on iris dataset with a vanilla mlp model.
"""
print("Training mlp with cross validation\n")

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

j=False
cvscores = []
for train, test in kfold.split(X, y.argmax(1)):
    #mlp model
    model = Sequential()
    model.add(Dense(4, input_shape=(input_shape,),activation='relu'))
    model.add(Dense(2,activation='relu'))
    model.add(Dense(3,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(2,activation='relu'))
    model.add(Dense(3,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(2,activation='relu'))
    model.add(Dense(3,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(2,activation='relu'))
    model.add(Dense(output_shape,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    if j==False:
        model.summary()
        j=True
    model.fit(X[train],y[train],batch_size=10,verbose=0,epochs=100)
    scores = model.evaluate(X[test], y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("Average MLP Score:")
mlp_score=[np.mean(cvscores),np.std(cvscores)]
print("%.2f%% (+/- %.2f%%)" % (mlp_score[0],mlp_score[1] ))
ml=len(model.layers)
mp=model.count_params()

del model
"""
        Training on iris dataset with a resnet model.
"""
print("Training resnet with cross validation\n")

def resnet_block(x):
    t=x.get_shape().as_list()[1]
    i=x
    x=Dense(3,activation='relu')(i)
    x=Dense(4,activation='relu')(x)
    x=Dense(t)(x)
    x=Add()([x,i])
    x=Activation('relu')(x)
    return x


j=False
cvscores = []
for train, test in kfold.split(X, y.argmax(1)):
    i=Input(shape=(input_shape,))
    x=Dense(2,activation='relu')(i)
    x=resnet_block(x)
    x=resnet_block(x)
    x=resnet_block(x)
    o=Dense(output_shape,activation='softmax')(x)
    model=Model(i,o)    
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    if j==False:
        model.summary()
        j=True        
    model.fit(X[train],y[train],batch_size=10,verbose=0,epochs=100)
    scores = model.evaluate(X[test], y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("Average Resnet Score:")
resnet_score=[np.mean(cvscores),np.std(cvscores)]
print("%.2f%% (+/- %.2f%%)" % (resnet_score[0],resnet_score[1] ))
rl=len(model.layers)
rp=model.count_params()

print("\n\n")
print("Complete result: \n")
print("MLP")
print('Layers:'+str(ml)+' Parameters'+str(mp))
print('Score: ')
print("%.2f%% (+/- %.2f%%)" % (mlp_score[0],mlp_score[1] ))
print("\nResnet")
print('Layers:'+str(rl)+' Parameters'+str(rp))
print('Score: ')
print("%.2f%% (+/- %.2f%%)" % (resnet_score[0],resnet_score[1] ))
