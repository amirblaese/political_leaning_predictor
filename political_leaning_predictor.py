import praw
from psaw import PushshiftAPI
import pandas as pd
import tensorflow as tf
import numpy as np
import math as math
from tensorflow import keras
import re
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import os
import glob
import shutil
import string
from sklearn.model_selection import train_test_split



# read in the files
s=pd.read_csv('socialism'+'_'+''+'_comments'+'.csv',lineterminator='\n')
l=pd.read_csv('conservative'+'_'+''+'_comments'+'.csv',lineterminator='\n')
s['label']=0 # 0 = socialism tag for classification
l['label']=1 # 1 = conservative tag for classification
print(str(len(l))+' con comments')
print(str(len(s))+' soc comments')

print(str(len(l)+len(s)),'files total')
#Filter the data sets to remove downvoted comments and [removed] comments
s_filtered=s.query('Scores > -1 and Body != "[removed]"')
s_filtered=s_filtered.reset_index()

print(str(len(s)-len((s_filtered)))+' soc files removed, # files now: '+ str(len(s_filtered)))
l_filtered=l.query('Scores > -1 and Body != "[removed]"')
l_filtered=l_filtered.reset_index()
#reset the index to have actual length of the datasets.
print(str(len(l)-len((l_filtered)))+' con files removed, # files now: '+ str(len(l_filtered)))
l_filtered=l_filtered.reset_index()
s_filtered=s_filtered.reset_index()

#now we make the two files equal length to prevent statistical bias.
l_filtered=l_filtered.truncate(before=None,after=(min(len(s_filtered),len(l_filtered))))
s_filtered=s_filtered.truncate(before=None,after=(min(len(s_filtered),len(l_filtered))))


print(str(len((l_filtered)))+' files after truncation')
print(str(len((s_filtered)))+' files after truncation')

#combine into one array
data=pd.concat([s_filtered,l_filtered], axis=0)
print(str(len(data))+' files total after filter + concatenation')
x=np.array(data['Body'])
y=np.array(data['label'])
#test train split, 20% seems about right
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



# def custom_standardization(input_data):
#   lowercase = tf.strings.lower(input_data)
#   stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
#   return tf.strings.regex_replace(stripped_html,'[%s]' % re.escape(string.punctuation),'')

max_features = 10000
embedding_dim = 128
sequence_length = 250

vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

#Vectorize the text here

def vectorize_text(text):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text)


 model_t = tf.keras.models.load_model('so_vs_co')


#use the vectorize function from above here


vectorize_layer.adapt(x_train)
vectorize_layer.adapt(x_test)

x_train_vctr=vectorize_text(x_train)

x_test_vctr=vectorize_text(x_test)


model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(2)])



model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Accuracy saturates after 3 epochs
model.fit(x_train_vctr, y_train, epochs=3)


loss, accuracy = model_t.evaluate(x_test_vctr,y_test)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

#This will reproduce loss and accuracy similar to trained data.


probability_model = tf.keras.Sequential([model_t, 
                                         tf.keras.layers.Softmax()])
wtf='workers must unite'
wtf_v=vectorize_text(wtf)

probability_model.predict(wtf_v)




def login(id,secret,agent):
  r = praw.Reddit(client_id=id,
                client_secret=secret,
                user_agent=agent)
  api=PushshiftAPI(r)
  return r,api

  api = PushshiftAPI(r)


  #analyze reddit users data here:

  inn=input('Enter reddit username: ')
a=r.redditor(str(inn))
s=a.comments
all_commentid=list(s.new())

all_commentbody=[None]*len(all_commentid)
all_commentbody_vctr=[None]*len(all_commentid)

scores=[None]*len(all_commentid)
so=np.zeros(len(all_commentid))
li=np.zeros(len(all_commentid))
print("Downloading user <",a,">'s comments...")
for i in range(len(all_commentid)):
  all_commentbody[i]=all_commentid[i].body

print("Complete...")  
print("Analysing", len(all_commentid),"comments...")  

#vectorize_layer.adapt(all_commentbody)
#all_commentbody[0]='Trump throws everyone under the bus. Eitheryone around trump leaks such information and it is consistent with all the other terrible things people reveal about truth. Either everyone but trump is credible and Trump is the only credible person in the whole world, or it is time for trump supporters to wake up and hear what literally everyone who is exposed to him already knows.'
for i in range(len(all_commentid)):
  all_commentbody_vctr[i]=vectorize_text(all_commentbody[i])
  scores[i]=probability_model.predict(all_commentbody_vctr[i])
  #print(scores[i])
  so[i]=scores[i][0][0]
  li[i]=scores[i][0][1]
print("Complete...")  
print("===========================================")  
print("Results:")  
print("===========================================")  
print("Probability of being left-leaning:  ", np.round(100*so.mean(),2),'+/-', np.round(100*np.std(so),0),'%')
print("Probability of being right-leaning:  ", np.round(100*li.mean(),2),'+/-', np.round(100*np.std(li),0),'%')
print("===========================================")  

print("Your most left comment is:")
print(all_commentbody[np.argmax(so)][0:100],"...", "score = ", max(so))
print("===========================================")  

print("Your most right comment is:")
print(all_commentbody[np.argmax(li)][0:100],"...", "score = ",max(li))




