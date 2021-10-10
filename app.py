import streamlit as st
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import numpy as np
import tensorflow 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image 
fig=plt.figure()
st.title('Cotton Disease Detection')


def predict(image):
  c_model='cottonmobilenet.h5'
  IMAGE_SHAPE=(224,224,3)
  models=load_model(c_model,compile=False,custom_objects={'KerasLayer': hub.KerasLayer})
  test_image=image.resize((224,224))
  test_image=img_to_array(test_image)
  test_image=test_image/255.0
  test_image=np.expand_dims(test_image,axis=0)
  classes=['diseased cotton leaf','diseased cotton plant','fresh cotton leaf','fresh cotton plant']
  predictions=models.predict(test_image)
  scores=tf.nn.softmax(predictions[0])
  results={
      'diseased cotton leaf':0,
      'diseased cotton plant':1,
      'fresh cotton leaf':2,
      'fresh cotton plant':3
  }
  results="This is a",classes[np.argmax(scores)]
  return results


def main():
  upload=st.file_uploader('UPLOAD IMAGE',type=['jpg','png','jpeg'])
  btn=st.button('Predict')
  if upload is not None:
    image=Image.open(upload)
    st.image(image,caption='Uploaded')
  if btn:
    if upload is None:
      st.write('Upload Image!!!')
    else:
      with st.spinner('Model Loading...'):
        plt.imshow(image)
        plt.axis('off')
        predictions=predict(image)
        st.success('Predicted')
        st.write(predictions)
        st.pyplot(fig)

if __name__=="__main__":

  main()
