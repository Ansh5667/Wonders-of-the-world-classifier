from flask import Flask, render_template, request
from PIL import Image
import tensorflow as tf
import numpy as np
import pickle


with open('data.pkl', 'rb') as file:
    data = pickle.load(file)

model = tf.keras.models.load_model('model.h5')





app = Flask(__name__)


@app.route('/', methods = ['GET','POST'])
def ulpoad():
    if request.method == 'POST':
        upimage = request.files['image']
        upimage.save('static/{}'.format(upimage.filename))
        path = 'static/{}'.format(upimage.filename)
        image = Image.open(upimage)
        image = image.resize((256, 256)) 
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.expand_dims(image, axis=0)

        
        label_to_class = {'burj khalifa': 0,
                            'chichen itza': 1,
                            'christ the reedemer': 2,
                            'eiffel tower': 3,
                            'great wall of china': 4,
                            'machu pichu': 5,
                            'pyramids of giza': 6,
                            'roman colosseum': 7,
                            'statue of liberty': 8,
                            'stonehenge': 9,
                            'taj mahal': 10,
                            'venezuela angel falls': 11}
        label_to_class = {label: class_name for class_name, label in label_to_class.items()}

        predictions = model.predict(image)

        for i in range(0, len(predictions[0])): 
            val = predictions[0][i]
            if val==1:
                for key, val in data.items():
                    if val==i:
                        predicted_class = key.replace('_',' ')
                break
        predicted_class = predicted_class.title()
        
        return render_template('index.html', output =predicted_class, upimage = path)

        
                
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)