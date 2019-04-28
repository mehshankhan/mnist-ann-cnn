#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 12:07:58 2018

@author: mehshan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 12:01:14 2018

@author: mehshan
"""

# Importing the Keras libraries and packages
from keras.models import load_model
model = load_model('results/mnistANN.h5')

from PIL import Image
import numpy as np

for index in range(10):
    img = Image.open('data/' + str(index) + '.png').convert("L")
    img = img.resize((28,28))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 784)
    # Predicting the Test set results
    y_pred = model.predict_classes(im2arr)
    print(y_pred)
