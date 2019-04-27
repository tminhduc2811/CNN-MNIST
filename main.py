from keras import *
import matplotlib.pyplot as plt
import numpy as np

# Load model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = models.model_from_json(loaded_model_json)

# Load weights into model
loaded_model.load_weights('model.h5')
print('Loaded model from disk')

# Test an image from data set
_, (x_test, y_test) = datasets.mnist.load_data()
y_predict = loaded_model.predict(x_test[12].reshape(1, 28, 28, 1))
print('Predicted value: %s, with: %.2f%% ' % (np.argmax(y_predict), np.amax(y_predict) * 100))

# Show image
plt.imshow(x_test[12].reshape(28, 28), cmap='gray')
plt.waitforbuttonpress(0)
