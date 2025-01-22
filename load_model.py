from tensorflow.keras.models import load_model

# Load the .h5 model
model = load_model('C:\Users\shash\aiproject\pneumonia-classification-web-app-python-streamlit\model\pneumonia_classifier.h5')

# Print model summary
model.summary()
