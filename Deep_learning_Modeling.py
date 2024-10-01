#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Data processing and manipulation Libraries
import numpy as np
import pandas as pd

#Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#Deep learning libraries
import tensorflow as tsf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam

#Evaluation and preprocessing libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


# In[5]:


#Loading the dataset
file_path=input("Please enter the path to your training data set: ")
data=pd.read_csv(file_path)


# In[6]:


#Data Inspection(Basic)
print("Data shape:", data.shape)
print(data.info())
print(data.describe())


# In[7]:


#Checking for the missing values
print(data.isnull().sum())


# In[8]:


#Filling in the missing values with the median method

# Separate numeric and categorical columns
numeric_columns = data.select_dtypes(include=['number']).columns
categorical_columns = data.select_dtypes(exclude=['number']).columns

# Fill missing values in numeric columns with the median
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# Fill missing values in categorical columns with the mode
data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])


# In[9]:


#Conversion of categorical values (if any) into nuerical values
data=pd.get_dummies(data, drop_first=True)


# In[16]:


# Splitting the data into features (X) and target (y)
for_X = input("Please enter the parameter(s) for features (X) (comma-separated for multiple columns): ").split(",")
for_y = input("Please enter the parameter for Target (y): ")

# Strip any leading or trailing spaces from input columns
for_X = [x.strip() for x in for_X]
for_y = for_y.strip()

# Check if the input columns exist in the DataFrame
missing_features = [col for col in for_X if col not in data.columns]
if for_y not in data.columns:
    print(f"Error: Target column '{for_y}' not found in DataFrame.")
elif missing_features:
    print(f"Error: The following feature columns were not found in DataFrame: {missing_features}")
else:
    # Drop the target column from features
    X = data.drop(for_X, axis=1)
    y = data[for_y]

    print("Features (X):")
    print(X.head())
    print("Target (y):")
    print(y.head())


# In[17]:


#Scaling features using StandardScalar (AKA: Normalization)
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)


# In[18]:


#Splitting the data into training and Testing sets
#Splitting the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)


# In[19]:


print("X_train shape:", X_train.shape)
print("X_test shape:",X_test.shape)


# In[20]:


from tensorflow.keras import Input

# Initializing the sequential model
model = Sequential()

# Add an Input layer
model.add(Input(shape=(X_train.shape[1],)))

# Add first hidden layer
model.add(Dense(128, activation='relu'))

# Adding one more hidden layer with a Drop-out
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))  # Drop-out necessary to avoid overfitting

# Adding an output layer for binary classification example
model.add(Dense(1, activation='sigmoid'))  # OR 'softmax' for multi-class classification

# Compiling the model with an optimizer, loss function, and metrics
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Show (print) the model's architecture
model.summary()


# In[21]:


#Training the built model
history=model.fit(X_train,y_train,validation_split=0.2,epochs=20,batch_size=32,verbose=1)

#Plot training history (loss and accuracy over epochs)
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[22]:


# Import necessary evaluation metrics with zero_division handling
from sklearn.metrics import classification_report, confusion_matrix

# Evaluate the trained model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions on the test data
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Model evaluation metrics with zero_division to handle precision issues
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[ ]:


# Saving the final model in the recommended Keras format
model.save('first_dl_model.keras')


# In[ ]:


#To load the model later
model = tf.keras.models.load_model('<name>.keras')


# In[ ]:




