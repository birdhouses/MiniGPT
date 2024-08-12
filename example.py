import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds

# Download NLTK data
nltk.download('punkt')

# Load IMDb dataset from TensorFlow Datasets
dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

# Convert dataset to lists of texts and labels
train_texts = []
train_labels = []
for text, label in tfds.as_numpy(dataset['train']):
    train_texts.append(text.decode('utf-8'))
    train_labels.append(label)

test_texts = []
test_labels = []
for text, label in tfds.as_numpy(dataset['test']):
    test_texts.append(text.decode('utf-8'))
    test_labels.append(label)

# Create DataFrame
train_data_tfds = pd.DataFrame({'text': train_texts, 'label': train_labels})
test_data_tfds = pd.DataFrame({'text': test_texts, 'label': test_labels})

# Prepare the dataset by concatenating the train and test sets
data = pd.concat([train_data_tfds, test_data_tfds], ignore_index=True)

# Remove special characters
data['text'] = data['text'].str.replace('[^a-zA-Z0-9\s]', '', regex=True)

# Tokenization
data['tokens'] = data['text'].apply(word_tokenize)

# Splitting Data into Training and Validation Sets
train_data, validation_data = train_test_split(data, test_size=0.2, random_state=42)

# Convert tokens to sequences and pad them
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_data['tokens'].tolist())
train_sequences = tokenizer.texts_to_sequences(train_data['tokens'].tolist())
validation_sequences = tokenizer.texts_to_sequences(validation_data['tokens'].tolist())

train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, padding='post')
validation_padded = tf.keras.preprocessing.sequence.pad_sequences(validation_sequences, padding='post')

# Define the neural network architecture
input_layer = Input(shape=(None,))
embedding_layer = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128)(input_layer)
lstm_layer = LSTM(128, return_sequences=True)(embedding_layer)
output_layer = Dense(len(tokenizer.word_index)+1, activation='softmax')(lstm_layer)

# Configure the model
model = Model(inputs=input_layer, outputs=output_layer)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_padded, train_padded, epochs=10, validation_data=(validation_padded, validation_padded))

# Validate the model
loss, accuracy = model.evaluate(validation_padded, validation_padded)
print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')
