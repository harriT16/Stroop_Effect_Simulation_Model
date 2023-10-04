import numpy as np
from keras.models import Sequential
from keras.layers import Dense

colors = {
    "Red": [1, 0, 0, 0],
    "Green": [0, 1, 0, 0],
    "Blue": [0, 0, 1, 0],
    "Yellow": [0, 0, 0, 1]
}

# Generate aligned pairs
aligned_data = [(color_vector, color_vector) for color_name, color_vector in colors.items()]

# Generate some Stroop effect pairs
stroop_data = [
    ([1, 0, 0, 0], [0, 0, 1, 0]),  # Red color, Blue word
    ([0, 1, 0, 0], [0, 0, 0, 1]),  # Green color, Yellow word
]

# Neural Network Model:
data = aligned_data * 9 + stroop_data  
np.random.shuffle(data)
X, y = zip(*data)
X = np.array(X)
y = np.array(y)

model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#training
model.fit(X, y, epochs=5000, verbose=0)

#testing
predictions = model.predict(X)
correct_predictions = np.argmax(predictions, axis=1) == np.argmax(y, axis=1)
accuracy = np.mean(correct_predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# To get entropy of predictions, which can give an insight into the model's confidence
import scipy.stats
entropies = [-np.sum(p * np.log2(p)) for p in predictions]
print("Average Entropy:", np.mean(entropies))

