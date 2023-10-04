# Stroop_Effect_Simulation_Model
**SIMULATING THE STROOP EFFECT USING NEURAL NETWORKS:**

1. **Project overview:**

-The Stroop effect is a demonstration of interference in the reaction time of a task when the name of a color is printed in a different color that is not denoted by the name, (the word "red" is printed in blue), naming the color of the word may take longer and may cost error.

-This project simulates the Stroop effect using a neural network. We will train that model based on (color-word) pairs, and then examine its accuracy and response time in normal scenarios versus the Stroop-effect scenarios.

1. **Project Walkthrough:**

-First, we will set up our dataset, currently considering four colors: Red, Green, Blue, and Yellow. Each color will be represented as hot hot-encoded binary vector:

Red: [1, 0, 0, 0]

Green: [0, 1, 0,0]

Blue: [0, 0, 1, 0]

Yellow: [0, 0, 0, 1]

-Therefore, the word can also be one-hot encoded in the same form as an input-output pair:

Input: [1,0,0,0] (Red color)

Output:[1,0,0,0] (the word Red)

-The Stroop Scenario (the word "Red" in a Blue color) will be represented by:

Input:[0,0,1,0] (Blue color)

Output:[1,0,0,0] (The word Red)

-The Neural Network Model will have:

+4 neurons for the input Layer, representing our 4 colors.

+8 neurons for each hidden layer, right now having 2 hidden layers, included with ReLU activation)

+4 neurons for the output layer, representing 4 words, included with softmax activation

-We will train our network using aligned pairs, then will introduce a few Stroop scenarios to see if our model can adapt.

-Last, we will test the model's accuracy in predicting words for aligned colors. I examine the model's "confidence" in predicting words for Stroop scenarios by looking at the entropy of the output layer.

1. **How to use the code:**

1. Set up the environment:

Ensure that Python is installed.

1. Install packages and libraries:

This model requires "numpy", "keras", and "scipy" to run. Install by using:

pip install numpy keras scipy

1. Run the script:

Run the script by:

python (or python3) stroop\_simulation.py

1. The output:

After running the script, it will train a neural network on the provided data. Once the training is complete, the script will output:

+The accuracy of the model: This represents how many predictions the model got right, converted to percentages.

+The average entropy: this is a measure of uncertainty. If the entropy is high for Stroop scenarios, it shows the model's uncertainty in making decisions, showing the same confusion observed in humans due to the Stroop effect.

![](RackMultipart20231004-1-w7d3k0_html_730b91ee701bc60c.png)

**IV. Code Walkthrough:**

1. Data Preparation:

![](RackMultipart20231004-1-w7d3k0_html_33270bb7be27ec42.png)

We're creating a dictionary of colors that maps color names to their corresponding one-hot encoded vectors. One-hot encoding is a representation of categorical variables as binary vectors. Then we will align pairs where the word and the color match, like "Red" written in red. These pairs serve as our "easy" examples. Our stroop\_data are Stroop pairs where the word and the color are mismatched.

1. Neural Network Model:

![](RackMultipart20231004-1-w7d3k0_html_3063eb413464b3b2.png)

Here, we're defining the structure of our neural network: A Sequential model that allows us to build a linear stack of layers. The first hidden layer has 8 neurons with a ReLU activation function and takes the input of dimension 4. The second hidden layer also has 8 neurons with a ReLU activation function. The output layer has 4 neurons with a softmax activation, representing probability-like scores, ensuring the output total is 1. The last line is to compile the model and specifies categorical\_crossentropy as the loss function since this is a multi-class classification problem, and the adam optimizer is to adjust the network weights.

1. Training:

model.fit(X, y, epochs=5000, verbose=0)

Training the model using the data for 5000 epochs.

1. Testing:

![](RackMultipart20231004-1-w7d3k0_html_fd8b4de55e00729d.png)

After training, the model will predict the outputs for our input data, then we calculate the percentage of correct prediction to get the model accuracy and its entropy using the above function, then take the average for entropy and convert accuracy to percentage.

**V. Cognitive Science-Related Theme in the Project:**

1. Cognitive Interference:

When presented with conflicting words such as the word Red but in Blue color, our cognitive system faces interference between the different pieces of information it's processing. In the model simulation, the neural network experienced higher uncertainty (the entropy) when receiving conflict inputs, showcasing this interference computationally.

1. Dual-process Theory:

One of the theories explaining the Stroop effect involves the dual-process theory. We have two types of cognitive processes: one automatic and one controlled, when these processes conflict, it takes us longer to respond since our brain has a delay in analyzing these conflicting information. Similarly, the model has to learn to "read" the color input and "say" the color name automatically but faces difficulty (increased entropy) when these don't align.

1. Learning:

The model training phase can be thought of as a learning process. This is the same as how humans learn to read or identify colors quickly – through repetition and feedback over time. However, as humans might fall into the Stroop scenarios due to the conflicting information, the network also shows different performance on "Stroop" versus normal data points.

1. Entropy as Uncertainty:

In the project, entropy was used as a measure of the network's uncertainty about its predictions. Entropy has been discussed in cognitive science as concerning decision-making, information processing, etc...In the project, it acts as a "proxy" for the difficulty experienced by the model.

1. Adaptation and efficiency:

The neural model was trained on aligned color-word pairs, much like how we are used to seeing words and their color. When presented with conflicting Stroop pairs, the efficiency of the model drops, showing the parallel in the idea that cognitive systems are optimized for "usual" scenarios and can get tripped up by "unusual" or conflicting ones.

Citation:

[https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)

https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
