# Mark is a lightweight library for machine learning in python.

Use MARK if you need a deep learning library that:

It allows you to quickly and easily create neural networks for simple tasks.
- Supports multi-layer direct distribution networks.
- Runs on CPU.

Mark is compatible with Python 3.5 and latest version.

## Getting started: 30 seconds to MARK

```python
from NeuralNetwork import NeuralNetwork
"""
Network initialization
The arguments to pass:
-- number of input neurons
-- number of layers and neurons
   [4] - 4 neurons in the hidden layer
-- number of output neurons
-- learning rate
"""

network = NeuralNetwork(2, [4], 1, 0.3)

#Network training
for x in range(60000):
  network.Train([0.0, 0.0],[0.0])
  network.Train([0.0, 1.0],[0.0])
  network.Train([1.0, 0.0],[0.0])
  network.Train([1.0, 1.0],[1.0])
		
# Network survey
print("{0}".format(network.Query([0.0, 0.0])))
print("{0}".format(network.Query([0.0, 1.0])))
print("{0}".format(network.Query([1.0, 0.0])))
print("{0}".format(network.Query([1.0, 1.0])))
```
### The result of the work of the network
```
[[0.00407605]]
[[0.0043606]]
[[0.00420513]]
[[0.99276879]]
```

## Installation

```
Install Mark from the GitHub source:

First, clone Mark using git:

git clone https://github.com/maxilamb/MARK.git

Then, cd to the Mark folder and run the install command:

cd mark

create your python file for interaction with core.py

run command  pip install numpy, scipy
```
