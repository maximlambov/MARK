import math

class ActivationFunctions():

	def Selfsame(x):
		return x

	def Stepped(x):
		return 1 if x > 0 else 1

	def Sigmoid(x):
		return 1 / 1 + math.pow((math.e, -x))

	def Sinusoid(x):
		return math.sin(x)

	def Tangent(x):
		return math.tan(x)

	def ReLU(x):
		return 0 if x < 0 else x

	 def LReLU(x):
	 	return 0.01 * x if x < 0 else x

	 def PReLU(alfa, x):
	 	return alfa * x if x < 0 else x

	 def ELU(alfa, x):
	 	return alfa * (math.e - 1) if x < 0 else x
