import numpy as np
import scipy.special

class Core():
    def __init__(self, numberOfInputs, hiddenLayers, numberOfOutputs, learningSpeed):
        self.numberOfInputs = numberOfInputs  # количество входов
        self.hiddenLayers = hiddenLayers  # количество нейронов в скрытых слоях, пример [50,25,50] (3 слоя по 50 и 25 нейронов)
        self.numberOfOutputs = numberOfOutputs  # количество выходов
        self.learningSpeed = learningSpeed  # скорость обучения
        self.hiddenLayers.insert(0, numberOfInputs)  # добавляем количество входов в список
        self.hiddenLayers.append(numberOfOutputs)  # добавляем количество выходов в список
        self.weights = np.array(list(np.random.random((hiddenLayers[x + 1], hiddenLayers[x])) for x in
                                     range(len(hiddenLayers) - 1)))  # инициализируем веса для каждого слоя
        self.errors = []
        self.signals = []
        self.ActivationFunction = lambda x: scipy.special.expit(x)  # сигмойда

    def Query(self, inputsData):
        signals = np.array(inputsData, ndmin=2).T
        for weights in self.weights:
            signals = self.ActivationFunction(np.dot(weights, signals))
        return signals

    def Train(self, inputsData, outputsData):
        self.errors = []
        self.signals = []

        signals = np.array(inputsData, ndmin=2).T  # преобразуем входные данные в двумерный массив
        self.signals.append(np.array(signals))
        outputsData = np.array(outputsData, ndmin=2).T  # преобразуем исходящие данные в двумерный массив

        for weight in self.weights:
            signals = self.ActivationFunction(np.dot(weight, signals))
            self.signals.append(np.array(signals))

        errors = outputsData - signals  # находим ошибки
        self.errors.append(np.array(errors))
        weightsLength = len(self.weights) - 1
        for x in range(weightsLength):
            errors = np.dot(self.weights[(weightsLength) - x].T, errors)
            self.errors.append(np.array(errors))

        for x in range(len(self.weights)):
            self.weights[weightsLength - x] += self.learningSpeed * np.dot(
                (self.errors[x] * self.signals[weightsLength - x + 1] * (1.0 - self.signals[weightsLength - x + 1])),
                np.transpose(self.signals[weightsLength - x]))


if __name__ == "__main__":
    print("methods of this class cannot be called directly")