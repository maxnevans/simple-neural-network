#include "pch.h"
#include "NeuralNetwork.h"


#include <numeric>
#include <execution>
#include <random>
#include <cassert>
#define expect(x) assert(x)

namespace NN
{
    const std::vector<int>& NeuralNetwork::getLayers() const
    {
        return layers;
    }
    std::vector<double> NeuralNetwork::getWeights(size_t layerA, size_t layerB) const
    {
        expect(layerA >= 0);
        expect(layerB >= 0);
        expect(layerB - layerA == 1);
        expect(layerB < layers.size());

        auto ilWeights = weights.at(layerA);

        return std::vector<double>(std::begin(ilWeights), std::end(ilWeights));
    }
    void NeuralNetwork::pushLayer(int countNeurons)
    {
        layers.push_back(countNeurons);

        if (layers.size() > 1)
            weights.emplace_back();
    }
    void NeuralNetwork::setupWeights(size_t layerA, size_t layerB, std::vector<double> weights)
    {
        expect(layerA >= 0);
        expect(layerB >= 0);
        expect(layerB - layerA == 1);
        expect(layerB < layers.size());
        expect(weights.size() > 0);
        expect(this->weights.size() >= layerA);
        expect(layers[layerA] * layers[layerB] == weights.size());

        this->weights[layerA] = vel(weights.data(), weights.size());
    }
    double NeuralNetwork::train(std::vector<double> input, std::vector<double> ans)
    {
        expect(input.size() > 0);
        expect(layers.size() > 1);
        expect(input.size() == layers[0]);
        expect(layers.back() == ans.size());

        auto outputs = p_classify(input);
        
        vel answer(ans.data(), ans.size());
        vel error = answer - outputs.back();

        p_backPropagation(lFactor, outputs, error);

        error = std::pow(error, 2);
        return std::reduce(std::begin(error), std::end(error)) / ans.size();
    }
    std::vector<double> NeuralNetwork::classify(std::vector<double> input)
    {
        auto output = p_classify(input).back();
        return std::vector<double>(std::begin(output), std::end(output));
    }
    void NeuralNetwork::setLearningFactor(double factor)
    {
        expect(factor > 0);
        expect(factor <= 1);
        lFactor = factor;
    }
    void NeuralNetwork::clear()
    {
        isInitialized = false;
        layers.clear();
        weights.clear();
    }
    std::vector<double> NeuralNetwork::randomizeWeights(double lowerLimit, double highterLimit, int countNeuronsLayerA, int countNeuronsLayerB)
    {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<double> dist(lowerLimit, highterLimit);

        std::vector<double> output;
        output.reserve(countNeuronsLayerA * countNeuronsLayerB);

        for (int i = 0; i < countNeuronsLayerA * countNeuronsLayerB; i++)
            output.emplace_back(dist(generator));

        return output;
    }
    std::vector<NeuralNetwork::vel> NeuralNetwork::p_classify(std::vector<double> inp)
    {
        expect(layers.size() > 1);
        expect(inp.size() == layers[0]);
        expect(weights.size() == layers.size() - 1);

        std::vector<vel> outputs;
        outputs.resize(layers.size());
        for (size_t i = 0; i < layers.size(); i++)
            outputs[i].resize(layers[i]);

        outputs[0] = vel(inp.data(), inp.size());

        for (size_t i = 0; i < weights.size(); i++)
        {
            /* Loop through layers */
            std::vector<double> nextLayerOutput;

            for (int j = 0; j < layers[i + 1]; j++)
            {
                /* Loop through each neuron on next level */
                vel nextLayerNeuronWeights = weights[i][std::slice(j * layers[i], layers[i], 1)];
                nextLayerNeuronWeights *= outputs[i];
                double nextLayerNeuronInputValue = std::reduce(std::execution::par_unseq, std::begin(nextLayerNeuronWeights), std::end(nextLayerNeuronWeights));
                double nextLayerNeuronOutput = p_applyActivationFunction(nextLayerNeuronInputValue);
                nextLayerOutput.push_back(nextLayerNeuronOutput);
            }
            outputs[i + 1] = std::valarray<double>(nextLayerOutput.data(), nextLayerOutput.size());
        }

        return outputs;
    }
    NeuralNetwork::vel NeuralNetwork::p_applyActivationFunctionDerivative(const vel& value)
    {
        return value * (1 - value);
    }
    double NeuralNetwork::p_applyActivationFunction(double value)
    {
        return 1 / (1 + std::exp(-value));
    }
    NeuralNetwork::vel NeuralNetwork::p_applyActivationFunction(const vel& value)
    {
        return 1 / (1 + std::exp(-value));
    }
    double NeuralNetwork::p_applyActivationFunctionDerivative(double value)
    {
        return value * (1.0 - value);
    }
    void NeuralNetwork::p_backPropagation(double learningFactor, const std::vector<vel>& outputs, const vel& errors)
    {
        vel prevDeltas = p_calcOutputDeltas(outputs.back(), errors);
        for (size_t i = layers.size() - 2; i > 0; i--)
        {
            vel gradW = p_calcGradientW(outputs[i], prevDeltas);
            vel deltaW = gradW * learningFactor;
            this->weights[i] += deltaW;

            vel deltas = p_calcDefaultDeltas(outputs[i], this->weights[i], prevDeltas);
            prevDeltas = deltas;
        }
    }
    NeuralNetwork::vel NeuralNetwork::p_calcOutputDeltas(const vel& outputs, const vel& errors)
    {
        return errors * p_applyActivationFunctionDerivative(outputs);
    }
    NeuralNetwork::vel NeuralNetwork::p_calcDefaultDeltas(const vel& outputs, const vel& weights, const vel& deltas)
    {
        auto currentLayerCountNeurons = outputs.size();
        auto nextLayerCountNeurons = weights.size() / outputs.size();

        auto tw = p_transposeFlatMatrix(weights, currentLayerCountNeurons, nextLayerCountNeurons);
        vel sums;
        sums.resize(currentLayerCountNeurons);

        for (size_t i = 0; i < currentLayerCountNeurons; i++)
        {
            auto test = vel(tw[std::slice(i * nextLayerCountNeurons, nextLayerCountNeurons, 1)]);
            test *= deltas;
            sums[i] = std::reduce(std::execution::par_unseq, std::begin(test), std::end(test));
        }

        return p_applyActivationFunctionDerivative(outputs) * sums;
    }
    NeuralNetwork::vel NeuralNetwork::p_transposeFlatMatrix(const vel& flatMatrix, int width, int height)
    {
        expect(flatMatrix.size() >= 0);
        expect(width >= 0);
        expect(height >= 0);
        expect(flatMatrix.size() == width * height);

        auto t = flatMatrix;

        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                t[j * height + i] = flatMatrix[i * width + j];

        return t;
    }
    NeuralNetwork::vel NeuralNetwork::p_calcGradientW(const vel& output, const vel& delta)
    {
        vel gradW(output.size() * delta.size());

        for (size_t i = 0; i < output.size(); i++)
        {
            gradW[std::slice(i * delta.size(), delta.size(), 1)] = output[i] * delta;
        }

        return gradW;
    }
}