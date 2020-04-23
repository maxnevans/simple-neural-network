#pragma once

#include <vector>
#include <valarray>

namespace NN
{
    class NeuralNetwork
    {
    private:
        using vel = std::valarray<double>;

    public:
        const std::vector<int>& getLayers() const;
        std::vector<double> getWeights(size_t layerA, size_t layerB) const;
        void pushLayer(int countNeurons);
        void setupWeights(size_t layerA, size_t layerB, std::vector<double> weights);
        double train(std::vector<double> input, std::vector<double> answer);
        std::vector<double> classify(std::vector<double> input);
        void setLearningFactor(double factor);
        void clear();

        static std::vector<double> randomizeWeights(double lowerLimit, double higherLimit, int countNeuronsLayerA, int countNeuronsLayerB);

    protected:
        std::vector<vel> p_classify(std::vector<double> input);
        double p_applyActivationFunction(double value);
        vel p_applyActivationFunction(const vel& value);
        double p_applyActivationFunctionDerivative(double value);
        vel p_applyActivationFunctionDerivative(const vel& value);
        void p_backPropagation(double learningFactor, const std::vector<vel>& outputs, const vel& errors);
        vel p_calcOutputDeltas(const vel& inputs, const vel& error);
        vel p_calcDefaultDeltas(const vel& inputs, const vel& weights, const vel& deltas);
        vel p_transposeFlatMatrix(const vel& flatMatrix, int width, int height);
        vel p_calcGradientW(const vel& output, const vel& delta);

    private:
        double lFactor = 0.5;
        bool isInitialized = false;
        std::vector<int> layers;
        std::vector<vel> weights;
    };
}