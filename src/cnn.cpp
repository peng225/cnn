#include "cnn.h"
#include <iostream>
#include <cassert>
#include <iterator>
#include <random>
#include <cmath>

/* ======================
    Utility functions
   ======================*/
template <class X>
void printVector(const std::vector<X>& vec)
{
    for(auto elem : vec){
        std::cout << elem << ", ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

template <class X>
void L2Normalization(std::vector<X>& weight, X& bias)
{
    float normalizer = 0;
    for(auto elem : weight){
        normalizer += elem * elem; 
    }
    normalizer += bias * bias;

    normalizer = sqrt(normalizer);
    for(auto& elem : weight){
        elem /= normalizer;
    }
    bias /= normalizer;
}

template <class X>
void L2Normalization(std::vector<X>& weight, std::vector<X>& bias)
{
    float normalizer = 0;
    for(auto elem : weight){
        normalizer += elem * elem; 
    }
    for(auto elem : bias){
        normalizer += elem * elem; 
    }

    normalizer = sqrt(normalizer);
    for(auto& elem : weight){
        elem /= normalizer;
    }
    for(auto& elem : bias){
        elem /= normalizer;
    }
}

float getValFromVecMap(const std::vector<float>& vec, int x, int y, int width, int height, int channel)
{
    return vec.at(x + y * width + (width * height) * channel);
}

void setValToVecMap(std::vector<float>& vec, int x, int y, int width, int height, int channel, float val)
{
    vec.at(x + y * width + (width * height) * channel) = val;
}

/* ======================
    Layer
   ======================*/
void Layer::setInputInfo(const DataSize& size, int numInputChannel)
{
    inputSize = size;
    this->numInputChannel = numInputChannel;
}

/* ======================
    ConvolutionLayer
   ======================*/
ConvolutionLayer::ConvolutionLayer(int zeroPad, int windowSize, int numOutputChannel) : 
    zeroPad(zeroPad), windowSize(windowSize)
{
    this->numOutputChannel = numOutputChannel;
    bias.resize(numOutputChannel);
}

void ConvolutionLayer::calcOutputSize()
{
    outputSize = DataSize(inputSize.first + 2 * zeroPad - windowSize + 1,
                        inputSize.second + 2 * zeroPad - windowSize + 1);
}

std::vector<float> ConvolutionLayer::apply(const std::vector<float>& input) const
{
    assert(input.size() == static_cast<size_t>(inputSize.first * inputSize.second * numInputChannel));
    std::vector<float> output(outputSize.first * outputSize.second * numOutputChannel);
    for(int outCh = 0; outCh < numOutputChannel; outCh++){
        for(int inCh = 0; inCh < numInputChannel; inCh++){
            for(int outY = 0; outY < outputSize.second; outY++){
                for(int outX = 0; outX < outputSize.first; outX++){
                    float convVal = 0;
                    for(int winY = 0; winY < windowSize; winY++){
                        if(winY - zeroPad + outY < 0 || inputSize.second <= winY - zeroPad + outY){
                            continue;
                        }
                        for(int winX = 0; winX < windowSize; winX++){
                            if(winX - zeroPad + outX < 0 || inputSize.first <= winX - zeroPad + outX){
                                continue;
                            }
                            auto w = getValFromVecMap(weight, winX, winY, windowSize, windowSize, inCh + inCh * outCh);
                            auto inVal = getValFromVecMap(input, winX - zeroPad + outX, winY - zeroPad + outY, 
                                            inputSize.first, inputSize.second, inCh);
                            convVal += w * inVal;
                        }
                    }
                    float currentOutVal = getValFromVecMap(output, outX, outY, outputSize.first, outputSize.second, outCh);
                    setValToVecMap(output, outX, outY, outputSize.first, outputSize.second, outCh, currentOutVal + convVal);
                }
            }
        }
        // Add bias to all entries
        for(int outY = 0; outY < outputSize.second; outY++){
            for(int outX = 0; outX < outputSize.first; outX++){
                float currentOutVal = getValFromVecMap(output, outX, outY, outputSize.first, outputSize.second, outCh);
                setValToVecMap(output, outX, outY, outputSize.first, outputSize.second, outCh, currentOutVal + bias.at(outCh));
            }
        }
    }
    return output;
}

void ConvolutionLayer::initWeight()
{
    weight.resize(windowSize * windowSize * numInputChannel * numOutputChannel);

    std::mt19937 mt(0);
    std::uniform_real_distribution<double> rd(0.0,1.0);
    for(auto& elem : weight){
        elem = rd(mt);
    }
    L2Normalization(weight, bias);
}

std::vector<float> ConvolutionLayer::updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError)
{
    assert(!propError.empty());
    /* Update weight */
    std::vector<float> dEdw(windowSize * windowSize * numInputChannel * numOutputChannel);
    for(int outCh = 0; outCh < numOutputChannel; outCh++){
        for(int inCh = 0; inCh < numInputChannel; inCh++){
            for(int winY = 0; winY < windowSize; winY++){
                for(int winX = 0; winX < windowSize; winX++){
                    float sumVal = 0;
                    for(int outY = 0; outY < outputSize.second; outY++){
                        if(winY - zeroPad + outY < 0 || inputSize.second <= winY - zeroPad + outY){
                            continue;
                        }
                        for(int outX = 0; outX < outputSize.first; outX++){
                            if(winX - zeroPad + outX < 0 || inputSize.second <= winX - zeroPad + outX){
                                continue;
                            }
                            auto pe = getValFromVecMap(propError, outX, outY,
                                        outputSize.first, outputSize.second, outCh);
                            auto inVal = getValFromVecMap(input, winX - zeroPad + outX, winY -zeroPad + outY,
                                        inputSize.first, inputSize.second, inCh);
                            sumVal += pe * inVal;
                        }
                    }
                    setValToVecMap(dEdw, winX, winY, windowSize, windowSize, inCh + inCh * outCh, sumVal);
                }
            }
        }
    }

    for(int i = 0; i < windowSize * windowSize * numInputChannel * numOutputChannel; i++){
        weight.at(i) -= GAMMA * dEdw.at(i);
    }


    /* Update bias */
    std::vector<float> dEdb(numOutputChannel);
    for(int outCh = 0; outCh < numOutputChannel; outCh++){
        for(int out; out < outputSize.first * outputSize.second; out++){
            auto pe = getValFromVecMap(propError, out, 0, outputSize.first * outputSize.second, 1, outCh);
            dEdb.at(outCh) += pe;
        }
        bias.at(outCh) -= GAMMA * dEdb.at(outCh);
    }
    L2Normalization(weight, bias);


    /* Next propError */
    std::vector<float> nextPropError(input.size());
    // TODO: implementation
    //for(int out = 0; static_cast<size_t>(out) < output.size(); out++){
    //    for(int in = 0; static_cast<size_t>(in) < input.size(); in++){
    //        nextPropError.at(in) += propError.at(out) * getValFromVecMap(weight, in, out, input.size());
    //    }
    //}

    return nextPropError;

}

/* ======================
    ReLULayer
   ======================*/
void ReLULayer::calcOutputSize()
{
    outputSize = inputSize;
    numOutputChannel = numInputChannel;
}

std::vector<float> ReLULayer::apply(const std::vector<float>& input) const
{
    assert(input.size() == static_cast<size_t>(inputSize.first * inputSize.second * numInputChannel));
    auto output = input;
    for(auto& elem : output){
        elem = elem >= 0 ? elem : 0;
    }
    return output;
}

std::vector<float> ReLULayer::updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError)
{
    assert(!propError.empty());
    /* Next propError */
    std::vector<float> nextPropError(input.size());
    for(int out = 0; static_cast<size_t>(out) < output.size(); out++){
        nextPropError.at(out) = output.at(out) == 0 ? 0 : propError.at(out);
    }
    return nextPropError;
}

/* ======================
    PoolingLayer
   ======================*/
void PoolingLayer::calcOutputSize()
{
    outputSize = DataSize(inputSize.first + 2*zeroPad - windowSize + 1,
                        inputSize.second + 2*zeroPad - windowSize + 1);
    numOutputChannel = numInputChannel;
}

std::vector<float> PoolingLayer::apply(const std::vector<float>& input) const
{
    assert(input.size() == static_cast<size_t>(inputSize.first * inputSize.second) * numInputChannel);
    std::vector<float> output(outputSize.first * outputSize.second * numOutputChannel);
    for(int channel = 0; channel < numInputChannel; channel++){
        for(int outY = 0; outY < outputSize.second; outY++){
            for(int outX = 0; outX < outputSize.first; outX++){
                float maxVal = 0;
                for(int winY = 0; winY < windowSize; winY++){
                    if(winY - zeroPad + outY < 0 || inputSize.second <= winY - zeroPad + outY){
                        continue;
                    }
                    for(int winX = 0; winX < windowSize; winX++){
                        if(winX - zeroPad + outX < 0 || inputSize.first <= winX - zeroPad + outX){
                            continue;
                        }
                        auto inVal = getValFromVecMap(input, winX - zeroPad + outX, winY - zeroPad + outY, 
                                        inputSize.first, inputSize.second, channel);
                        if(maxVal <= inVal){
                            maxVal = inVal;
                        }
                    }
                }
                setValToVecMap(output, outX, outY, outputSize.first, outputSize.second,
                                channel, maxVal);
            }
        }
    }
    return output;
}

std::vector<float> PoolingLayer::updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError)
{
    assert(!propError.empty());
    assert(numInputChannel == numOutputChannel);
    /* Next propError */
    std::vector<float> nextPropError(input.size());
    for(int channel = 0; channel < numInputChannel; channel++){
        for(int outY = 0; outY < outputSize.second; outY++){
            for(int outX = 0; outX < outputSize.first; outX++){
                for(int winY = 0; winY < windowSize; winY++){
                    if(winY - zeroPad + outY < 0 || inputSize.second <= winY - zeroPad + outY){
                        continue;
                    }
                    auto outVal = getValFromVecMap(output, outX, outY,
                                    outputSize.first, outputSize.second, channel);
                    for(int winX = 0; winX < windowSize; winX++){
                        if(winX - zeroPad + outX < 0 || inputSize.first <= winX - zeroPad + outX){
                            continue;
                        }
                        auto inVal = getValFromVecMap(input, winX - zeroPad + outX, winY - zeroPad + outY,
                                        inputSize.first, inputSize.second, channel);
                        if(inVal == outVal){
                            auto pe = getValFromVecMap(propError, outX, outY,
                                        outputSize.first, outputSize.second, channel); 
                            auto currentNpe = getValFromVecMap(nextPropError, winX - zeroPad + outX, winY - zeroPad + outY,
                                        inputSize.first, inputSize.second, channel); 
                            setValToVecMap(nextPropError, winX - zeroPad + outX, winY - zeroPad + outY,
                                        inputSize.first, inputSize.second, channel, pe + currentNpe);
                        }
                    }
                }
            }
        }
    }

    return nextPropError;
}

/* ======================
    FullConnectLayer
   ======================*/
FullConnectLayer::FullConnectLayer(DataSize size) : bias(0)
{
    outputSize = size;
    numOutputChannel = 1;
}

void FullConnectLayer::calcOutputSize()
{
    /* Do nothing */
}

std::vector<float> FullConnectLayer::apply(const std::vector<float>& input) const
{
    assert(input.size() == static_cast<size_t>(inputSize.first * inputSize.second * numInputChannel));
    std::vector<float> output(outputSize.first * outputSize.second);
    assert(outputSize.second == 1);
    assert(numOutputChannel == 1);

    for(int out = 0; out < outputSize.first; out++){
        float sumVal = 0;
        for(int channel = 0; channel < numInputChannel; channel++){
            for(int in = 0; in < inputSize.first * inputSize.second; in++){
                auto w = getValFromVecMap(weight, in, out, inputSize.first * inputSize.second, 1, channel);
                auto inVal = getValFromVecMap(input, in, 0, inputSize.first * inputSize.second, 1, channel);
                sumVal += w * inVal;
            }
        }
        output.at(out) = sumVal + bias;
    }
    return output;
}

void FullConnectLayer::initWeight()
{
    weight.resize(inputSize.first * inputSize.second
                * outputSize.first * outputSize.second * numInputChannel);
    std::mt19937 mt(0);
    std::uniform_real_distribution<double> rd(0.0,1.0);
    for(auto& elem : weight){
        elem = rd(mt);
    }
    L2Normalization(weight, bias);
}

std::vector<float> FullConnectLayer::updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError)
{
    assert(!propError.empty());
    /* Update weight */
    std::vector<float> dEdw(input.size() * output.size());
    for(int out = 0; static_cast<size_t>(out) < output.size(); out++){
        for(int in = 0; static_cast<size_t>(in) < input.size(); in++){
            setValToVecMap(dEdw, in, out, input.size(), propError.at(out) * input.at(in), 1, 0);
        }
    }
    std::cout << "FC layer dEdw:" << std::endl;
    printVector(dEdw);


    std::cout << "FC layer before weight:" << std::endl;
    printVector(weight);
    for(int i = 0; static_cast<size_t>(i) < input.size() * output.size(); i++){
        weight.at(i) -= GAMMA * dEdw.at(i);
    }


    /* Update bias */
    std::cout << "FC layer before bias: " << bias << std::endl;
    float dEdb = 0;
    for(const auto elem : propError){
        dEdb += elem;
    }
    bias -= GAMMA * dEdb;
    L2Normalization(weight, bias);
    std::cout << "FC layer after weight:" << std::endl;
    printVector(weight);
    std::cout << "FC layer after bias: " << bias << std::endl;


    /* Next propError */
    std::vector<float> nextPropError(input.size());
    for(int out = 0; static_cast<size_t>(out) < output.size(); out++){
        for(int in = 0; static_cast<size_t>(in) < input.size(); in++){
            nextPropError.at(in) += propError.at(out) * getValFromVecMap(weight, in, out, input.size(), 1, 0);
        }
    }

    return nextPropError;
}

/* ======================
    DeepNetwork
   ======================*/
bool DeepNetwork::setInputInfo(DataSize size, int numChannel)
{
    if(size.first <= 0 || size.second <= 0){
        std::cerr << "ERROR: invalid parameter" << std::endl;
        return false;
    }

    inputSize = size;
    numInputChannel = numChannel;

    return true;
}

void DeepNetwork::addLayer(std::shared_ptr<Layer> layer)
{
    if(layers.empty()){
        layer->setInputInfo(inputSize, numInputChannel);
    }else{
        layer->setInputInfo(layers.back()->getOutputSize(), layers.back()->getNumOutputChannel());
    }
    layer->calcOutputSize();
    layer->initWeight();
    layers.push_back(layer);
}

std::vector<std::vector<float>> DeepNetwork::feedInput(const std::vector<float>& input) const
{
    std::vector<std::vector<float>> outputs;
    outputs.reserve(layers.size() + 1);
    outputs.push_back(input);
    for(auto& layer : layers){
        outputs.push_back(layer->apply(outputs.back()));
    }
    return outputs;
}

void DeepNetwork::backPropagate(const std::vector<float>& input, const std::vector<float>& correctOutput)
{
    auto outputs = feedInput(input);
    std::vector<float> propError(outputs.back().size());
    assert(outputs.back().size() == correctOutput.size());

    for(int i = 0; static_cast<size_t>(i) < outputs.back().size(); i++){
        propError.at(i) = outputs.back().at(i) - correctOutput.at(i);
    }

    std::cout << "outputs" << std::endl;
    for(auto output : outputs){
        printVector(output);
    }

    std::cout << "Initial propError:" << std::endl;
    printVector(propError);

    int index = outputs.size() - 1;
    assert(outputs.size() - 1 == layers.size());

    for(auto layer = std::rbegin(layers); layer != std::rend(layers); layer++){
        propError = (*layer)->updateWeight(outputs.at(index - 1), outputs.at(index), propError);
        index--;
    }
}

