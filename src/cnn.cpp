#include "cnn.h"
#include <iostream>
#include <cassert>

float getValFromVecMap(const std::vector<float>& vec, int x, int y, int width)
{
    return vec.at(x + y * width);
}

void setValToVecMap(std::vector<float>& vec, int x, int y, int width, float val)
{
    vec.at(x + y * width) = val;
}

/* ======================
    ConvolutionLayer
   ======================*/
ConvolutionLayer::ConvolutionLayer(int zeroPad, int windowSize) : 
    zeroPad(zeroPad), windowSize(windowSize)
{
    weight.resize(windowSize * windowSize);
}

void ConvolutionLayer::calcOutputSize()
{
    outputSize = DataSize(inputSize.first + 2*zeroPad - windowSize,
                        inputSize.second + 2*zeroPad - windowSize);
}

std::vector<float> ConvolutionLayer::apply(const std::vector<float>& input) const
{
    assert(input.size() == static_cast<size_t>(inputSize.first * inputSize.second));
    std::vector<float> output(outputSize.first * outputSize.second);
    for(int outY = 0; outY < outputSize.second; outY++){
        for(int outX = 0; outX < outputSize.first; outX++){
            float convVal = 0;
            for(int winY = 0; winY < windowSize; winY++){
                if(winY - zeroPad + outY < 0){
                    continue;
                }
                for(int winX = 0; winX < windowSize; winX++){
                    if(winX - zeroPad + outX < 0){
                        continue;
                    }
                    auto w = getValFromVecMap(weight, winX, winY, windowSize);
                    auto inVal = getValFromVecMap(input, winX - zeroPad + outX, 
                                    winY - zeroPad + outY, inputSize.first);
                    convVal += w * inVal;
                }
            }
            setValToVecMap(output, outX, outY, outputSize.first, convVal);
        }
    }
    return output;
}

/* ======================
    ReLULayer
   ======================*/
void ReLULayer::calcOutputSize()
{
    outputSize = inputSize;
}

std::vector<float> ReLULayer::apply(const std::vector<float>& input) const
{
    assert(input.size() == static_cast<size_t>(inputSize.first * inputSize.second));
    auto output = input;
    for(auto& elem : output){
        elem = elem >= 0 ? elem : 0;
    }
    return output;
}

/* ======================
    PoolingLayer
   ======================*/
void PoolingLayer::calcOutputSize()
{
    outputSize = DataSize(inputSize.first + 2*zeroPad - windowSize,
                        inputSize.second + 2*zeroPad - windowSize);
}

std::vector<float> PoolingLayer::apply(const std::vector<float>& input) const
{
    assert(input.size() == static_cast<size_t>(inputSize.first * inputSize.second));
    std::vector<float> output(outputSize.first * outputSize.second);
    for(int outY = 0; outY < outputSize.second; outY++){
        for(int outX = 0; outX < outputSize.first; outX++){
            float maxVal = 0;
            for(int winY = 0; winY < windowSize; winY++){
                if(winY - zeroPad + outY < 0){
                    continue;
                }
                for(int winX = 0; winX < windowSize; winX++){
                    if(winX - zeroPad + outX < 0){
                        continue;
                    }
                    auto inVal = getValFromVecMap(input, winX - zeroPad + outX, 
                                    winY - zeroPad + outY, inputSize.first);
                    if(maxVal <= inVal){
                        maxVal = inVal;
                    }
                }
            }
            setValToVecMap(output, outX, outY, outputSize.first, maxVal);
        }
    }
    return output;
}

/* ======================
    FullConnectLayer
   ======================*/
FullConnectLayer::FullConnectLayer(DataSize size)
{
    outputSize = size;
}

void FullConnectLayer::calcOutputSize()
{
    /* Do nothing */
}

std::vector<float> FullConnectLayer::apply(const std::vector<float>& input) const
{
    assert(input.size() == static_cast<size_t>(inputSize.first * inputSize.second));
    std::vector<float> output(outputSize.first * outputSize.second);
    assert(outputSize.second == 1);

    for(int out = 0; out < outputSize.first; out++){
        float sumVal = 0;
        for(int in = 0; in < inputSize.first * inputSize.second; in++){
            auto w = getValFromVecMap(weight, in, out, inputSize.first * inputSize.second);
            auto inVal = input.at(in);
            sumVal += w * inVal;
        }
        output.at(out) = sumVal;
    }
    return output;
}

void FullConnectLayer::initWeight()
{
    weight.resize(inputSize.first * inputSize.second
                * outputSize.first * outputSize.second);
}

/* ======================
    DeepNetwork
   ======================*/
// bool DeepNetwork::setInputInfo(int width, int height, int numChannel)
bool DeepNetwork::setInputInfo(DataSize size)
{
    // if(width <= 0 || height <= 0 || numChannel <= 0){
    if(size.first <= 0 || size.second <= 0){
        std::cerr << "ERROR: invalid parameter" << std::endl;
        return false;
    }

    inputSize = size;
    // this-> numChannel = numChannel;

    return true;
}

void DeepNetwork::addLayer(std::shared_ptr<Layer> layer)
{
    if(layers.empty()){
        layer->setInputSize(inputSize);
    }else{
        layer->setInputSize(layers.back()->getOutputSize());
    }
    layer->calcOutputSize();
    layer->initWeight();
    layers.push_back(layer);
}

std::vector<float> DeepNetwork::feedInput(const std::vector<float>& input) const
{
    auto output = input;
    for(auto& layer : layers){
        output = layer->apply(output);
    }
    return output;
}

