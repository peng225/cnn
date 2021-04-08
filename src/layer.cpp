#include "layer.h"
#include "utility.h"
#include <iostream>
#include <cassert>
#include <random>
#include <cmath>
#include <algorithm>

/* ======================
    Utility functions
   ======================*/
template <class X>
void normalize(std::vector<X>& weight, X& bias)
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
void normalize(std::vector<X>& weight, std::vector<X>& bias)
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
    return vec[x + y * width + (width * height) * channel];
}

void setValToVecMap(std::vector<float>& vec, int x, int y, int width, int height, int channel, float val)
{
    vec[x + y * width + (width * height) * channel] = val;
}

void addValToVecMap(std::vector<float>& vec, int x, int y, int width, int height, int channel, float val)
{
    vec[x + y * width + (width * height) * channel] += val;
}


/* ======================
    Layer
   ======================*/
Layer::Layer() : verbose(false)
{
}

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
    assert(windowSize <= inputSize.first + 2 * zeroPad);
    assert(windowSize <= inputSize.second + 2 * zeroPad);
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
                            auto w = getValFromVecMap(weight, winX, winY, windowSize, windowSize, inCh + numInputChannel * outCh);
                            auto inVal = getValFromVecMap(input, winX - zeroPad + outX, winY - zeroPad + outY, 
                                            inputSize.first, inputSize.second, inCh);
                            convVal += w * inVal;
                        }
                    }
                    addValToVecMap(output, outX, outY, outputSize.first, outputSize.second, outCh, convVal);
                }
            }
        }
        // Add bias to all entries
        for(int outY = 0; outY < outputSize.second; outY++){
            for(int outX = 0; outX < outputSize.first; outX++){
                addValToVecMap(output, outX, outY, outputSize.first, outputSize.second, outCh, bias[outCh]);
            }
        }
    }
    return output;
}

void ConvolutionLayer::initWeight()
{
    weight.resize(windowSize * windowSize * numInputChannel * numOutputChannel);

    std::random_device seedGen;
    std::mt19937 mt(seedGen());
    std::uniform_real_distribution<double> rd(-1.0, 1.0);
    for(auto& elem : weight){
        elem = rd(mt);
    }
    for(auto& elem : bias){
        elem = rd(mt);
    }
}

std::vector<float> ConvolutionLayer::updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError,
                double reduceRate)
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
                    setValToVecMap(dEdw, winX, winY, windowSize, windowSize, inCh + numInputChannel * outCh, sumVal);
                }
            }
        }
    }

    if(verbose) {
        std::cout << "Conv layer before weight:" << std::endl;
        printVector(weight);
        std::cout << "Conv layer dEdw:" << std::endl;
        printVector(dEdw);
        printVector(weight);
    }

    if(diffWeight.empty()) {
        diffWeight.resize(weight.size());
        diffBias.resize(bias.size());
    }
    for(int i = 0; i < windowSize * windowSize * numInputChannel * numOutputChannel; i++){
        diffWeight.at(i) -= reduceRate * GAMMA * dEdw.at(i);
        diffWeight.at(i) -= LAMBDA * reduceRate * GAMMA * weight.at(i);
    }

    if(verbose) {
        std::cout << "Conv layer after weight:" << std::endl;
        printVector(weight);
    }

    /* Update bias */
    if(verbose) {
        std::cout << "Conv layer before bias:" << std::endl;
        printVector(bias);
    }
    std::vector<float> dEdb(numOutputChannel);
    for(int outCh = 0; outCh < numOutputChannel; outCh++){
        for(int out = 0; out < outputSize.first * outputSize.second; out++){
            auto pe = getValFromVecMap(propError, out, 0, outputSize.first * outputSize.second, 1, outCh);
            dEdb.at(outCh) += pe;
        }
        diffBias.at(outCh) -= reduceRate * GAMMA * dEdb.at(outCh);
        diffBias.at(outCh) -= LAMBDA * reduceRate * GAMMA * bias.at(outCh);
    }
    if(verbose) {
        std::cout << "Conv layer after bias:" << std::endl;
        printVector(bias);
    }

    /* Next propError */
    std::vector<float> nextPropError(input.size());
    for(int inCh = 0; inCh < numInputChannel; inCh++){
        for(int inY = 0; inY < inputSize.second; inY++){
            for(int inX = 0; inX < inputSize.first; inX++){
                float sumVal = 0;
                for(int outCh = 0; outCh < numOutputChannel; outCh++){
                    for(int winY = 0; winY < windowSize; winY++){
                        if(inY - winY + zeroPad < 0 || outputSize.second <= inY - winY + zeroPad){
                            continue;
                        }
                        for(int winX = 0; winX < windowSize; winX++){
                            if(inX - winX + zeroPad < 0 || outputSize.first <= inX - winX + zeroPad){
                                continue;
                            }
                            auto pe = getValFromVecMap(propError,
                                        inX - winX + zeroPad, inY - winY + zeroPad,
                                        outputSize.first, outputSize.second, outCh);
                            auto w = getValFromVecMap(weight, winX, winY, windowSize, windowSize, inCh + numInputChannel * outCh);
                            sumVal += pe * w;
                        }
                    }
                }
                setValToVecMap(nextPropError, inX, inY, inputSize.first, inputSize.second, inCh, sumVal);
            }
        }
    }

    return nextPropError;
}

void ConvolutionLayer::dumpWeight() const
{
    for(int outCh = 0 ; outCh < numOutputChannel; outCh++){
        std::cout << "==== outCh: " << outCh << " ====" << std::endl;
        for(int inCh = 0 ; inCh < numInputChannel; inCh++){
            std::cout << "==== inCh: " << inCh << " ====" << std::endl;
            std::cout << "weight:" << std::endl;
            for(int winY = 0; winY < windowSize; winY++){
                for(int winX = 0; winX < windowSize; winX++){
                    std::cout << getValFromVecMap(weight, winX, winY, windowSize, windowSize, inCh + numInputChannel * outCh) << ", ";
                }
                std::cout << std::endl;
            }
        }
        std::cout << "bias:" << std::endl;
        std::cout << bias.at(outCh) << std::endl;
    }
}

void ConvolutionLayer::saveWeight(std::ofstream& ofs) const
{
    if(verbose){
        dumpWeight();
    }
    ofs << weight.size() << std::endl;
    for(auto w : weight){
        ofs << w << std::endl;
    }
    ofs << bias.size() << std::endl;
    for(auto b : bias){
        ofs << b << std::endl;
    }
}

void ConvolutionLayer::loadWeight(std::ifstream& ifs)
{
    std::string buf;
    if(std::getline(ifs, buf)){
        weight.resize(std::stof(buf));
    }else{
        std::cerr << "failed to load weight size" << std::endl;
        return;
    }
    for(auto& w : weight){
        if(std::getline(ifs, buf)){
            w = std::stof(buf);
        }else{
            std::cerr << "failed to load weight" << std::endl;
            return;
        }
    }

    if(std::getline(ifs, buf)){
        bias.resize(std::stoi(buf));
    }else{
        std::cerr << "failed to load bias size" << std::endl;
        return;
    }
    for(auto& b : bias){
        if(std::getline(ifs, buf)){
            b = std::stof(buf);
        }else{
            std::cerr << "failed to load bias" << std::endl;
            return;
        }
    }
    if(verbose) {
        dumpWeight();
    }
}

void ConvolutionLayer::flush()
{
    assert((diffWeight.empty() && diffBias.empty())
        || (!diffWeight.empty() && !diffBias.empty()));
    if(!diffWeight.empty()) {
        for(int i = 0; static_cast<size_t>(i) < weight.size(); i++) {
            weight.at(i) += diffWeight.at(i);
            assert(std::isfinite(weight.at(i)));
        }

        for(int i = 0; static_cast<size_t>(i) < bias.size(); i++) {
            bias.at(i) += diffBias.at(i);
            assert(std::isfinite(bias.at(i)));
        }

        diffWeight.clear();
        diffBias.clear();
    }
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
                const std::vector<float>& propError,
                double reduceRate)
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
    assert(windowSize <= inputSize.first + 2 * zeroPad);
    assert(windowSize <= inputSize.second + 2 * zeroPad);
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
                const std::vector<float>& propError,
                double reduceRate)
{
    assert(!propError.empty());
    assert(numInputChannel == numOutputChannel);
    /* Next propError */
    std::vector<float> nextPropError(input.size());
    for(int channel = 0; channel < numInputChannel; channel++){
        for(int outY = 0; outY < outputSize.second; outY++){
            for(int outX = 0; outX < outputSize.first; outX++){
                auto outVal = getValFromVecMap(output, outX, outY,
                                outputSize.first, outputSize.second, channel);
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
        output[out] = sumVal + bias;
    }
    return output;
}

void FullConnectLayer::initWeight()
{
    weight.resize(inputSize.first * inputSize.second
                * outputSize.first * outputSize.second * numInputChannel);
    std::random_device seedGen;
    std::mt19937 mt(seedGen());
    std::uniform_real_distribution<double> rd(-1.0,1.0);
    for(auto& elem : weight){
        elem = rd(mt);
    }
    bias = rd(mt);
}

std::vector<float> FullConnectLayer::updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError,
                double reduceRate)
{
    assert(!propError.empty());
    /* Update weight */
    std::vector<float> dEdw(input.size() * output.size());
    for(int out = 0; static_cast<size_t>(out) < output.size(); out++){
        for(int in = 0; static_cast<size_t>(in) < input.size(); in++){
            setValToVecMap(dEdw, in, out, input.size(), 1, 0, propError.at(out) * input.at(in));
        }
    }
    if(verbose) {
        std::cout << "FC layer dEdw:" << std::endl;
        printVector(dEdw);
    }

    if(verbose) {
        std::cout << "FC layer before weight:" << std::endl;
        printVector(weight);
    }
    if(diffWeight.empty()) {
        diffWeight.resize(weight.size());
        diffBias = 0;
    }
    for(int i = 0; static_cast<size_t>(i) < input.size() * output.size(); i++){
        diffWeight.at(i) -= reduceRate * GAMMA * dEdw.at(i);
        diffWeight.at(i) -= LAMBDA * reduceRate * GAMMA * weight.at(i);
    }


    /* Update bias */
    if(verbose) {
        std::cout << "FC layer before bias: " << bias << std::endl;
    }
    float dEdb = 0;
    for(const auto elem : propError){
        dEdb += elem;
    }
    diffBias -= reduceRate * GAMMA * dEdb;
    diffBias -= LAMBDA * reduceRate * GAMMA * bias;
    if(verbose) {
        std::cout << "FC layer after weight:" << std::endl;
        printVector(weight);
        std::cout << "FC layer after bias: " << bias << std::endl;
    }


    /* Next propError */
    std::vector<float> nextPropError(input.size());
    for(int out = 0; static_cast<size_t>(out) < output.size(); out++){
        for(int in = 0; static_cast<size_t>(in) < input.size(); in++){
            nextPropError.at(in) += propError.at(out) * getValFromVecMap(weight, in, out, input.size(), 1, 0);
        }
    }

    return nextPropError;
}

void FullConnectLayer::saveWeight(std::ofstream& ofs) const
{
    ofs << weight.size() << std::endl;
    for(auto w : weight){
        ofs << w << std::endl;
    }
    ofs << bias << std::endl;
}

void FullConnectLayer::loadWeight(std::ifstream& ifs)
{
    std::string buf;
    if(std::getline(ifs, buf)){
        weight.resize(std::stoi(buf));
    }else{
        std::cerr << "failed to load weight size" << std::endl;
        return;
    }
    for(auto& w : weight){
        if(std::getline(ifs, buf)){
            w = std::stof(buf);
        }else{
            std::cerr << "failed to load weight" << std::endl;
            return;
        }
    }

    if(std::getline(ifs, buf)){
        bias = std::stof(buf);
    }else{
        std::cerr << "failed to load bias" << std::endl;
        return;
    }
}

void FullConnectLayer::flush()
{
    if(!diffWeight.empty()) {
        for(int i = 0; static_cast<size_t>(i) < weight.size(); i++) {
            weight.at(i) += diffWeight.at(i);
            assert(std::isfinite(weight.at(i)));
        }

        bias += diffBias;
        assert(std::isfinite(bias));

        diffWeight.clear();
        diffBias = 0;
    }
}


/* ======================
    SoftmaxLayer
   ======================*/
void SoftmaxLayer::calcOutputSize()
{
    outputSize = inputSize;
    numOutputChannel = numInputChannel;
    assert(numInputChannel == 1);
    assert(numOutputChannel == 1);
}

std::vector<float> SoftmaxLayer::apply(const std::vector<float>& input) const
{
    assert(input.size() == static_cast<size_t>(inputSize.first * inputSize.second * numInputChannel));
    return softmax(input);
}

std::vector<float> SoftmaxLayer::updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError,
                double reduceRate)
{
    assert(!propError.empty());
    assert(input.size() == output.size());
    /* Next propError */
    std::vector<float> nextPropError(input.size());
    for(int in = 0; static_cast<size_t>(in) < input.size(); in++){
        nextPropError.at(in) = propError.at(in);
        for(int out = 0; static_cast<size_t>(out) < output.size(); out++){
            nextPropError.at(in) -= propError.at(out) * output.at(out);
        }
        nextPropError.at(in) *= output.at(in);
    }
    return nextPropError;
}

std::vector<float> SoftmaxLayer::softmax(const std::vector<float>& input) const
{
    assert(2 <= input.size());
    auto output = input;
    float expSum = 0;

    auto max = *std::max_element(std::begin(output), std::end(output));
    for(auto elem : output){
        assert(elem - max <= 0.0);
        expSum += exp(elem - max);
    }

    for(auto& elem : output){
        elem = exp(elem - max) / expSum;
        assert(std::isfinite(elem));
    }
    return output;
}

/* ======================
    SigmoidLayer
   ======================*/
void SigmoidLayer::calcOutputSize()
{
    outputSize = inputSize;
    numOutputChannel = numInputChannel;
    assert(numInputChannel == 1);
    assert(numOutputChannel == 1);
}

std::vector<float> SigmoidLayer::apply(const std::vector<float>& input) const
{
    assert(input.size() == static_cast<size_t>(inputSize.first * inputSize.second * numInputChannel));
    return sigmoid(input);
}

std::vector<float> SigmoidLayer::updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError,
                double reduceRate)
{
    assert(!propError.empty());
    /* Next propError */
    std::vector<float> nextPropError(input.size());
    for(int out = 0; static_cast<size_t>(out) < output.size(); out++){
        nextPropError.at(out) = propError.at(out)
                                * output.at(out) * (1 - output.at(out));
    }
    return nextPropError;
}

std::vector<float> SigmoidLayer::sigmoid(const std::vector<float>& input) const
{
    auto output = input;

    for(auto& elem : output){
        elem = 1 / (1 + exp(-elem));
    }
    return output;
}

/* ======================
    StandardizeLayer
   ======================*/
StandardizeLayer::StandardizeLayer(int nb) : numBatch(nb)
{
}

void StandardizeLayer::calcOutputSize()
{
    outputSize = inputSize;
    numOutputChannel = numInputChannel;
    assert(numInputChannel % numBatch == 0);
}

std::vector<float> StandardizeLayer::apply(const std::vector<float>& input) const
{
    assert(input.size() == static_cast<size_t>(inputSize.first * inputSize.second * numInputChannel));
    auto output = input;
    auto leftItr = std::begin(output);
    auto rightItr = std::begin(output);
    advance(rightItr, numBatch * inputSize.first * inputSize.second);
    for(int i = 0; i < numInputChannel/numBatch; i++) {
        standardize(leftItr, rightItr);
        leftItr = rightItr;
        advance(rightItr, numBatch * inputSize.first * inputSize.second);
    }
    assert(leftItr == std::end(output));
    return output;
}

std::vector<float> StandardizeLayer::updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError,
                double reduceRate)
{
    assert(!propError.empty());
    assert(propError.size() == input.size());
    assert(propError.size() == output.size());
    /* Next propError */
    std::vector<float> nextPropError;
    nextPropError = propError;

    auto leftItr = std::begin(nextPropError);
    auto rightItr = std::begin(nextPropError);
    advance(rightItr, numBatch * inputSize.first * inputSize.second);
    auto inputLeftItr = std::begin(input);
    auto inputRightItr = std::begin(input);
    advance(inputRightItr, numBatch * inputSize.first * inputSize.second);
    for(int i = 0; i < numInputChannel/numBatch; i++) {
        auto stddev = getStddev(inputLeftItr, inputRightItr,
                        getMean(inputLeftItr, inputRightItr));
        for(auto itr = leftItr; itr != rightItr; itr++) {
            *itr /= stddev;
            assert(std::isfinite(*itr));
        }
        leftItr = rightItr;
        advance(rightItr, numBatch * inputSize.first * inputSize.second);
        inputLeftItr = inputRightItr;
        advance(inputRightItr, numBatch * inputSize.first * inputSize.second);
    }
    assert(leftItr == std::end(nextPropError));
    assert(inputLeftItr == std::end(input));
    return nextPropError;
}

void StandardizeLayer::standardize(std::vector<float>::iterator leftItr,
        std::vector<float>::iterator rightItr) const
{
    auto mean = getMean(leftItr, rightItr);
    auto stddev = getStddev(leftItr, rightItr, mean);
    for(auto itr = leftItr; itr != rightItr; itr++) {
        *itr = (*itr - mean)/stddev;
        assert(std::isfinite(*itr));
    }
}

float StandardizeLayer::getMean(
        std::vector<float>::const_iterator leftItr,
        std::vector<float>::const_iterator rightItr) const
{
    float mean = 0;
    for(auto itr = leftItr; itr != rightItr; itr++) {
        mean += *itr;
    }
    return mean / distance(leftItr, rightItr);
}

float StandardizeLayer::getStddev(
        std::vector<float>::const_iterator leftItr,
        std::vector<float>::const_iterator rightItr, float mean) const
{
    assert(distance(leftItr, rightItr) > 1);
    float stddev = 0;
    for(auto itr = leftItr; itr != rightItr; itr++) {
        stddev += (*itr - mean) * (*itr - mean);
    }
    assert(stddev >= 0);
    return stddev <= 1e-10 ?
        1e-10 : sqrt(stddev / (distance(leftItr, rightItr) - 1));
}

