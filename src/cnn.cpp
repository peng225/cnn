#include "cnn.h"
#include "utility.h"
#include <iostream>
#include <cassert>
#include <iterator>

/* ======================
    DeepNetwork
   ======================*/
DeepNetwork::DeepNetwork()
    : minibatchSize(1), inputCount(0), lossFunc(LossFunction::MSE)
{
}

DeepNetwork::DeepNetwork(int mbSize)
    : minibatchSize(mbSize), inputCount(0), lossFunc(LossFunction::MSE)
{
}

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
    layers.emplace_back(layer);
}

std::vector<std::vector<float>> DeepNetwork::feedInput(const std::vector<float>& input) const
{
    std::vector<std::vector<float>> outputs;
    outputs.reserve(layers.size() + 1);
    outputs.emplace_back(input);
    for(auto& layer : layers){
        outputs.emplace_back(layer->apply(outputs.back()));
    }
    return outputs;
}

void DeepNetwork::backPropagate(const std::vector<float>& input, const std::vector<float>& correctOutput, double reduceRate, bool verbose)
{
    assert(0 < reduceRate && reduceRate <= 1.0);
    assert(inputCount < minibatchSize);

    auto outputs = feedInput(input);
    std::vector<float> propError(outputs.back().size());
    assert(outputs.back().size() == correctOutput.size());

    switch(lossFunc) {
    case LossFunction::MSE:
        for(int i = 0; static_cast<size_t>(i) < outputs.back().size();
                i++){
            propError.at(i) = outputs.back().at(i) - correctOutput.at(i);
        }
        break;
    case LossFunction::CRS_ENT:
        // 損失関数: -y_c log(y) - (1-y_c)log(1-y)
        // 微分: -y_c/y + (1-y_c)/(1-y)
        for(int i = 0; static_cast<size_t>(i) < outputs.back().size();
                i++){
            float divisor1, divisor2;
            assert(0 <= outputs.back().at(i)
                && outputs.back().at(i) <= 1.0);
            if(correctOutput.at(i) != 0.0) {
                divisor1 = std::max(1e-5F, outputs.back().at(i));
                propError.at(i) -= correctOutput.at(i) / divisor1;
            }
            if(correctOutput.at(i) != 1.0) {
                divisor2 = std::max(1e-5F, 1 - outputs.back().at(i));
                propError.at(i) += (1.0 - correctOutput.at(i)) / divisor2;
            }
        }
        break;
    default:
        std::cerr << "Invalid loss function." << std::endl;
        std::exit(1);
    }

    if(verbose) {
        std::cout << "outputs" << std::endl;
        for(auto output : outputs){
            assert(!output.empty());
            printVector(output);
        }

        std::cout << "Initial propError:" << std::endl;
        printVector(propError);
    }

    int index = outputs.size() - 1;
    assert(outputs.size() - 1 == layers.size());

    for(auto layer = std::rbegin(layers); layer != std::rend(layers); layer++){
        propError = (*layer)->updateWeight(outputs.at(index - 1), outputs.at(index), propError, reduceRate);
        if(verbose) {
            std::cout << "Next propError:" << std::endl;
            printVector(propError);
        }
        index--;
    }

    inputCount++;
    if(inputCount == minibatchSize) {
        inputCount = 0;
        flush();
    }
}


void DeepNetwork::saveWeight(std::string filename) const
{
    std::ofstream ofs(filename);
    if(ofs.fail()){
        std::cerr << "failed to open file " << filename << std::endl;
        return;
    }

    for(const auto& layer : layers){
        layer->saveWeight(ofs);
    }
}

void DeepNetwork::loadWeight(std::string filename)
{
    std::ifstream ifs(filename);
    if(ifs.fail()){
        std::cerr << "failed to open file " << filename << std::endl;
        return;
    }

    for(const auto& layer : layers){
        layer->loadWeight(ifs);
    }
}


void DeepNetwork::setVerboseMode(bool mode)
{
    for(const auto& layer : layers){
        layer->setVerboseMode(mode);
    }
}

void DeepNetwork::setLossFunction(LossFunction lf)
{
    lossFunc = lf;
}

void DeepNetwork::flush()
{
    for(const auto& layer : layers){
        layer->flush();
    }
}

