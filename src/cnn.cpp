#include "cnn.h"
#include "utility.h"
#include <iostream>
#include <cassert>
#include <iterator>

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
        std::cout << "Next propError:" << std::endl;
        printVector(propError);
        index--;
    }
}

