#pragma once
#include "layer.h"
#include <list>
#include <vector>
#include <memory>

class DeepNetwork
{
public:
    bool setInputInfo(DataSize size, int numChannel);
    void addLayer(std::shared_ptr<Layer> layer);
    std::vector<std::vector<float>> feedInput(const std::vector<float>& input) const;
    void backPropagate(const std::vector<float>& input, const std::vector<float>& correctOutput,
                       double reduceRate = 1.0, bool verbose = false);
    void saveWeight(std::string filename) const;
    void loadWeight(std::string filename);
private:
    DataSize inputSize;
    int numInputChannel;
    std::list<std::shared_ptr<Layer>> layers;
};

