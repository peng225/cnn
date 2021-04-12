#pragma once
#include "layer.h"
#include <list>
#include <vector>
#include <memory>

enum class LossFunction
{
    MSE,  // 二乗誤差
    CRS_ENT  // クロスエントロピー
};

class DeepNetwork
{
public:
    DeepNetwork();
    DeepNetwork(int mbSize);
    bool setInputInfo(DataSize size, int numChannel);
    void addLayer(std::shared_ptr<Layer> layer);
    std::vector<std::vector<float>> feedInput(const std::vector<float>& input) const;
    void backPropagate(const std::vector<float>& input, const std::vector<float>& correctOutput,
                       double reduceRate = 1.0, bool verbose = false);
    void saveWeight(std::string filename) const;
    void loadWeight(std::string filename);
    void setVerboseMode(bool mode);
    void setLossFunction(LossFunction lf);
    void flush();
private:
    DataSize inputSize;
    int numInputChannel;
    int minibatchSize;
    int inputCount;
    LossFunction lossFunc;
    std::list<std::shared_ptr<Layer>> layers;
};

