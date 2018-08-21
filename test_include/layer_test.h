#include <gtest/gtest.h>
#include "cnn.h"
#include <vector>

class ConvolutionLayerTest : public ::testing::Test
{
protected:
    std::vector<float>* getWeight(ConvolutionLayer& cl)
    {
        return &cl.weight;
    }
    std::vector<float>* getBias(ConvolutionLayer& cl)
    {
        return &cl.bias;
    }
};

class ReLULayerTest : public ::testing::Test
{
};

class PoolingLayerTest : public ::testing::Test
{
};

class FullConnectLayerTest : public ::testing::Test
{
protected:
    std::vector<float>* getWeight(FullConnectLayer& fl)
    {
        return &fl.weight;
    }
    float* getBias(FullConnectLayer& fl)
    {
        return &fl.bias;
    }

};

