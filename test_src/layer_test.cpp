#include "layer_test.h"

TEST_F(ConvolutionLayerTest, apply)
{
    ConvolutionLayer cl(1, 3, 1);
    std::vector<float> input(9);
    cl.setInputInfo(DataSize(3, 3), 1);
    cl.calcOutputSize();
    cl.initWeight();
    auto output = cl.apply(input);
    EXPECT_EQ(9UL, output.size());
}

TEST_F(ReLULayerTest, apply)
{
    ReLULayer rl;
    int numChannel = 2;
    std::vector<float> input(9 * numChannel);
    input.at(0) = 1;
    input.at(1) = -1;
    input.at(10) = -1;
    rl.setInputInfo(DataSize(3, 3), numChannel);
    rl.calcOutputSize();
    rl.initWeight();

    auto output = rl.apply(input);

    EXPECT_EQ(9UL * numChannel, output.size());
    EXPECT_EQ(1, output.at(0));
    EXPECT_EQ(0, output.at(1));
    EXPECT_EQ(0, output.at(10));
}

TEST_F(ReLULayerTest, updateWeight)
{
    ReLULayer rl;
    std::vector<float> input(9);
    input.at(0) = 1;
    input.at(1) = 0;
    rl.setInputInfo(DataSize(3, 3), 1);
    rl.calcOutputSize();
    rl.initWeight();

    auto output = rl.apply(input);
    std::vector<float> propError(input.size());
    propError.at(0) = 0.1;
    propError.at(1) = 0.2;
    propError.at(2) = 0.3;
    auto newPropError = rl.updateWeight(input, output, propError);

    EXPECT_EQ(input.size(), newPropError.size());
    EXPECT_FLOAT_EQ(0.1, newPropError.at(0));
    EXPECT_FLOAT_EQ(0, newPropError.at(1));
    EXPECT_FLOAT_EQ(0, newPropError.at(2));
}

TEST_F(PoolingLayerTest, apply)
{
    PoolingLayer pl(1, 3);
    int numChannel = 2;
    std::vector<float> input(9 * numChannel);
    input.at(0) = 1;
    input.at(6) = 2;
    input.at(9) = 1;
    input.at(15) = 2;
    pl.setInputInfo(DataSize(3, 3), numChannel);
    pl.calcOutputSize();
    pl.initWeight();

    auto output = pl.apply(input);

    EXPECT_EQ(9UL * numChannel, output.size());
    // Channel 1
    EXPECT_EQ(1, output.at(1));
    EXPECT_EQ(0, output.at(2));
    EXPECT_EQ(2, output.at(6));
    // Channel 2
    EXPECT_EQ(1, output.at(10));
    EXPECT_EQ(0, output.at(11));
    EXPECT_EQ(2, output.at(15));
}

TEST_F(PoolingLayerTest, updateWeight)
{
    PoolingLayer pl(1, 3);
    std::vector<float> input(9);
    input.at(0) = 1;
    input.at(5) = 2;
    input.at(8) = 3;
    pl.setInputInfo(DataSize(3, 3), 1);
    pl.calcOutputSize();
    pl.initWeight();

    auto output = pl.apply(input);
    std::vector<float> propError(input.size());
    propError.at(0) = 0.1;
    propError.at(1) = 0.2;
    propError.at(6) = 0.3;
    auto newPropError = pl.updateWeight(input, output, propError);

    EXPECT_EQ(input.size(), newPropError.size());
    EXPECT_FLOAT_EQ(0.1, newPropError.at(0));
    EXPECT_FLOAT_EQ(0, newPropError.at(1));
    EXPECT_FLOAT_EQ(0.3, newPropError.at(6));
}

