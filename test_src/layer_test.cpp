#include "layer_test.h"
#include "utility.h"
#include <fstream>
#include <algorithm>
#include <cmath>

TEST_F(ConvolutionLayerTest, apply_and_updateWeight_nozeropad)
{
    ConvolutionLayer cl(0, 2, 1);
    std::vector<float> input(9);
    input.at(0) = 1;
    input.at(1) = 1;
    input.at(2) = 1;
    cl.setInputInfo(DataSize(3, 3), 1);
    cl.calcOutputSize();
    cl.initWeight();

    constexpr size_t OUTPUT_SIZE = 4;
    std::vector<float> correctOutput(OUTPUT_SIZE);
    correctOutput.at(0) = 1;
    correctOutput.at(1) = 1;

    std::vector<float> propError(OUTPUT_SIZE);

    constexpr int NUM_ITR = 2;
    std::vector<float> error(NUM_ITR);
    for(int i = 0; i < NUM_ITR; i++) {
        auto output = cl.apply(input);
        EXPECT_EQ(OUTPUT_SIZE, output.size());

        for(int j = 0; static_cast<size_t>(j) < output.size(); j++) {
            propError.at(j) = output.at(j) - correctOutput.at(j);
        }
        std::for_each(std::begin(propError), std::end(propError),
                 [&](float elem) {error.at(i) += abs(elem);});
        cl.updateWeight(input, output, propError);
        cl.flush();
    }
    for(int i = 1; i < NUM_ITR; i++) {
        EXPECT_LT(error.at(i), error.at(i-1));
    }
}


TEST_F(ConvolutionLayerTest, apply_and_updateWeight)
{
    ConvolutionLayer cl(1, 3, 1);
    std::vector<float> input(9);
    input.at(0) = 1;
    input.at(1) = 1;
    input.at(2) = 1;
    cl.setInputInfo(DataSize(3, 3), 1);
    cl.calcOutputSize();
    cl.initWeight();

    std::vector<float> correctOutput(9);
    correctOutput.at(4) = 1;
    correctOutput.at(5) = 1;
    correctOutput.at(6) = 1;

    std::vector<float> propError(9);

    constexpr int NUM_ITR = 2;
    std::vector<float> error(NUM_ITR);
    for(int i = 0; i < NUM_ITR; i++) {
        auto output = cl.apply(input);
        EXPECT_EQ(9UL, output.size());

        for(int j = 0; static_cast<size_t>(j) < output.size(); j++) {
            propError.at(j) = output.at(j) - correctOutput.at(j);
        }
        std::for_each(std::begin(propError), std::end(propError),
                 [&](float elem) {error.at(i) += abs(elem);});
        cl.updateWeight(input, output, propError);
        cl.flush();
    }
    for(int i = 1; i < NUM_ITR; i++) {
        EXPECT_LT(error.at(i), error.at(i-1));
    }
}

TEST_F(ConvolutionLayerTest, save_and_load)
{
    ConvolutionLayer cl(1, 3, 1);
    cl.setInputInfo(DataSize(3, 3), 1);
    cl.calcOutputSize();
    cl.initWeight();

    std::ofstream ofs("save_and_load_test");
    cl.saveWeight(ofs);
    ofs.close();

    auto copiedWeight = *getWeight(cl);
    auto copiedBias = *getBias(cl);
    getWeight(cl)->clear();
    getBias(cl)->at(0) = 3;

    std::ifstream ifs("save_and_load_test");
    cl.loadWeight(ifs);
    EXPECT_NEAR(copiedWeight.at(0), getWeight(cl)->at(0), 0.0001);
    EXPECT_NEAR(copiedBias.at(0), getBias(cl)->at(0), 0.0001);
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
    auto nextPropError = rl.updateWeight(input, output, propError);

    EXPECT_EQ(input.size(), nextPropError.size());
    EXPECT_FLOAT_EQ(0.1, nextPropError.at(0));
    EXPECT_FLOAT_EQ(0, nextPropError.at(1));
    EXPECT_FLOAT_EQ(0, nextPropError.at(2));
}

/*
   Channel1
   1, 0, 0, 0, 0,
   2, 0, 0, 0, 0,
   0, 0, 0, 0, 0,
   0, 0, 0, 0, 0,
   3, 0, 0, 0, 0

   Channel2
   1, 0, 0, 0, 0,
   0, 0, 0, 0, 0,
   0, 0, 0, 0, 0,
   0, 0, 0, 0, 0,
   0, 0, 0, 0, 0
*/
TEST_F(PoolingLayerTest, apply)
{
    PoolingLayer pl(1, 3);
    int numChannel = 2;
    std::vector<float> input(25 * numChannel, -1);
    input.at(0) = 1;
    input.at(5) = 2;
    input.at(20) = 3;
    input.at(25) = 1;
    pl.setInputInfo(DataSize(5, 5), numChannel);
    pl.calcOutputSize();
    pl.initWeight();

    auto output = pl.apply(input);

    EXPECT_EQ(25UL * numChannel, output.size());
    // Channel 1
    EXPECT_EQ(2, output.at(0));
    EXPECT_EQ(2, output.at(1));
    EXPECT_EQ(0, output.at(2));
    EXPECT_EQ(3, output.at(16));
    EXPECT_EQ(0, output.at(18));
    EXPECT_EQ(0, output.at(19));
    // Channel 2
    EXPECT_EQ(1, output.at(25));
    EXPECT_EQ(0, output.at(29));
    EXPECT_EQ(0, output.at(32));
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
    auto nextPropError = pl.updateWeight(input, output, propError);

    EXPECT_EQ(input.size(), nextPropError.size());
    EXPECT_FLOAT_EQ(0.1, nextPropError.at(0));
    EXPECT_FLOAT_EQ(0, nextPropError.at(1));
    EXPECT_FLOAT_EQ(0.3, nextPropError.at(6));
}

TEST_F(FullConnectLayerTest, apply_and_updateWeight)
{
    FullConnectLayer fl(DataSize(4, 1));

    std::vector<float> input(8);
    input.at(0) = 1;
    input.at(1) = 1;
    input.at(5) = 1;
    fl.setInputInfo(DataSize(4, 2), 1);
    fl.calcOutputSize();
    fl.initWeight();

    std::vector<float> correctOutput(4);
    correctOutput.at(0) = 1;
    correctOutput.at(2) = 1;

    constexpr size_t OUTPUT_SIZE = 4;
    std::vector<float> propError(OUTPUT_SIZE);

    constexpr int NUM_ITR = 2;
    std::vector<float> error(NUM_ITR);
    for(int i = 0; i < NUM_ITR; i++) {
        auto output = fl.apply(input);
        EXPECT_EQ(OUTPUT_SIZE, output.size());

        for(int j = 0; static_cast<size_t>(j) < output.size(); j++) {
            propError.at(j) = output.at(j) - correctOutput.at(j);
        }
        std::for_each(std::begin(propError), std::end(propError),
                 [&](float elem) {error.at(i) += abs(elem);});
        fl.updateWeight(input, output, propError);
        fl.flush();
    }
    for(int i = 1; i < NUM_ITR; i++) {
        EXPECT_LT(error.at(i), error.at(i-1));
    }
}

TEST_F(FullConnectLayerTest, save_and_load)
{
    FullConnectLayer fl(DataSize(2, 1));
    fl.setInputInfo(DataSize(3, 3), 1);
    fl.calcOutputSize();
    fl.initWeight();

    std::ofstream ofs("save_and_load_test");
    fl.saveWeight(ofs);
    ofs.close();

    auto copiedWeight = *getWeight(fl);
    auto copiedBias = *getBias(fl);
    getWeight(fl)->clear();
    *getBias(fl) = 3;

    std::ifstream ifs("save_and_load_test");
    fl.loadWeight(ifs);
    EXPECT_NEAR(copiedWeight.at(0), getWeight(fl)->at(0), 0.0001);
    EXPECT_NEAR(copiedBias, *getBias(fl), 0.0001);
}

TEST_F(SoftmaxLayerTest, apply_and_updateWeight)
{
    SoftmaxLayer sml;
    std::vector<float> input(9);
    sml.setInputInfo(DataSize(3, 3), 1);
    sml.calcOutputSize();
    sml.initWeight();

    input.at(1) = 0.5;
    input.at(3) = 0.2;
    input.at(5) = 0.1;

    auto output = sml.apply(input);
    EXPECT_GT(output.at(1), output.at(0));
    EXPECT_GT(output.at(1), output.at(3));
    EXPECT_GT(output.at(1), output.at(5));

    std::vector<float> propError(output.size());
    output = sml.apply(input);
    propError.at(1) = output.at(1);
    propError.at(3) = output.at(3);
    propError.at(5) = output.at(1) - 1;
    propError = sml.updateWeight(input, output, propError);

    EXPECT_GT(propError.at(1), 0);
    EXPECT_GT(propError.at(3), 0);
    EXPECT_LT(propError.at(5), 0);
}

/***********************************************/
/* Layer Combination Test                      */
/***********************************************/

TEST_F(ConvolutionLayerTest, Convolution_and_Relu)
{
    ConvolutionLayer cl(1, 3, 1);
    ReLULayer rl;
    std::vector<float> input(9);
    input.at(0) = 1;
    input.at(1) = 1;
    input.at(2) = 1;
    cl.setInputInfo(DataSize(3, 3), 1);
    cl.calcOutputSize();
    cl.initWeight();
    rl.setInputInfo(DataSize(3, 3), 1);
    rl.calcOutputSize();
    rl.initWeight();

    std::vector<float> correctOutput(9);
    correctOutput.at(4) = 1;
    correctOutput.at(5) = 1;
    correctOutput.at(6) = 1;

    std::vector<float> propError(9);

    constexpr int NUM_ITR = 2;
    std::vector<float> error(NUM_ITR);
    for(int i = 0; i < NUM_ITR; i++) {
        auto clOutput = cl.apply(input);
        auto rlOutput = rl.apply(clOutput);

        for(int j = 0; static_cast<size_t>(j) < rlOutput.size(); j++) {
            propError.at(j) = rlOutput.at(j) - correctOutput.at(j);
        }
        std::for_each(std::begin(propError), std::end(propError),
                 [&](float elem) {error.at(i) += abs(elem);});
        propError = rl.updateWeight(clOutput, rlOutput, propError);
        cl.updateWeight(input, clOutput, propError);
        cl.flush();
    }
    for(int i = 1; i < NUM_ITR; i++) {
        EXPECT_LE(error.at(i), error.at(i-1));
    }
}

TEST_F(ConvolutionLayerTest, Conv_Relu_Pool_Conv_Relu)
{
    ConvolutionLayer cl1(1, 3, 1), cl2(1, 3, 1);
    PoolingLayer pl(1, 3);
    ReLULayer rl;
    std::vector<float> input(9);
    input.at(0) = 1;
    input.at(1) = 1;
    input.at(2) = 1;
    cl1.setInputInfo(DataSize(3, 3), 1);
    cl1.calcOutputSize();
    cl1.initWeight();
    cl2.setInputInfo(DataSize(3, 3), 1);
    cl2.calcOutputSize();
    cl2.initWeight();
    pl.setInputInfo(DataSize(3, 3), 1);
    pl.calcOutputSize();
    pl.initWeight();
    rl.setInputInfo(DataSize(3, 3), 1);
    rl.calcOutputSize();
    rl.initWeight();

    std::vector<float> correctOutput(9);
    correctOutput.at(4) = 1;
    correctOutput.at(5) = 1;
    correctOutput.at(6) = 1;

    std::vector<float> propError(9);

    constexpr int NUM_ITR = 2;
    std::vector<float> error(NUM_ITR);
    for(int i = 0; i < NUM_ITR; i++) {
        auto cl1Output = cl1.apply(input);
        auto rl1Output = rl.apply(cl1Output);
        auto plOutput = pl.apply(rl1Output);
        auto cl2Output = cl2.apply(plOutput);
        auto rl2Output = rl.apply(cl2Output);

        for(int j = 0; static_cast<size_t>(j) < rl2Output.size(); j++) {
            propError.at(j) = rl2Output.at(j) - correctOutput.at(j);
        }
        std::for_each(std::begin(propError), std::end(propError),
                 [&](float elem) {error.at(i) += abs(elem);});
        propError = rl.updateWeight(cl2Output, rl2Output, propError);
        propError = cl2.updateWeight(plOutput, cl2Output, propError);
        propError = pl.updateWeight(rl1Output, plOutput, propError);
        propError = rl.updateWeight(cl1Output, rl1Output, propError);
        cl1.updateWeight(input, cl1Output, propError);
        cl1.flush();
        cl2.flush();
    }
    for(int i = 1; i < NUM_ITR; i++) {
        EXPECT_LE(error.at(i), error.at(i-1));
    }
}

TEST_F(FullConnectLayerTest, FullConnect_and_Softmax)
{
    FullConnectLayer fl(DataSize(9, 1));
    SoftmaxLayer sml;
    std::vector<float> input(9);
    input.at(0) = 1;
    input.at(1) = 1;
    input.at(2) = 1;
    fl.setInputInfo(DataSize(3, 3), 1);
    fl.calcOutputSize();
    fl.initWeight();
    sml.setInputInfo(DataSize(3, 3), 1);
    sml.calcOutputSize();
    sml.initWeight();

    std::vector<float> correctOutput(9);
    correctOutput.at(4) = 1;
    correctOutput.at(5) = 1;
    correctOutput.at(6) = 1;

    std::vector<float> propError(9);

    constexpr int NUM_ITR = 2;
    std::vector<float> error(NUM_ITR);
    for(int i = 0; i < NUM_ITR; i++) {
        auto flOutput = fl.apply(input);
        auto smlOutput = sml.apply(flOutput);

        for(int j = 0; static_cast<size_t>(j) < smlOutput.size(); j++) {
            propError.at(j) = smlOutput.at(j) - correctOutput.at(j);
        }
        std::for_each(std::begin(propError), std::end(propError),
                 [&](float elem) {error.at(i) += abs(elem);});
        propError = sml.updateWeight(flOutput, smlOutput, propError);
        fl.updateWeight(input, flOutput, propError);
        fl.flush();
    }
    for(int i = 1; i < NUM_ITR; i++) {
        EXPECT_LT(error.at(i), error.at(i-1));
    }
}

TEST_F(FullConnectLayerTest, Full_Relu_Pool_Full_Relu)
{
    FullConnectLayer fl1(DataSize(9, 1)), fl2(DataSize(9, 1));
    PoolingLayer pl(1, 3);
    ReLULayer rl;
    std::vector<float> input(9);
    input.at(0) = 1;
    input.at(1) = 1;
    input.at(2) = 1;
    fl1.setInputInfo(DataSize(3, 3), 1);
    fl1.calcOutputSize();
    fl1.initWeight();
    fl2.setInputInfo(DataSize(3, 3), 1);
    fl2.calcOutputSize();
    fl2.initWeight();
    pl.setInputInfo(DataSize(3, 3), 1);
    pl.calcOutputSize();
    pl.initWeight();
    rl.setInputInfo(DataSize(3, 3), 1);
    rl.calcOutputSize();
    rl.initWeight();

    std::vector<float> correctOutput(9);
    correctOutput.at(4) = 1;
    correctOutput.at(5) = 1;
    correctOutput.at(6) = 1;

    std::vector<float> propError(9);

    constexpr int NUM_ITR = 2;
    std::vector<float> error(NUM_ITR);
    for(int i = 0; i < NUM_ITR; i++) {
        auto fl1Output = fl1.apply(input);
        auto rl1Output = rl.apply(fl1Output);
        auto plOutput = pl.apply(rl1Output);
        auto fl2Output = fl2.apply(plOutput);
        auto rl2Output = rl.apply(fl2Output);

        for(int j = 0; static_cast<size_t>(j) < rl2Output.size(); j++) {
            propError.at(j) = rl2Output.at(j) - correctOutput.at(j);
        }
        std::for_each(std::begin(propError), std::end(propError),
                 [&](float elem) {error.at(i) += abs(elem);});
        propError = rl.updateWeight(fl2Output, rl2Output, propError);
        propError = fl2.updateWeight(plOutput, fl2Output, propError);
        propError = pl.updateWeight(rl1Output, plOutput, propError);
        propError = rl.updateWeight(fl1Output, rl1Output, propError);
        fl1.updateWeight(input, fl1Output, propError);
        fl1.flush();
        fl2.flush();
    }
    for(int i = 1; i < NUM_ITR; i++) {
        EXPECT_LE(error.at(i), error.at(i-1));
    }
}

TEST_F(StandardizeLayerTest, apply_and_updateWeight_1channel)
{
    StandardizeLayer stdl(1);
    std::vector<float> input(8);
    input.at(0) = 1;
    input.at(1) = 1;
    input.at(2) = 2;
    input.at(3) = 1;
    input.at(4) = -1;
    stdl.setInputInfo(DataSize(4, 2), 1);
    stdl.calcOutputSize();
    stdl.initWeight();

    constexpr size_t OUTPUT_SIZE = 8;
    std::vector<float> correctOutput(OUTPUT_SIZE);
    correctOutput.at(0) = 0.5;
    correctOutput.at(1) = -0.5;
    correctOutput.at(2) = 1;

    std::vector<float> propError(OUTPUT_SIZE);

    auto output = stdl.apply(input);
    EXPECT_EQ(OUTPUT_SIZE, output.size());
    EXPECT_FLOAT_EQ(0.540061724867, output.at(0));
    EXPECT_FLOAT_EQ(0.540061724867, output.at(1));
    EXPECT_FLOAT_EQ(1.6201851746, output.at(2));
    EXPECT_FLOAT_EQ(0.540061724867, output.at(3));
    EXPECT_FLOAT_EQ(-1.6201851746, output.at(4));
    EXPECT_FLOAT_EQ(-0.540061724867, output.at(5));
    EXPECT_FLOAT_EQ(-0.540061724867, output.at(6));
    EXPECT_FLOAT_EQ(-0.540061724867, output.at(7));

    for(int j = 0; static_cast<size_t>(j) < output.size(); j++) {
        propError.at(j) = output.at(j) - correctOutput.at(j);
    }
    propError = stdl.updateWeight(input, output, propError);
    EXPECT_EQ(OUTPUT_SIZE, propError.size());
    EXPECT_FLOAT_EQ(0.04327160846566419, propError.at(0));
}

TEST_F(StandardizeLayerTest, apply_and_updateWeight_8channel_1batch)
{
    StandardizeLayer stdl(1);
    std::vector<float> input(64);
    input.at(0) = 1;
    input.at(1) = 1;
    input.at(2) = 2;
    input.at(3) = 1;
    input.at(4) = -1;
    input.at(8) = 100;
    input.at(16) = 100;
    input.at(32) = 100;
    stdl.setInputInfo(DataSize(4, 2), 8);
    stdl.calcOutputSize();
    stdl.initWeight();

    constexpr size_t OUTPUT_SIZE = 64;
    std::vector<float> correctOutput(OUTPUT_SIZE);
    correctOutput.at(0) = 0.5;
    correctOutput.at(1) = -0.5;
    correctOutput.at(2) = 1;

    std::vector<float> propError(OUTPUT_SIZE);

    auto output = stdl.apply(input);
    EXPECT_EQ(OUTPUT_SIZE, output.size());
    EXPECT_FLOAT_EQ(0.540061724867, output.at(0));
    EXPECT_FLOAT_EQ(0.540061724867, output.at(1));
    EXPECT_FLOAT_EQ(1.6201851746, output.at(2));
    EXPECT_FLOAT_EQ(0.540061724867, output.at(3));
    EXPECT_FLOAT_EQ(-1.6201851746, output.at(4));
    EXPECT_FLOAT_EQ(-0.540061724867, output.at(5));
    EXPECT_FLOAT_EQ(-0.540061724867, output.at(6));
    EXPECT_FLOAT_EQ(-0.540061724867, output.at(7));

    for(int j = 0; static_cast<size_t>(j) < output.size(); j++) {
        propError.at(j) = output.at(j) - correctOutput.at(j);
    }
    propError = stdl.updateWeight(input, output, propError);
    EXPECT_EQ(OUTPUT_SIZE, propError.size());
    EXPECT_FLOAT_EQ(0.04327160846566419, propError.at(0));
}

TEST_F(StandardizeLayerTest, apply_and_updateWeight_8channel_2batch)
{
    StandardizeLayer stdl(2);
    std::vector<float> input(64);
    input.at(0) = 1;
    input.at(1) = 1;
    input.at(2) = 2;
    input.at(3) = 1;
    input.at(4) = -1;
    input.at(8) = 4;
    input.at(16) = 100;
    input.at(32) = 100;
    stdl.setInputInfo(DataSize(4, 2), 8);
    stdl.calcOutputSize();
    stdl.initWeight();

    constexpr size_t OUTPUT_SIZE = 64;
    std::vector<float> correctOutput(OUTPUT_SIZE);
    correctOutput.at(0) = 0.4;
    correctOutput.at(1) = -0.5;
    correctOutput.at(2) = 1;

    std::vector<float> propError(OUTPUT_SIZE);

    auto output = stdl.apply(input);
    EXPECT_EQ(OUTPUT_SIZE, output.size());
    EXPECT_FLOAT_EQ(0.43301270189221935, output.at(0));
    EXPECT_FLOAT_EQ(0.43301270189221935, output.at(1));
    EXPECT_FLOAT_EQ(1.299038105676658, output.at(2));
    EXPECT_FLOAT_EQ(0.43301270189221935, output.at(3));
    EXPECT_FLOAT_EQ(-1.299038105676658, output.at(4));
    EXPECT_FLOAT_EQ(-0.43301270189221935, output.at(5));
    EXPECT_FLOAT_EQ(-0.43301270189221935, output.at(6));
    EXPECT_FLOAT_EQ(-0.43301270189221935, output.at(7));
    EXPECT_FLOAT_EQ(3.0310889132455356, output.at(8));
    EXPECT_FLOAT_EQ(-0.43301270189221935, output.at(9));

    for(int j = 0; static_cast<size_t>(j) < output.size(); j++) {
        propError.at(j) = output.at(j) - correctOutput.at(j);
    }
    propError = stdl.updateWeight(input, output, propError);
    EXPECT_EQ(OUTPUT_SIZE, propError.size());
    EXPECT_FLOAT_EQ(0.028589851814324063, propError.at(0));
    EXPECT_FLOAT_EQ(0, propError.back());
}

