#include "cnn.h"
#include "utility.h"
#include <iostream>
#include <cassert>
#include <algorithm>

int main(int argc, char** argv)
{
    // Construct network
    DeepNetwork dn;
    dn.setInputInfo(DataSize(4, 4), 1);
    const int NUM_OUTPUT = 1;

    auto l1 = std::make_shared<ConvolutionLayer>(1, 3, 2);
    dn.addLayer(l1);

    auto l2 = std::make_shared<ReLULayer>();
    dn.addLayer(l2);

    auto l3 = std::make_shared<PoolingLayer>(0, 3);
    dn.addLayer(l3);

    auto l4 = std::make_shared<FullConnectLayer>(DataSize(NUM_OUTPUT, 1));
    dn.addLayer(l4);

    l1->dumpWeight();
    // Create training data
    std::vector<std::pair<std::vector<float>, std::vector<float>>> training;
    // Learn horizontal line
    for(int i = 1; i < 3; i++){
        std::vector<float> input(16);
        input.at(0 + i*4) = input.at(1 + i*4) = input.at(2 + i*4) = input.at(3 + i*4) = 1;
        std::vector<float> correctOutput(NUM_OUTPUT);
        correctOutput.at(0) = 0;
        training.push_back(std::pair<std::vector<float>, std::vector<float>>(input, correctOutput));
    }

    // Learn vertical line
    // for(int i = 1; i < 2; i++){
    //     std::vector<float> input(16);
    //     input.at(i) = input.at(3 + i) = input.at(6 + i) = input.at(9 + i) = 1;
    //     std::vector<float> correctOutput(2);
    //     correctOutput.at(0) = 1;
    //     correctOutput.at(1) = 0;
    //     training.push_back(std::pair<std::vector<float>, std::vector<float>>(input, correctOutput));
    // }

    for(int numTrain = 0; numTrain < 10; numTrain++){
        random_shuffle(std::begin(training), std::end(training));
        for(auto& elem : training){
            dn.backPropagate(elem.first, elem.second);
            std::cout << "output after learning:" << std::endl;
            auto outputs = dn.feedInput(elem.first);
            printVector(outputs.back());
        }
    }



    // Check another horizontal line
    std::vector<float> input(16);
    // input.at(0) = input.at(1) = input.at(2) = input.at(3) = 1;
    // auto outputs = dn.feedInput(input);
    // assert(outputs.back().size() == static_cast<size_t>(2));
    // printVector(outputs.back());

    // input.clear();
    // input.resize(16);
    // input.at(12) = input.at(13) = input.at(14) = input.at(15) = 1;
    // outputs = dn.feedInput(input);
    // assert(outputs.back().size() == static_cast<size_t>(2));
    // printVector(outputs.back());

    // Check another vertical line
    // input.clear();
    // input.resize(16);
    // input.at(0) = input.at(4) = input.at(8) = input.at(12) = 1;
    // outputs = dn.feedInput(input);
    // assert(outputs.back().size() == static_cast<size_t>(2));
    // printVector(outputs.back());

    // input.clear();
    // input.resize(16);
    // input.at(3) = input.at(7) = input.at(11) = input.at(15) = 1;
    // outputs = dn.feedInput(input);
    // assert(outputs.back().size() == static_cast<size_t>(2));
    // printVector(outputs.back());
    // l1->dumpWeight();


    for(int i = 0; i < 16; i++){
        std::cout << "i = " << i << std::endl;
        input.clear();
        input.resize(16);
        input.at(i) = 1;
        auto outputs = dn.feedInput(input);
        assert(outputs.back().size() == static_cast<size_t>(NUM_OUTPUT));
        printVector(outputs.back());
    }

    return 0;
}
