#include "cnn.h"
#include <iostream>
#include <cassert>

template <class X>
void printVector(const std::vector<X>& vec)
{
    for(auto elem : vec){
        std::cout << elem << ", ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

int main(int argc, char** argv)
{
    // Construct network
    DeepNetwork dn;
    dn.setInputInfo(DataSize(3, 3), 1);

    auto l1 = std::make_shared<ConvolutionLayer>(1, 2, 2);
    dn.addLayer(l1);

    auto l2 = std::make_shared<ReLULayer>();
    dn.addLayer(l2);

    auto l3 = std::make_shared<PoolingLayer>(0, 2);
    dn.addLayer(l3);

    auto l4 = std::make_shared<FullConnectLayer>(DataSize(2, 1));
    dn.addLayer(l4);

    // Learn horizontal line
    for(int i = 0; i < 2; i++){
        std::cout << "=== Horizontal line " << i << std::endl;
        std::vector<float> input(9);
        input.at(0 + i*3) = input.at(1 + i*3) = input.at(2 + i*3) = 1;
        std::vector<float> correctOutput(2);
        correctOutput.at(0) = 0;
        correctOutput.at(1) = 1;
        dn.backPropagate(input, correctOutput);

        std::cout << "output after learning:" << std::endl;
        auto outputs = dn.feedInput(input);
        printVector(outputs.back());
    }

    // Learn vertical line
    for(int i = 0; i < 2; i++){
        std::cout << "=== Vertical line " << i << std::endl;
        std::vector<float> input(9);
        input.at(i) = input.at(3 + i) = input.at(6 + i) = 1;
        std::vector<float> correctOutput(2);
        correctOutput.at(0) = 1;
        correctOutput.at(1) = 0;
        dn.backPropagate(input, correctOutput);

        std::cout << "output after learning:" << std::endl;
        auto outputs = dn.feedInput(input);
        printVector(outputs.back());
    }



    // Check another horizontal line
    std::vector<float> input(9);
    input.at(6) = input.at(7) = input.at(8) = 1;
    auto outputs = dn.feedInput(input);
    assert(outputs.back().size() == static_cast<size_t>(2));
    printVector(outputs.back());

    // Check another vertical line
    input.clear();
    input.resize(9);
    input.at(2) = input.at(5) = input.at(8) = 1;
    outputs = dn.feedInput(input);
    assert(outputs.back().size() == static_cast<size_t>(2));
    printVector(outputs.back());

    return 0;
}
