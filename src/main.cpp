#include "cnn.h"
#include <iostream>
#include <cassert>

int main(int argc, char** argv)
{
    DeepNetwork dn;
    dn.setInputInfo(DataSize(3, 3));

    auto l1 = std::make_shared<ConvolutionLayer>(1, 2);
    dn.addLayer(l1);

    auto l2 = std::make_shared<ReLULayer>();
    dn.addLayer(l2);

    auto l3 = std::make_shared<PoolingLayer>(0, 2);
    dn.addLayer(l3);

    auto l4 = std::make_shared<FullConnectLayer>(DataSize(2, 1));
    dn.addLayer(l4);

    std::vector<float> input(9);

    auto output = dn.feedInput(input);

    assert(output.size() == static_cast<size_t>(2));

    for(const auto& elem : output){
        std::cout << elem << std::endl;
    }

    return 0;
}
