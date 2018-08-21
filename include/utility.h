#include <iostream>

/* ======================
    Utility functions
   ======================*/
template <class X>
void printVector(const std::vector<X>& vec)
{
    return;
    for(auto elem : vec){
        std::cout << elem << ", ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
}


