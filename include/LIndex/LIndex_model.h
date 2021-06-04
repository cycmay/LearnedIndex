#pragma once
#include <array>
#include <vector>

namespace LIndex{

template <class key_t>
class LModel
{
    // the linear regression model of learned index
private:
    
public:
    LModel();
    ~LModel();
    // train parameters using data set
    void training(std::vector<key_t> &keys, std::vector<uint64_t> &positions);
    // predict position
    uint64_t predict(const key_t &key) const;
    // the array of model parameters
    std::array<double, 2> weights;

};

}



