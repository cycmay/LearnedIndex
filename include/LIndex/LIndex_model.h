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
    LModel(){};
    LModel(const LModel<key_t> &a);
    ~LModel();
    typedef typename std::vector<key_t>::iterator vector_iterator_t;
    typedef typename std::vector<key_t>::const_iterator vector_const_iterator_t; 

    friend bool operator<(const LModel<key_t> &a, const LModel<key_t> &b){
        return a.min_key.data<b.min_key.data;
    }
    friend bool operator>(const LModel<key_t> &a, const LModel<key_t> &b){
        return a.min_key.data>b.min_key.data;
    }
    friend bool operator<=(const LModel<key_t> &a, const LModel<key_t> &b){
        return a.min_key.data<=b.min_key.data;
    }
    friend bool operator>=(const LModel<key_t> &a, const LModel<key_t> &b){
        return a.min_key.data>=b.min_key.data;
    }

    // train parameters using data set
    void training(std::vector<key_t> &keys, std::vector<uint64_t> &positions);
    void training(vector_const_iterator_t k_begin, 
                    vector_const_iterator_t k_end, 
                    std::vector<uint64_t>::const_iterator p_begin,
                    std::vector<uint64_t>::const_iterator p_end);
    // predict position
    uint64_t predict(const key_t &key) const;
    // the array of model parameters
    std::array<double, 2> weights;
    // key range
    key_t min_key, max_key;
    // loss
    double loss;

};

}



