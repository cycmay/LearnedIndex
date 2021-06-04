#include <iostream>

#include "LIndex/LIndex_model.h"

#pragma once

namespace LIndex{

template <class ket_t>
LModel<ket_t>::LModel()
{
    std::cout<<"hello"<<std::endl;
}
template <class ket_t>
LModel<ket_t>::~LModel()
{
}

template<class key_t>
void LModel<key_t>::training(std::vector<key_t> &keys, std::vector<uint64_t> &positions)
{
    double t1=0, t2=0, t3=0, t4=0;
    for(size_t i=0; i<keys.size(); ++i)
    {
        double _key = keys[i].data;
        t1 += _key*_key;
        t2 += _key;
        t3 += _key*positions[i];
        t4 += positions[i];
    }
    this->weights[0] = (t3*keys.size() - t2*t4) / (t1*keys.size() - t2*t2); 
    this->weights[1] = (t1*t4 - t2*t3) / (t1*keys.size() - t2*t2);

}

template<class key_t>
uint64_t LModel<key_t>::predict(const key_t &key) const{
    return (uint64_t)(this->weights[0]*key.data+this->weights[1]);
}


}




