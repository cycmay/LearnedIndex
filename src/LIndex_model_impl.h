#pragma once

#include <iostream>
#include <math.h>

#include "LIndex/LIndex_model.h"


namespace LIndex{


template <class ket_t>
LModel<ket_t>::LModel(const LModel<ket_t> &a)
{
    for(size_t i=0;i<a.weights.size();i++){
        this->weights[i] = a.weights[i];
    }
    this->loss = a.loss;
}

template <class ket_t>
LModel<ket_t>::~LModel()
{
}

template<class key_t>
void LModel<key_t>::training(std::vector<key_t> &keys, std::vector<uint64_t> &positions)
{
    training(keys.cbegin(), keys.cend(), positions.cbegin(), positions.cend());
    return;
}

template<class key_t>
void LModel<key_t>::training(vector_const_iterator_t k_begin, 
                            vector_const_iterator_t k_end, 
                            std::vector<uint64_t>::const_iterator p_begin,
                            std::vector<uint64_t>::const_iterator p_end){
    if(k_end-k_begin<1){
        return;
    }
    if(k_end-k_begin<2){
        this->min_key = *k_begin;
        this->max_key = *(k_end-1);
        this->loss=0.0;
        return;
    }
    
    double t1=0, t2=0, t3=0, t4=0;
    for(auto i=k_begin, j=p_begin; i!=k_end&&j!=p_end; i++,j++){
        double _key=(*i).data;
        t1 += _key*_key;
        t2 += _key;
        t3 += _key*(*j);
        t4 += *j;
    }
    this->weights[0] = (t3*(uint32_t)(k_end-k_begin) - t2*t4) / (t1*(uint32_t)(k_end-k_begin) - t2*t2); 
    this->weights[1] = (t1*t4 - t2*t3) / (t1*(uint32_t)(k_end-k_begin) - t2*t2);

    // calculate loss
    loss = 0.0;
    for(auto it=k_begin, jt=p_begin; it!=k_end&&jt!=p_end; it++, jt++){
        loss += pow((weights[0]*(it->data)+weights[1])-(*jt), 2)/(p_end-p_begin);
    }

    // save min & max key
    this->min_key = *k_begin;
    this->max_key = *(k_end-1);
    return;
}

template<class key_t>
uint64_t LModel<key_t>::predict(const key_t &key) const{
    return (uint64_t)(this->weights[0]*key.data+this->weights[1]);
}


}




