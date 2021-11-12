#pragma once

#include <iostream>
#include <math.h>

#include "LIndex/LIndexModel.h"


namespace LIndex{


template <class ket_t>
LModel<ket_t>::LModel(const LModel<ket_t> &a)
{
    for(size_t i=0;i<a.weights.size();i++){
        this->weights[i] = a.weights[i];
    }
    this->min_key=a.min_key;
    this->max_key=a.max_key;
    this->loss = a.loss;
}

template <class ket_t>
LModel<ket_t>::~LModel()
{
}

template<class key_t>
void LModel<key_t>::training(std::vector<key_t> &keys, std::vector<uint64_t> &positions)
{
    training(keys, positions, 0, keys.size()-1, 0, positions.size()-1);
    return;
}

template<class key_t>
void LModel<key_t>::training(
                            std::vector<key_t> &keys, std::vector<uint64_t> &positions,
                            uint64_t k_begin, 
                            uint64_t k_end, 
                            uint64_t p_begin,
                            uint64_t p_end){
    if(k_end-k_begin<0){
        return;
    }
    if(k_end-k_begin<1){
        this->min_key = keys[k_begin];
        this->max_key = keys[k_end];
        this->weights[0]=1.0;
        this->weights[1]=(double)positions[p_begin]-(double)keys[k_begin].data;
        this->loss=0.0;
        return;
    }
    
    double t1=0, t2=0, t3=0, t4=0;
    for(auto i=k_begin, j=p_begin; i<=k_end&&j<=p_end; i++,j++){
        double _key = keys[i].data;
        t1 += _key*_key;
        t2 += _key;
        t3 += _key*(positions[j]);
        t4 += positions[j];
    }
    this->weights[0] = (t3*(uint64_t)(k_end-k_begin+1) - t2*t4) / (t1*(uint64_t)(k_end-k_begin+1) - t2*t2); 
    this->weights[1] = (t1*t4 - t2*t3) / (t1*(uint64_t)(k_end-k_begin+1) - t2*t2);

    // calculate loss
    loss = 0.0;
    for(auto it=k_begin, jt=p_begin; it<=k_end&&jt<=p_end; it++, jt++){
        loss += pow((weights[0]*(double)(keys[it].data)+weights[1])-(double)positions[jt], 2.0)/abs((double)(positions[p_end]-(double)positions[p_begin]));
    }

    // save min & max key
    this->min_key = keys[k_begin];
    this->max_key = keys[k_end];
    return;
}

template<class key_t>
uint64_t LModel<key_t>::predict(const key_t &key) const{
    return (uint64_t)(this->weights[0]*key.data+this->weights[1]);
}


}




