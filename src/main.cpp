#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include "LIndex/LIndex_model.h"
#include "LIndex/LIndex_model_impl.h"

class Key{
    public:
    int64_t data;
    Key(int64_t d):data(d){};

    friend bool operator<(const Key &l, const Key &r){return l.data<r.data;}
    friend bool operator<=(const Key &l, const Key &r){return l.data<=r.data;}
};

size_t Keys_number = 1000000;


int main(int, char**) {

    std::vector<Key> keys;
    std::vector<uint64_t> positions(keys.size());
    std::vector<uint64_t> values(keys.size());

    typedef Key key_t;
    LIndex::LModel<key_t> test = LIndex::LModel<key_t>();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> rand_int64(
        0, Keys_number*2.7);

    keys.reserve(Keys_number);
    

    for (size_t i = 0; i < Keys_number; ++i) {
        keys.push_back(Key(rand_int64(gen)));
        positions.push_back((uint64_t)i);
        values.push_back(1);
    }

    std::sort(keys.begin(), keys.end());
   
    test.training(keys, positions);

    printf("weight: [%.3f, %.3f]\n", test.weights[0], test.weights[1]);

    std::uniform_int_distribution<int64_t> rand_keynumber(0, Keys_number);
    for(size_t i=0;i<5;i++){
        uint64_t pos = rand_keynumber(gen);
        printf("Key [%ld] predict-postion:[%ld] real-positon [%ld]\n", keys[pos].data, test.predict(keys[pos]), (uint64_t)pos);
    }
   
}
