#pragma once

#include "LIndex/LIndex_model.h"
#include "LIndex/LIndex_model_set.h"

namespace LIndex{
    template<class key_t>
    class LIndex{
        public:
        LIndex(){};
        ~LIndex(){};

        void init(const std::vector<key_t> &keys, const std::vector<uint64_t> &positions){
            this->learned_model_set.init(keys, positions);
        }
        uint64_t get(const key_t &key){
            LModel<key_t> m = this->learned_model_set.bsearch_model_left(key);
            uint64_t prd_pos = m.predict(key);
            return prd_pos;
        }

        private:
        LModelSet<key_t> learned_model_set;

    };
}