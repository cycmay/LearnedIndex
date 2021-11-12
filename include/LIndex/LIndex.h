#pragma once

#include "LIndex/LIndexModel.h"
#include "LIndex/LIndexModelSet.h"

namespace LIndex{
    template<class key_t>
    class LIndex{
        public:
            LIndex(){
               
            };
            ~LIndex(){
                
            };

            void load_set(const std::vector<key_t> &keys, const std::vector<uint64_t> &positions){
                LModelSet<key_t> learned_model_set;
                learned_model_set.init(keys, positions);
                this->learned_model_set = learned_model_set;
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