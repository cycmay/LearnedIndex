#pragma once
#include <vector>
#include "LIndex/LIndex_model.h"
#include "LIndex/binseca.h"

namespace LIndex{
    template<class key_t>
    class LModelSet{
        public:
        typedef typename std::vector<key_t>::iterator vector_iterator_t;
        typedef typename std::vector<key_t>::const_iterator vector_const_iterator_t; 
        Binseca<LModel<key_t> > model_set;
        
        public:
        LModelSet(){};
        ~LModelSet(){
            this->model_set.clear();
        };

        private:
        double improvement_spliting_here(vector_const_iterator_t keys_begin, 
                                        vector_const_iterator_t keys_end,
                                        uint64_t pos_begin,
                                        uint64_t pos_end,
                                        vector_const_iterator_t point,
                                        uint64_t point_p){
            return abs((pos_begin-pos_end)*point->data+(keys_end->data-keys_begin->data)*point_p+keys_begin->data*pos_end-pos_begin*keys_end->data)/sqrt(pow((pos_begin-pos_end), 2) +pow((keys_begin->data-keys_end->data),2));
        }

        void top_down_split_models(vector_const_iterator_t keys_begin, 
                                    vector_const_iterator_t keys_end, 
                                    std::vector<uint64_t>::const_iterator pos_begin,
                                    std::vector<uint64_t>::const_iterator pos_end,
                                    double max_error){
            if(keys_end-keys_begin==0){
                return;
            }
            if(keys_end-keys_begin==1){
                LModel<key_t> *m = new LModel<key_t>();
                m->weights[0]=1.0;
                m->weights[1]=(double)*pos_begin;
                m->min_key = *keys_begin;
                m->max_key = *keys_begin;
                m->loss = 0.0;
                this->model_set.binsert_left(m);
                return;
            }
            if(keys_end-keys_begin==2){
                LModel<key_t> *m = new LModel<key_t>();
                m->weights[0]=(*(pos_end-1)-*(pos_begin))/((*keys_end).data-(*keys_begin).data);
                m->weights[1]=(*pos_begin)-m->weights[0]*((*keys_begin).data);
                m->min_key = *keys_begin;
                m->max_key = *(keys_end-1);
                m->loss = 0.0;
                this->model_set.binsert_left(m);
                return;
            }
            double best_so_far = 0;
            vector_const_iterator_t break_point = keys_begin;
            std::vector<uint64_t>::const_iterator break_point_p = pos_begin;

            double improvement_in_approximation = 0.0;

            // find best place to split linear
            vector_const_iterator_t it = keys_begin+1;
            std::vector<uint64_t>::const_iterator pt = pos_begin+1;
            for(;it!=keys_end-1&&pt!=pos_end-1; 
                    it++, pt++){
                improvement_in_approximation = improvement_spliting_here(keys_begin,
                                                                        keys_end,
                                                                        *(pos_begin-1),
                                                                        *(pos_end-1),
                                                                        it,
                                                                        *(pt)
                                                                        );
                if(improvement_in_approximation>best_so_far){
                    break_point = it;
                    break_point_p = pt;
                    best_so_far = improvement_in_approximation;
                }
            }

            // Recursively split the left&right segment
            if(best_so_far>max_error){
                top_down_split_models(keys_begin, 
                                        break_point+1,
                                        pos_begin,
                                        break_point_p+1,
                                        max_error);
                top_down_split_models(break_point+1,
                                        keys_end,
                                        break_point_p+1,
                                        pos_end,
                                        max_error);
            }else{
                LModel<key_t> *m = new LModel<key_t>();
                m->training(keys_begin, keys_end, pos_begin, pos_end);
                model_set.binsert_left(m);
            }
        }
    
        public:
        void split_model(std::vector<key_t> &keys, std::vector<uint64_t> &pos, double max_error){
            top_down_split_models(keys.cbegin(),
                                    keys.cend(),
                                    pos.cbegin(),
                                    pos.cend(),
                                    max_error);
            return;
        }
        
    };
}