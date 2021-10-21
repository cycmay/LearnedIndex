#pragma once
#include <vector>
#include "LIndex/LIndex_model.h"
#include "LIndex/binseca.h"
#include "LIndex/matplotlibcpp.h"

namespace LIndex{
    template<class key_t>
    class LModelSet{
        public:
        typedef typename std::vector<key_t>::iterator vector_iterator_t;
        typedef typename std::vector<key_t>::const_iterator vector_const_iterator_t; 
        Binseca<LModel<key_t> > model_set;

        std::vector<key_t> keys;
        std::vector<uint64_t> positions;
        
        public:
        LModelSet(){};
        ~LModelSet(){
            // for(auto it=this->model_set.cbegin();it!=this->model_set.cend();it++){
            //     delete *it;
            // }
            this->model_set.clear();
        };
        void init(const std::vector<key_t> &keys, const std::vector<uint64_t> &positions){
            this->keys = keys;
            this->positions = positions;
            this->split_model(keys, positions, this->error);
            this->save_png(keys, positions);

        }

        private:
        double error = 0.10;
        double improvement_spliting_here(uint64_t keys_begin, 
                                        uint64_t keys_end,
                                        uint64_t pos_begin,
                                        uint64_t pos_end,
                                        uint64_t point,
                                        uint64_t point_p){
            // L = abs((y1-y2)x3+(x2-x1)y3+x1y2-y1x2)/sqrt((y1-y2) ^ 2 +(x1-x2) ^ 2)
            uint64_t x1=this->keys[keys_begin].data;
            uint64_t x2=this->keys[keys_end].data;
            uint64_t y1=this->positions[pos_begin];
            uint64_t y2=this->positions[pos_end];
            uint64_t x3 = this->keys[point].data;
            uint64_t y3 = this->positions[point_p];

            return (fabs((y2 - y1) * x3 +(x1 - x2) * y3 + ((x2 * y1) -(x1 * y2)))) / (sqrt(pow(y2 - y1, 2) + pow(x1 - x2, 2)));
        }

        void top_down_split_models(uint64_t keys_begin, 
                                    uint64_t keys_end, 
                                    uint64_t pos_begin,
                                    uint64_t pos_end,
                                    double max_error){
            if(keys_end-keys_begin<0){
                return;
            }
            if(keys_end-keys_begin==0){
                LModel<key_t> m;
                m.weights[0]=1.0;
                m.weights[1]=(double)pos_begin;
                m.min_key = this->keys[keys_begin];
                m.max_key = this->keys[keys_begin];
                m.loss = 0.0;
                this->model_set.binsert_left(m);
                return;
            }
            if(keys_end-keys_begin==1){
                LModel<key_t> m;
                m.weights[0]=(pos_end-pos_begin)/(this->keys[keys_end].data-this->keys[keys_begin].data);
                m.weights[1]=pos_begin-m.weights[0]*(this->keys[keys_begin].data);
                m.min_key = this->keys[keys_begin];
                m.max_key = this->keys[keys_end];
                m.loss = 0.0;
                this->model_set.binsert_left(m);
                return;
            }
            double best_so_far = 0;
            uint64_t break_point = keys_begin, break_point_p = pos_begin;

            double improvement_in_approximation = 0.0;

            // find best place to split linear
            uint64_t it = keys_begin+1, pt = pos_begin+1;
            for(;it<keys_end&&pt<pos_end; it++, pt++){
                improvement_in_approximation = improvement_spliting_here(keys_begin,
                                                                        keys_end,
                                                                        pos_begin,
                                                                        pos_end,
                                                                        it,
                                                                        pt
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
                                        break_point,
                                        pos_begin,
                                        break_point_p,
                                        max_error);
                top_down_split_models(break_point,
                                        keys_end,
                                        break_point_p,
                                        pos_end,
                                        max_error);
            }else{
                LModel<key_t> m;
                m.training(this->keys, this->positions, 
                            keys_begin, 
                            keys_end, 
                            pos_begin, 
                            pos_end);
                model_set.binsert_left(m);
            }
        }
    
        void split_model(const std::vector<key_t> &keys, const std::vector<uint64_t> &pos, double max_error){
            top_down_split_models(0,
                                    keys.size()-1,
                                    0,
                                    pos.size()-1,
                                    max_error);
            return;
        }

        public:
        LModel<key_t> bsearch_model_left(const key_t &key){
            LModel<key_t> tmp;
            tmp.min_key = key;
            LModel<key_t> m = this->model_set.bsearch_left(tmp);
            return m;
        }

        void save_png(const std::vector<key_t> &keys, const std::vector<uint64_t> &positions){

            // show pillow
            std::vector<uint64_t> x(keys.size());
            std::vector<uint64_t> y(positions.size());
            for(size_t i=0;i<keys.size();i++){
                x.at(i)=keys[i].data;
                y.at(i)=positions[i];
            }
            namespace plt=matplotlibcpp;
            plt::plot(x,y);
            // Enable legend.
            plt::legend();


        
            for(auto it=this->model_set.cbegin(); it!=this->model_set.cend(); it++){
                printf("min_key: %ld, max_key %ld\n", (*it).min_key.data, (*it).max_key.data);
                printf("weight: [%.3f, %.3f]\n", (*it).weights[0], (*it).weights[1]);
                printf("loss: %.3f\n", (*it).loss);

                std::vector<uint64_t> x;
                std::vector<uint64_t> y;
                for(auto i=(*it).min_key.data;i<(*it).max_key.data;i++){
                    x.push_back(i);
                    y.push_back((*it).predict(i));
                }
                plt::plot(x,y);
            }
             // save figure
            const char* filename = "./basic.png";
            std::cout << "Saving result to " << filename << std::endl;;
            plt::save(filename);

        }
        
    };
}