#pragma once

#include <vector>
#include "LIndex/LIndexModel.h"
#include "LIndex/Binseca.h"
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

        key_t min_key;
        key_t max_key;

        LModelSet(){};
        LModelSet(const std::vector<key_t> &keys, const std::vector<uint64_t> &positions){
            this->init(keys, positions);
        };
        ~LModelSet(){
            // for(auto it=this->model_set.cbegin();it!=this->model_set.cend();it++){
            //     delete *it;
            // }
            this->model_set.clear();
        };

        friend bool operator<(const LModelSet&a,const LModelSet&b){
            return a.max_key<a.max_key;
        }
        friend bool operator<=(const LModelSet&a,const LModelSet&b){
            return a.max_key<=a.max_key;
        }

        void init(const std::vector<key_t> &keys, const std::vector<uint64_t> &positions){
            this->keys = keys;
            this->positions = positions;
            this->split_model(keys, positions, this->error);
            this->min_key = this->model_set.get_min().min_key;
            this->max_key = this->model_set.get_max().max_key;
            this->save_png(keys, positions);

        }

        private:
        double error = 50.0;
        double improvement_spliting_here(uint64_t keys_begin, 
                                        uint64_t keys_end,
                                        uint64_t pos_begin,
                                        uint64_t pos_end,
                                        uint64_t point,
                                        uint64_t point_p){
            // L = abs((y1-y2)x3+(x2-x1)y3+x1y2-y1x2)/sqrt((y1-y2) ^ 2 +(x1-x2) ^ 2)
            double x1=this->keys[keys_begin].data;
            double x2=this->keys[keys_end].data;
            double y1=this->positions[pos_begin];
            double y2=this->positions[pos_end];
            double x3 = this->keys[point].data;
            double y3 = this->positions[point_p];

            return (fabs((y2 - y1) * x3 +(x1 - x2) * y3 + ((x2 * y1) -(x1 * y2)))) / (sqrt(pow(y2 - y1, 2.0) + pow(x1 - x2, 2.0)));
        }

        void top_down_split_models(uint64_t keys_begin, 
                                    uint64_t keys_end, 
                                    uint64_t pos_begin,
                                    uint64_t pos_end,
                                    double max_error){
            
            if(keys_end-keys_begin<=1){
                LModel<key_t> m;
                m.training(this->keys, this->positions, 
                            keys_begin, 
                            keys_end, 
                            pos_begin, 
                            pos_end);
                this->model_set.binsert_left(m);
                return;
            }
            double best_so_far = 0.0;
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
                this->model_set.binsert_left(m);
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
            return this->model_set.bsearch_left(tmp);
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