#pragma once

#include "LIndex/LIndexModel.h"
#include "LIndex/LIndexModelSet.h"
#include "LIndex/IntervalTree.h"

namespace LIndex{
    template<class key_t>
    class LIndex{
        public:
            LIndex(){
               this->intervalTree = new IntervalTree<lmodelset_t>();
               this->intervalTree->root=this->intervalTree->NIL;
            }
            ~LIndex(){
            }

            void load_set(const std::vector<key_t> &keys, const std::vector<uint64_t> &positions){
                LModelSet<key_t> learned_model_set;
                learned_model_set.init(keys, positions);
                // this->learned_model_set = learned_model_set;
                this->intervalTree->IntervalT_Insert(learned_model_set);
            }
            uint64_t get(const key_t &key){
                // construct interval
                lmodelset_t sInte;
                sInte.low = (int)key.data;
                sInte.high = (int)key.data;
                IntervalTNode<lmodelset_t> * res_node = nullptr; 
                res_node = this->intervalTree->IntervalT_Search(sInte);
                if(res_node==this->intervalTree->NIL){
                    return -1;
                }else{
                    LModel<key_t> res_model = res_node->inte.bsearch_model_left(key); 
                    return res_model.predict(key);
                }
                
            }

            void rangeQuery(const key_t &start_key, const key_t &end_key){
                if(start_key>end_key){
                    return;
                }
                // construct interval
                lmodelset_t sInte;
                sInte.low = (int)start_key.data;
                sInte.high = (int)end_key.data;
                std::vector<IntervalTNode<lmodelset_t> *> res; 
                res = this->intervalTree->IntervalT_Search_All(sInte);
                for(auto it=res.cbegin();it!=res.cend();it++){
                    std::cout<<"The overlap interval is:"<<std::endl;
                    std::cout<<"["<<(*it)->inte.low<<"  "<<(*it)->inte.high<<"]";
                    if((*it)->color==0)
                        std::cout<<"   color:RED     ";
                    else
                        std::cout<<"   color:BLACK   ";
                    std::cout<<"Max:"<<(*it)->max<<std::endl;
                }
                
                return;
            }

            void inorderWalkThreading(){
                this->intervalTree->inorderWalkThreading();
            }

        public:
            using lmodelset_t = LModelSet<key_t>;
        private:
            IntervalTree<lmodelset_t> *intervalTree;


    };
}