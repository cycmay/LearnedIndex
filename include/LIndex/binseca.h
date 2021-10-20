#include <vector>

namespace LIndex{
    template<class T>
    class Binseca{

        public:
        Binseca(){};
        ~Binseca(){};

        private:
        std::vector<T *> arr;
        typedef typename std::vector<T*>::const_iterator arr_cit_t;

        public:
        uint64_t bsearch_left(const T &e){
            uint64_t low=0, high=this->arr.size();
            while(low<high){
                uint64_t mid = (high-low)/2+low;
                if(this->arr[mid]<e){
                    low = mid+1;
                }else{
                    high = mid;
                }

            }
            return low;
        }
        void binsert_left(T * e_p){
            this->arr.push_back(e_p);
            size_t i=this->arr.size()-1;
            while(i>0){
                if(*this->arr[i-1]>=*e_p){
                    this->arr[i]=this->arr[i-1];
                }else{
                    break;
                }
                i--;
            }
            this->arr[i]=e_p;
            return;
        }
        
        void clear(){
            for(auto it=this->arr.cbegin();it!=this->arr.cend();it++){
                delete *it;
            }
            this->arr.clear();
        }

        arr_cit_t cbegin(){
            return this->arr.cbegin();
        }
        arr_cit_t cend(){
            return this->arr.cend();
        }
    };
}