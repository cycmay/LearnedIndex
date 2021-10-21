#include <vector>

namespace LIndex{
    template<class T>
    class Binseca{

        public:
        Binseca(){};
        ~Binseca(){
            this->clear();
        };

        private:
        std::vector<T> arr;
        typedef typename std::vector<T>::const_iterator arr_cit_t;

        public:
        T bsearch_left(const T &e){
            uint64_t low=0, high=this->arr.size();
            while(low<high){
                uint64_t mid = (high-low)/2+low;
                if(this->arr[mid]<e){
                    low = mid+1;
                }else{
                    high = mid;
                }

            }
            return this->arr[low];
        }
        void binsert_left(const T e){
            this->arr.emplace_back(e);
            size_t i=this->arr.size()-1;
            while(i>0){
                if(this->arr[i-1]>=e){
                    this->arr[i]=this->arr[i-1];
                }else{
                    break;
                }
                i--;
            }
            this->arr[i]=e;
            return;
        }
        
        void clear(){
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