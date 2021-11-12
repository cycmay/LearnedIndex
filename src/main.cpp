#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include "LIndex/LIndexModel.h"
#include "LIndex/LIndexModelSet.h"
#include "LIndex/LIndex.h"
#include "LIndex_model_impl.hpp"
#include "LIndex/IntervalTree.h"

#include "LIndex/matplotlibcpp.h"

//删除字符串中空格，制表符tab等无效字符
std::string trim(std::string& str)
{
	//str.find_first_not_of(" \t\r\n"),在字符串str中从索引0开始，返回首次不匹配"\t\r\n"的位置
	str.erase(0,str.find_first_not_of(" \t\r\n"));
	str.erase(str.find_last_not_of(" \t\r\n") + 1);
	return str;
}


class Key{
    public:
    uint64_t data;
    Key(){};
    Key(uint64_t d):data(d){};
    Key & operator=(const Key &a){
        this->data=a.data;
        return *this;
    }
    Key(const Key &a){
        this->data=a.data;
    }

    friend bool operator<(const Key &l, const Key &r){return l.data<r.data;}
    friend bool operator<=(const Key &l, const Key &r){return l.data<=r.data;}
};

size_t Keys_number = 1000;


void test_linear_model(){
    // std::vector<Key> keys;
    // std::vector<uint64_t> positions(keys.size());
    // std::vector<uint64_t> values(keys.size());

    // typedef Key key_t;
    // LIndex::LModel<key_t> test;

    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<int64_t> rand_int64(
    //     0, Keys_number*2.7);

    // keys.reserve(Keys_number);
    

    // for (size_t i = 0; i < Keys_number; ++i) {
    //     keys.push_back(Key(rand_int64(gen)));
    //     positions.push_back((uint64_t)i);
    //     values.push_back(1);
    // }

    // std::sort(keys.begin(), keys.end());

    std::vector<Key> keys;
    std::vector<uint64_t> positions(keys.size());
    std::vector<uint64_t> values(keys.size());

    typedef Key key_t;
    LIndex::LModel<key_t> test;

    std::ifstream fin("/home/bicycle/dev/LearnIndex/src/data/DEXSZUS.csv");
	std::string line; 
    
	for (size_t a=0;a<=500;a++){
        getline(fin, line);
		std::istringstream sin(line);
		std::vector<std::string> fields;
		std::string field;
		while (getline(sin, field, ','))
		{
			fields.push_back(field);
		}
        uint64_t _key = atoi(trim(fields[0]).c_str());
        uint64_t _pos = atof(trim(fields[1]).c_str())*10000;
        if(!_key||!_pos){
            continue;
        }else{
            keys.push_back(Key(_key));
		    positions.push_back(_pos);
        }
    }

    // show pillow
    std::vector<uint64_t> x(keys.size());
    std::vector<uint64_t> y(positions.size());
    for(size_t i=0;i<keys.size();i++){
        x[i]=keys[i].data;
        y[i]=positions[i];
    }
    namespace plt=matplotlibcpp;
    plt::plot(x,y);
    // Enable legend.
    plt::legend();
    test.training(keys, positions);
    printf("min_key: %ld, max_key %ld\n", test.min_key.data, test.max_key.data);
    printf("weight: [%.3f, %.3f]\n", test.weights[0], test.weights[1]);
    printf("loss: %.3f\n", test.loss);
    x.clear();
    y.clear();
    for(auto i=test.min_key.data;i<test.max_key.data;i++){
        x.push_back(i);
        y.push_back(test.predict(i));
    }
    plt::plot(x,y);
   
        // save figure
    const char* filename = "./basic_lmodel.png";
    std::cout << "Saving result to " << filename << std::endl;;
    plt::save(filename);
   
    

   
}



void test_lindex(){
    std::vector<Key> keys;
    std::vector<uint64_t> positions(keys.size());
    std::vector<uint64_t> values(keys.size());

    typedef Key key_t;

    std::ifstream fin("/home/bicycle/dev/LearnIndex/src/data/DEXSZUS.csv");
	std::string line; 
	while (getline(fin, line))
	{
		std::istringstream sin(line);
		std::vector<std::string> fields;
		std::string field;
		while (getline(sin, field, ','))
		{
			fields.push_back(field);
		}
        uint64_t _key = atoi(trim(fields[0]).c_str());
        uint64_t _pos = atof(trim(fields[1]).c_str())*10000;
        if(!_key||!_pos){
            continue;
        }else{
            keys.push_back(Key(_key));
		    positions.push_back(_pos);
        }
		 
	}
    LIndex::LIndex<key_t> lindex;

    // show pillow
    std::vector<uint64_t> x(keys.size());
    std::vector<uint64_t> y(positions.size());

    lindex.load_set(keys, positions);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> rand_keynumber(0, keys.size());
    for(size_t i=0;i<10;i++){
        uint64_t p = rand_keynumber(gen);
        printf("Key [%ld] predict-position:[%ld] real-posiiton [%ld]\n", keys[p].data, lindex.get(keys[p]), (uint64_t)positions[p]);
    }
}


class interval{
    public:
        int low;
        int high;

        interval(){};
        interval(int low, int high){
            this->low=low;
            this->high=high;
        };
        interval & operator=(const interval &inte){
            this->low = inte.low;
            this->high = inte.high;
            return *this;
        };
};

void test_interval_tree(){
	interval A[]={interval(16,21), 
                    interval(8,9), 
                    interval(25,30), 
                    interval(5,8),
                    interval(15,23),
                    interval(17,19), 
                    interval(26,26),
                    interval(0,3), 
                    interval(6,10),
                    interval(19,20)};

	int n=sizeof(A)/sizeof(interval);
	
	std::cout<<"/*---------------------Create Interval Tree-------------------*/"<<std::endl;
	LIndex::IntervalTree<interval> *T=new LIndex::IntervalTree<interval>();
	T->root=T->NIL;
	for(int i=0;i<n;i++)
		T->IntervalT_Insert(T,A[i]);
	std::cout<<"The interval tree is:"<<std::endl;
	T->IntervalT_InorderWalk(T->root);
	std::cout<<"The root of the tree is:"<<T->root->inte.low<<"   "<<T->root->inte.high<<std::endl;
	std::cout<<"/*-------------------------------------------------------------*/"<<std::endl;
 
	std::cout<<"/*--------------------Searching Interval Tree------------------*/"<<std::endl;
	interval sInt;
	std::cout<<"Please input the searching interval:";
	std::cin>>sInt.low>>sInt.high;
	LIndex::IntervalTNode<interval> *sITNode=T->NIL;
	sITNode=T->IntervalT_Search(T,sInt);
	if(sITNode==T->NIL)
		std::cout<<"The searching interval doesn't exist in the tree."<<std::endl;
	else{
		std::cout<<"The overlap interval is:"<<std::endl;
		std::cout<<"["<<sITNode->inte.low<<"  "<<sITNode->inte.high<<"]";
		if(sITNode->color==0)
			std::cout<<"   color:RED     ";
		else
			std::cout<<"   color:BLACK   ";
		std::cout<<"Max:"<<sITNode->max<<std::endl;
		}
	std::cout<<"/*------------------Deleting INterval Tree--------------------*/"<<std::endl;
	interval dInt;
	std::cout<<"Please input the deleting interval:";
	std::cin>>dInt.low>>dInt.high;
	LIndex::IntervalTNode<interval>  *dITNode=T->NIL;
	dITNode=T->IntervalT_Search(T,dInt);
	if(dITNode==T->NIL)
		std::cout<<"The deleting interval doesn't exist in the tree."<<std::endl;
	else
	{ 
		T->IntervalT_Delete(T,dITNode);
		std::cout<<"After deleting ,the interval tree is:"<<std::endl;
		T->IntervalT_InorderWalk(T->root);
		std::cout<<"The root of the tree is:"<<T->root->inte.low<<"   "<<T->root->inte.high<<std::endl;
		}
	std::cout<<"/*------------------------------------------------------------*/"<<std::endl;
 
 
}

int main(int, char**) {
    // uint64_t x1=25937;
    // uint64_t x2=43812;
    // uint64_t y1=43179;
    // uint64_t y2=9838;
    // uint64_t x3=25938;
    // uint64_t y3=43117;
    // double tmp1 = ((y2 - y1) * x3 +(x1 - x2) * y3 + ((x2 * y1) -(x1 * y2))) ;
    // uint64_t t1=pow((double)y2 - (double)y1, 2);
    // double tmp2 = (sqrt(pow(y2 - y1, 2) + pow(x1 - x2, 2)));
    // double res = tmp1/tmp2 ;
    // std::cout<<res<<std::endl;

    // test_linear_model();
    // test_lindex();
    test_interval_tree();
    return 0;
}
