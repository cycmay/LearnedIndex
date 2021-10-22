#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include "LIndex/LIndex_model.h"
#include "LIndex/LIndex_model_set.h"
#include "LIndex/Lindex.h"
#include "LIndex_model_impl.h"

#include "LIndex/matplotlibcpp.h"

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
    std::vector<Key> keys;
    std::vector<uint64_t> positions(keys.size());
    std::vector<uint64_t> values(keys.size());

    typedef Key key_t;
    LIndex::LModel<key_t> test;

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
    test.training(keys, positions);
    printf("min_key: %ld, max_key %ld\n", test.min_key.data, test.max_key.data);
    printf("weight: [%.3f, %.3f]\n", test.weights[0], test.weights[1]);
    printf("loss: %.3f\n", test.loss);

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


//删除字符串中空格，制表符tab等无效字符
std::string trim(std::string& str)
{
	//str.find_first_not_of(" \t\r\n"),在字符串str中从索引0开始，返回首次不匹配"\t\r\n"的位置
	str.erase(0,str.find_first_not_of(" \t\r\n"));
	str.erase(str.find_last_not_of(" \t\r\n") + 1);
	return str;
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

    lindex.init(keys, positions);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> rand_keynumber(0, Keys_number);
    for(size_t i=0;i<5;i++){
        uint64_t p = rand_keynumber(gen);
        printf("Key [%ld] predict-postion:[%ld] real-positon [%ld]\n", keys[p].data, lindex.get(keys[p]), (uint64_t)positions[p]);
    }
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

    test_linear_model();
    test_lindex();
    return 0;
}
