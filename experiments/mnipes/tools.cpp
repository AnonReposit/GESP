#include "tools.hpp"
#include "M_NIPES.hpp"
#include <cmath>


class _sort_indices_double
{
   private:
     double* mparr;
   public:
     _sort_indices_double(double* parr) : mparr(parr) {}
     bool operator()(int i, int j) const { return mparr[i]<mparr[j]; }
};

void stopwatch::tic()
{ 
    tt_tic = are::hr_clock::now();
}

double stopwatch::toc()
{
    auto now = are::hr_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(now - tt_tic);
    return time_span.count();
}




bool are_doubles_equal(double x1, double x2)
{
    if (fabs(x1 - x2) < 1e-70)
    {
        return true;
    }else{
        return false;
    }
    
}



void compute_order_from_double_to_int(double* v, int len, bool* order_res, bool reverse){
    
    if (reverse)
    {
        for (int i = 0; i < len; i++)
        {
            v[i] = - v[i];
        }
        
    }


    int* temp = new int[len];
    for (int i = 0; i < len; i++)
    {
        temp[i] = i;
    }
    std::sort(temp, temp+len, _sort_indices_double(v));
    for (int i = 0; i < len; i++)
    {
        order_res[temp[i]] = i;
    }
    
    if (reverse)
    {
        for (int i = 0; i < len; i++)
        {
            v[i] = - v[i];
        }
        
    }
    delete[] temp;
}


void compute_order_from_double_to_double(double* v, int len, double* order_res, bool reverse, bool respect_ties){
    int* temp = new int[len];

    double* copy_of_v = new double[len];
    memcpy(copy_of_v, v, sizeof(double)*len);

    // PrintArray(v, len);
    if (reverse)
    {
        multiply_array_with_value(v, -1, len);
    }
    
    for (int i = 0; i < len; i++)
    {
        temp[i] = i;
    }

    std::sort(temp, temp+len, _sort_indices_double(v));

    if (reverse)
    {
        multiply_array_with_value(v, -1, len);
    }

    for (int i = 0; i < len; i++)
    {
        order_res[temp[i]] = (double) i;
    }


    
    
    if (respect_ties)
    {
        if (len == 1)
        {
            v[0] = 0.0;
            return;
        }
        assert(len  >= 2);
        int n_equal = 0;
        double last;


        for (int i = 1; i < len; i++)
        {
            last = copy_of_v[temp[i-1]];

            if (are_doubles_equal(last, copy_of_v[temp[i]]))
            {
                n_equal++;
            }

            if (!are_doubles_equal(last, copy_of_v[temp[i]]) || i == len - 1)
            {
                int last_eq_idx = i-1;

                if(are_doubles_equal(last, copy_of_v[temp[i]])){
                    last_eq_idx = i;
                }

                if (n_equal > 0)
                {
                    double av_rank = (order_res[temp[last_eq_idx]] + order_res[temp[last_eq_idx-n_equal]]) / 2.0;

                    for (int j = 0; j <= n_equal ; j++)
                    {
                        order_res[temp[last_eq_idx-j]] = av_rank;
                    }
                    n_equal = 0;
                }
            }
        }
    }

    // PrintArray(order_res, len);
    // cout << endl;
    // cout << "---\n";
        
    delete[] copy_of_v;
    delete[] temp;
}

double average(std::vector<double> v)
{
    return accumulate( v.begin(), v.end(), 0.0)/v.size();              

}


std::string hash_string(const std::string &str)
{

    long unsigned int res = 0;
    long unsigned int i = 0;
    for (char const &c : str)
    {
        i = (i + 1) % 133;
        res += ((long unsigned int)c) * i + i;
        res %= 890000UL;
    }
    return to_string(static_cast<long unsigned int> ((double) res) % 89000L + 1000UL);

}
