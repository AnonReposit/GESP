#ifndef NIPES_HPP
#define NIPES_HPP


#if defined (VREP)
#include "v_repLib.h"
#elif defined (COPPELIASIM)
#include "simLib.h"
#endif

#include "simulatedER/mazeEnv.h"
#include "ARE/learning/ipop_cmaes.hpp"
#include "ARE/learning/Novelty.hpp"
#include "simulatedER/nn2/NN2Individual.hpp"
#include "ARE/Settings.h"
#include "obstacleAvoidance.hpp"
#include "../mnipes/tools.hpp"

#define BESTASREF_FITNESS_ARRAY_SIZE 2000

namespace are{


typedef enum DescriptorType{
    FINAL_POSITION = 0,
    VISITED_ZONES = 1
}DescriptorType;

class NIPESIndividual : public sim::NN2Individual
{
public:
    NIPESIndividual() : sim::NN2Individual(){}
    NIPESIndividual(const Genome::Ptr& morph_gen,const NNParamGenome::Ptr& ctrl_gen)
        : sim::NN2Individual(morph_gen,ctrl_gen){}
    NIPESIndividual(const NIPESIndividual& ind)
        : sim::NN2Individual(ind),
          visited_zones(ind.visited_zones),
          descriptor_type(ind.descriptor_type){}


    Eigen::VectorXd descriptor() override;
    void set_visited_zones(const Eigen::MatrixXi& vz){visited_zones = vz;}
    void set_descriptor_type(DescriptorType dt){descriptor_type = dt;}
    void set_max_eval_time(float in_max_eval_time){this->max_eval_time = in_max_eval_time;}
    float get_max_eval_time(){return max_eval_time;}
    
    friend class boost::serialization::access;
    template<class archive>
    void serialize(archive &arch, const unsigned int v)
    {
        arch & boost::serialization::base_object<NN2Individual>(*this);
        arch & max_eval_time;
        arch & observed_fintesses;
        arch & fitness_checkpoints;
        arch & bestasref_ref_fitnesses;
        arch & bestasref_observed_fitnesses;
        arch & consumed_runtime;
    }

    std::string to_string() override;
    void from_string(const std::string &str) override;

    double observed_fintesses[20] = {0};
    double fitness_checkpoints[20] = {0};
    double bestasref_ref_fitnesses[BESTASREF_FITNESS_ARRAY_SIZE] = {0};
    double bestasref_observed_fitnesses[BESTASREF_FITNESS_ARRAY_SIZE] = {0};
    double consumed_runtime = 0;

private:

    Eigen::MatrixXi visited_zones;
    DescriptorType descriptor_type = FINAL_POSITION;
    float max_eval_time = 0;
};

class NIPES : public EA
{
public:
    NIPES() : EA(){}
    NIPES(const misc::RandNum::Ptr& rn, const settings::ParametersMapPtr& param) : EA(rn, param){}
    ~NIPES(){
        cmaStrategy.reset();
    }

    void init() override;
    void epoch() override;
    void init_next_pop() override;
    bool update(const Environment::Ptr&) override;

    void setObjectives(size_t indIdx, const std::vector<double> &objectives) override;

    bool is_finish() override;
    bool finish_eval(const Environment::Ptr &env) override;
    void write_measure_ranks_to_results();
    void updateNoveltyEnergybudgetArchive();
    void cma_iteration();
    void print_fitness_iteration();
    void write_results();
    double getFitness(const Environment::Ptr &env);
    void savefCheckpoints();
    void loadfCheckpoints();
    void bestasrefGetfCheckpointsFromIndividual(int individualIndex);
    void getfCheckpointsFromIndividuals();

    std::string compute_population_genome_hash();
    std::string getIndividualHash(Individual::Ptr ind);

    void set_currentMaxEvalTime(double new_currentMaxEvalTime);
    double get_currentMaxEvalTime();

    bool restarted(){return !cmaStrategy->log_stopping_criterias.empty();}
    std::string pop_stopping_criterias(){
        std::string res = cmaStrategy->log_stopping_criterias.back();
        cmaStrategy->log_stopping_criterias.pop_back();
        return res;
    }
    const std::vector<Eigen::VectorXd> &get_archive(){return archive;}

protected:
    IPOPCMAStrategy::Ptr cmaStrategy;
    cma::CMASolutions best_run;
    bool _is_finish = false;
    std::vector<Eigen::VectorXd> archive;
    int n_iterations_isReevaluating = 0;
    double og_maxEvalTime;
    stopwatch sw = stopwatch();
    stopwatch total_time_sw = stopwatch();
    double best_fitness = -__DBL_MAX__;
    bool isReevaluating=false;

    std::string result_filename;
    std::string subexperiment_name;

    double total_time_simulating;

    // Vars for init_next_pop
    int pop_size = -1;
    dMat new_samples; 
    int nbr_weights; 

    int nbr_bias;
    std::vector<double> weights;
    std::vector<double> biases;    

    int n_of_halvings;
    // In the first n_of_halvings positions, contains the runtimes in which we should check for minimum fitness.
    bool update_fitness_checkpoints=false;
    double time_checkpoints[20];
    double fitness_checkpoints[20];
    double bestasref_ref_fitnesses[BESTASREF_FITNESS_ARRAY_SIZE] = {0};
    long unsigned int tick;
    long bestasref_size_of_fitnesses;
    std::vector<bool> finish_eval_array;
};

}

#endif //NIPES_HPP


