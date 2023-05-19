#include "NIPES.hpp"
#include <sstream>
#include <ctime>
#include <fstream>
#include <sstream>
#include <regex>
#include "simulatedER/nn2/NN2Individual.hpp"
#include "ARE/Individual.h"
#include <math.h>
#include <fstream>
#include <iostream>

const char *build_str = "NIPES.cpp compilation time: " VERSION " " __DATE__ " " __TIME__;


using namespace are;

Eigen::VectorXd NIPESIndividual::descriptor()
{
    if(descriptor_type == FINAL_POSITION){
        double arena_size = settings::getParameter<settings::Double>(parameters,"#arenaSize").value;
        Eigen::VectorXd desc(3);
        desc << (final_position[0]+arena_size/2.)/arena_size, (final_position[1]+arena_size/2.)/arena_size, (final_position[2]+arena_size/2.)/arena_size;
        return desc;
    }else if(descriptor_type == VISITED_ZONES){
        Eigen::MatrixXd vz = visited_zones.cast<double>();
        Eigen::VectorXd desc(Eigen::Map<Eigen::VectorXd>(vz.data(),vz.cols()*vz.rows()));
        return desc;
    }
}

std::string NIPESIndividual::to_string()
{
    std::stringstream sstream;
    boost::archive::text_oarchive oarch(sstream);
    oarch.register_type<NIPESIndividual>();
    oarch.register_type<NN2Individual>();
    oarch.register_type<NNParamGenome>();
    oarch << *this;
    return sstream.str();
}

void NIPESIndividual::from_string(const std::string &str){
    std::stringstream sstream;
    sstream << str;
    boost::archive::text_iarchive iarch(sstream);
    iarch.register_type<NIPESIndividual>();
    iarch.register_type<NN2Individual>();
    iarch.register_type<NNParamGenome>();
    iarch >> *this;

    //set the parameters and randNum of the genome because their are not included in the serialisation
    ctrlGenome->set_parameters(parameters);
    ctrlGenome->set_randNum(randNum);
    morphGenome->set_parameters(parameters);
    morphGenome->set_randNum(randNum);
}

double NIPES::get_currentMaxEvalTime()
{
    if (population.size() == 0)
    {
        // std::cout << "get_currentMaxEvalTime() -> population.size() == 0" << std::endl;
        return 0.0;
    }
    
    auto NIPESind = std::dynamic_pointer_cast<NIPESIndividual>(population[0]);
    return NIPESind->get_max_eval_time();
}

double NIPES::getFitness(const Environment::Ptr &env)
{
    Individual::Ptr ind = std::dynamic_pointer_cast<sim::NN2Individual>(population[currentIndIndex]);
    std::vector<double> fitness = env->fitnessFunction(ind);
    return fitness[0];
}

void NIPES::set_currentMaxEvalTime(double new_currentMaxEvalTime)
{
    if (population.size() == 0)
    {
        return;
    }

    for (auto ind: population)
    {
        auto NIPESind = std::dynamic_pointer_cast<NIPESIndividual>(ind);
        NIPESind->set_max_eval_time((float) new_currentMaxEvalTime);
    }
    // std::cout << "set_currentMaxEvalTime() -> " << get_currentMaxEvalTime() << std::endl;
}


void NIPES::init(){

    total_time_sw.tic();
    total_time_simulating = 0.0;
    og_maxEvalTime = settings::getParameter<settings::Float>(parameters,"#maxEvalTime").value;
    subexperiment_name = settings::getParameter<settings::String>(parameters,"#subexperimentName").value;
    std::cout << build_str << std::endl;

    if (
        subexperiment_name != "measure_ranks" && 
        subexperiment_name != "constant"      &&
        subexperiment_name != "bestasref")
    {
        std::cerr << "ERROR: subexperimentName = " << subexperiment_name << " not recognized." << std::endl;
        exit(1);
    }
    



    result_filename =  settings::getParameter<settings::String>(parameters,"#repository").value + 
                       std::string("/") + 
                       settings::getParameter<settings::String>(parameters,"#resultFile").value;


    int lenStag = settings::getParameter<settings::Integer>(parameters,"#lengthOfStagnation").value;

    int pop_size = settings::getParameter<settings::Integer>(parameters,"#populationSize").value;
    this->pop_size = pop_size;
    float max_weight = settings::getParameter<settings::Float>(parameters,"#MaxWeight").value;
    double step_size = settings::getParameter<settings::Double>(parameters,"#CMAESStep").value;
    double ftarget = settings::getParameter<settings::Double>(parameters,"#FTarget").value;
    bool verbose = settings::getParameter<settings::Boolean>(parameters,"#verbose").value;
    bool elitist_restart = settings::getParameter<settings::Boolean>(parameters,"#elitistRestart").value;
    double novelty_ratio = settings::getParameter<settings::Double>(parameters,"#noveltyRatio").value;
    double novelty_decr = settings::getParameter<settings::Double>(parameters,"#noveltyDecrement").value;
    float pop_stag_thres = settings::getParameter<settings::Float>(parameters,"#populationStagnationThreshold").value;

    Novelty::k_value = settings::getParameter<settings::Integer>(parameters,"#kValue").value;
    Novelty::novelty_thr = settings::getParameter<settings::Double>(parameters,"#noveltyThreshold").value;
    Novelty::archive_adding_prob = settings::getParameter<settings::Double>(parameters,"#archiveAddingProb").value;

    int nn_type = settings::getParameter<settings::Integer>(parameters,"#NNType").value;
    const int nb_input = settings::getParameter<settings::Integer>(parameters,"#NbrInputNeurones").value;
    const int nb_hidden = settings::getParameter<settings::Integer>(parameters,"#NbrHiddenNeurones").value;
    const int nb_output = settings::getParameter<settings::Integer>(parameters,"#NbrOutputNeurones").value;

    if(nn_type == settings::nnType::FFNN)
        NN2Control<ffnn_t>::nbr_parameters(nb_input,nb_hidden,nb_output,nbr_weights,nbr_bias);
    else if(nn_type == settings::nnType::RNN)
        NN2Control<rnn_t>::nbr_parameters(nb_input,nb_hidden,nb_output,nbr_weights,nbr_bias);
    else if(nn_type == settings::nnType::ELMAN)
        NN2Control<elman_t>::nbr_parameters(nb_input,nb_hidden,nb_output,nbr_weights,nbr_bias);
    else {
        std::cerr << "unknown type of neural network" << std::endl;
        return;
    }


    std::vector<double> initial_point = randomNum->randVectd(-max_weight,max_weight,nbr_weights + nbr_bias);


    double lb[nbr_weights+nbr_bias], ub[nbr_weights+nbr_bias];
    for(int i = 0; i < nbr_weights+nbr_bias; i++){
        lb[i] = -max_weight;
        ub[i] = max_weight;
    }

    geno_pheno_t gp(lb,ub,nbr_weights+nbr_bias);

    cma::CMAParameters<geno_pheno_t> cmaParam(initial_point,step_size,pop_size,randomNum->getSeed(),gp);
    cmaParam.set_ftarget(ftarget);
    cmaParam.set_quiet(!verbose);


    cmaStrategy.reset(new IPOPCMAStrategy([](const double*,const int&)->double{},cmaParam));
    cmaStrategy->set_elitist_restart(elitist_restart);
    cmaStrategy->set_length_of_stagnation(lenStag);
    cmaStrategy->set_novelty_ratio(novelty_ratio);
    cmaStrategy->set_novelty_decr(novelty_decr);
    cmaStrategy->set_pop_stag_thres(pop_stag_thres);


    new_samples = cmaStrategy->ask();
    weights.resize(nbr_weights);
    biases.resize(nbr_bias);

    for(int u = 0; u < pop_size; u++){

        for(int v = 0; v < nbr_weights; v++)
            weights[v] = new_samples(v,u);
        for(int w = nbr_weights; w < nbr_weights+nbr_bias; w++)
            biases[w-nbr_weights] = new_samples(w,u);

        EmptyGenome::Ptr morph_gen(new EmptyGenome);
        NNParamGenome::Ptr ctrl_gen(new NNParamGenome);
        ctrl_gen->set_weights(weights);
        ctrl_gen->set_biases(biases);
        Individual::Ptr ind(new NIPESIndividual(morph_gen,ctrl_gen));
        ind->set_parameters(parameters);
        ind->set_randNum(randomNum);
        population.push_back(ind);
    }

    set_currentMaxEvalTime(og_maxEvalTime);



    if (subexperiment_name == "bestasref")
    {
        // Update fitness checkpoints after first iteration.
        update_fitness_checkpoints = true;
        std::fill_n(bestasref_ref_fitnesses, BESTASREF_FITNESS_ARRAY_SIZE, -__DBL_MAX__);
        float time_delta = settings::getParameter<settings::Float>(parameters, "#timeStep").value;
        bestasref_size_of_fitnesses = lround(og_maxEvalTime / time_delta);
        finish_eval_array.resize(pop_size);
        for (size_t i = 0; i < pop_size; i++)
        {
            finish_eval_array[i] = false;
        }
        savefCheckpoints();
    }
}


void NIPES::write_measure_ranks_to_results()
{
    std::vector<double> f_scores(population.size());
    std::vector<double> ranks(population.size());
    f_scores.resize(population.size());
    for (size_t i = 0; i < population.size(); i++)
    {   
        auto ind = population[i];
        double fitness;
        fitness = std::dynamic_pointer_cast<NIPESIndividual>(ind)->getObjectives()[0];
        f_scores[i] = fitness;
    }

    compute_order_from_double_to_double(f_scores.data(),population.size(),ranks.data(), false, true);

 
    std::cout << "- Saving population " << compute_population_genome_hash() << std::endl;

    std::stringstream res_to_write;
    res_to_write << std::setprecision(28);
    res_to_write << settings::getParameter<settings::String>(parameters,"#preTextInResultFile").value << ",";
    res_to_write << get_currentMaxEvalTime()<< ",";
    res_to_write << numberEvaluation << ",";
    res_to_write << "(";
    res_to_write << iterable_to_str(ranks.begin(), ranks.end());
    res_to_write << "),(";
    res_to_write << iterable_to_str(f_scores.begin(), f_scores.end());
    res_to_write << "),";
    res_to_write << compute_population_genome_hash();
    res_to_write << "\n";
    append_line_to_file(result_filename,res_to_write.str());
}


void NIPES::updateNoveltyEnergybudgetArchive()
{
    double energy_budget = settings::getParameter<settings::Double>(parameters,"#energyBudget").value;
    bool energy_reduction = settings::getParameter<settings::Boolean>(parameters,"#energyReduction").value;
    /**Energy Cost**/
    if(energy_reduction){
        for(const auto &ind : population){
            double ec = std::dynamic_pointer_cast<sim::NN2Individual>(ind)->get_energy_cost();
            if(ec > energy_budget) ec = energy_budget;
            std::dynamic_pointer_cast<sim::NN2Individual>(ind)->addObjective(1 - ec/energy_budget);
        }
    }

    /** NOVELTY **/
    if(settings::getParameter<settings::Double>(parameters,"#noveltyRatio").value > 0.){
        if(Novelty::k_value >= population.size())
            Novelty::k_value = population.size()/2;
        else Novelty::k_value = settings::getParameter<settings::Integer>(parameters,"#kValue").value;

        std::vector<Eigen::VectorXd> pop_desc;
        for(const auto& ind : population)
            pop_desc.push_back(ind->descriptor());
        //compute novelty
        for(const auto& ind : population){
            Eigen::VectorXd ind_desc = ind->descriptor();
            double ind_nov = Novelty::sparseness(Novelty::distances(ind_desc,archive,pop_desc));
            std::dynamic_pointer_cast<sim::NN2Individual>(ind)->addObjective(ind_nov);
        }

        //update archive
        for(const auto& ind : population){
            Eigen::VectorXd ind_desc = ind->descriptor();
            double ind_nov = ind->getObjectives().back();
            Novelty::update_archive(ind_desc,ind_nov,archive,randomNum);
        }
    }

}



void NIPES::cma_iteration(){

    bool verbose = settings::getParameter<settings::Boolean>(parameters,"#verbose").value;
    bool withRestart = settings::getParameter<settings::Boolean>(parameters,"#withRestart").value;
    bool incrPop = settings::getParameter<settings::Boolean>(parameters,"#incrPop").value;
    bool elitist_restart = settings::getParameter<settings::Boolean>(parameters,"#elitistRestart").value;
    std::vector<IPOPCMAStrategy::individual_t> pop;
    for (const auto &ind : population)
    {
        IPOPCMAStrategy::individual_t cma_ind;
        cma_ind.genome = std::dynamic_pointer_cast<NNParamGenome>(ind->get_ctrl_genome())->get_full_genome();
        cma_ind.descriptor = std::dynamic_pointer_cast<sim::NN2Individual>(ind)->get_final_position();
        cma_ind.objectives = std::dynamic_pointer_cast<sim::NN2Individual>(ind)->getObjectives();
        pop.push_back(cma_ind);
    }

    cmaStrategy->set_population(pop);
    cmaStrategy->eval();
    cmaStrategy->tell();
    bool stop = cmaStrategy->stop();
    //    if(cmaStrategy->have_reached_ftarget()){
    //        _is_finish = true;
    ////        return;
    //    }

        if(withRestart && stop){
            if(verbose)
                std::cout << "Restart !" << std::endl;

            cmaStrategy->capture_best_solution(best_run);

            if(incrPop)
                cmaStrategy->lambda_inc();

            cmaStrategy->reset_search_state();
            if(!elitist_restart){
                float max_weight = settings::getParameter<settings::Float>(parameters,"#MaxWeight").value;
                cmaStrategy->get_parameters().set_x0(-max_weight,max_weight);
            }
        }
    std::cout << "Popsize: " << population.size() << ", " << population.max_size();
}


void NIPES::print_fitness_iteration()
{
    for (const auto &ind : population)
    {
        std::cout << "objective, novelty: " << iterable_to_str(std::dynamic_pointer_cast<NIPESIndividual>(ind)->getObjectives().begin(), std::dynamic_pointer_cast<NIPESIndividual>(ind)->getObjectives().end()) << std::endl;
        double fitness;
        fitness = std::dynamic_pointer_cast<NIPESIndividual>(ind)->getObjectives()[0];

        if (fitness > best_fitness)
        {
            best_fitness = fitness;
        }
    }
}

std::string NIPES::compute_population_genome_hash()
{
    std::ostringstream strs;
    // strs << std::setprecision(0);
    for (const auto &ind : population)
    {
        strs << "#" << getIndividualHash(ind);
    }
    std::string str = strs.str();
    return str;
}

void NIPES::loadfCheckpoints()
{

    // std::ifstream in("fitness_checkpoints.data", std::ios::in | std::ios::binary);
    // in.read((char *)&fitness_checkpoints, sizeof fitness_checkpoints);

    // // see how many bytes have been read
    // std::cout << in.gcount() << " bytes read\n";
    // in.close();


    // std::cout << "loaded fitness_checkpoints: ";
    // PrintArray(fitness_checkpoints, n_of_halvings);

    if (subexperiment_name == "bestasref")
    {
        // bestasref
        for (size_t i = 0; i < bestasref_size_of_fitnesses; i++)
        {
            auto ind = population[currentIndIndex];
            auto NIPESind = std::dynamic_pointer_cast<NIPESIndividual>(ind);
            bestasref_ref_fitnesses[i] = NIPESind->bestasref_ref_fitnesses[i];
        }
    }
}

void NIPES::bestasrefGetfCheckpointsFromIndividual(int individualIndex)
{
        const double EPSILON = 0.0000001;
        auto ind = population[individualIndex];
        auto NIPESind = std::dynamic_pointer_cast<NIPESIndividual>(ind);
        double fitness = NIPESind->getObjectives()[0];
        if (abs(fitness - best_fitness) < EPSILON)
        {
            std::cout << "We DO NOT relaxing best fitness refs if fitness is equal to bk." << std::endl;

            // std::cout << "Relaxing best fitness refs" << std::endl;
            // PrintArray(NIPESind->bestasref_observed_fitnesses, bestasref_size_of_fitnesses);
            // std::cout << "    -     " << std::endl;
            // PrintArray(bestasref_ref_fitnesses, bestasref_size_of_fitnesses);
            // std::cout << "* * *" << std::endl;

            // for (size_t i = 0; i < bestasref_size_of_fitnesses; i++)
            // {
            //     // min() makes the stopping criteria less and less strict
            //     bestasref_ref_fitnesses[i] = std::min(bestasref_ref_fitnesses[i], NIPESind->bestasref_observed_fitnesses[i]);

            //     // // max() makes the stopping criteria more and more strict
            //     // bestasref_ref_fitnesses[i] = std::max(bestasref_ref_fitnesses[i], NIPESind->bestasref_observed_fitnesses[i]);
            // }
        }
        else if (fitness > best_fitness)
        {
            if (subexperiment_name == "constant")
            {
                std::cout << "Updating new best fitness." << std::endl;
                best_fitness = fitness;
                return;
            }

            std::cout << "Updating fitnesses due to new best fitness." << std::endl;
            PrintArray(NIPESind->bestasref_observed_fitnesses, bestasref_size_of_fitnesses);
            std::cout << "    -     " << std::endl;
            PrintArray(bestasref_ref_fitnesses, bestasref_size_of_fitnesses);
            std::cout << "* * *" << std::endl;

            best_fitness = fitness;
            for (size_t i = 0; i < bestasref_size_of_fitnesses; i++)
            {
                bestasref_ref_fitnesses[i] = NIPESind->bestasref_observed_fitnesses[i];
            }
        }

}


void NIPES::getfCheckpointsFromIndividuals()
{
    std::cout << "getfCheckpointsFromIndividuals(): " << std::endl;



    if (subexperiment_name == "bestasref")
    {
        // bestasref
        for (size_t j = 0; j < pop_size; j++)
        {
            bestasrefGetfCheckpointsFromIndividual(j);
        }
    }
}

void NIPES::savefCheckpoints()
{

    //     std::cout << "saving fitness_checkpoints: " << std::endl;
    //     PrintArray(fitness_checkpoints, n_of_halvings);

    //     std::ofstream out("fitness_checkpoints.data", std::ios::out | std::ios::binary);
    //     if (!out)
    //     {
    //         std::cout << "Cannot open file 'fitness_checkpoints.data' on write mode. Exit...";
    //         exit(1);
    //     }

    //   out.write((char *) &fitness_checkpoints, sizeof fitness_checkpoints);
    //   out.close();


    if (subexperiment_name == "bestasref")
    {
        // bestasref
        for (size_t j = 0; j < pop_size; j++)
        {
            auto ind = population[j];
            for (size_t i = 0; i < bestasref_size_of_fitnesses; i++)
            {
                auto NIPESind = std::dynamic_pointer_cast<NIPESIndividual>(ind);
                NIPESind->bestasref_ref_fitnesses[i] = bestasref_ref_fitnesses[i];
            }
        }
    }
}

void NIPES::epoch(){

    if (pop_size == 0)
    {
        return;
    }

    const static std::string preTextInResultFile = settings::getParameter<settings::String>(parameters,"#preTextInResultFile").value;
    std::cout << "- epoch(), " << "preTextInResultFile=" << preTextInResultFile << ", maxruntime=" << get_currentMaxEvalTime()<< ", evals=" << numberEvaluation <<", isReeval=" << isReevaluating << ", gen = " << get_generation() << ", time=" << std::time(nullptr) << std::endl;



    if (subexperiment_name == "bestasref")
    {
        for (size_t j = 0; j < pop_size; j++)
        {
            auto ind = population[j];
            auto NIPESind = std::dynamic_pointer_cast<NIPESIndividual>(ind);
            total_time_simulating += NIPESind->consumed_runtime;
        }

        if (update_fitness_checkpoints)
        {
            getfCheckpointsFromIndividuals();
            savefCheckpoints();
            update_fitness_checkpoints = false;
        }



        write_results();
        updateNoveltyEnergybudgetArchive();
        cma_iteration();
        print_fitness_iteration();
        for (size_t i = 0; i < pop_size; i++)
        {
            finish_eval_array[i] = false;
        }
        
        return;
    }

    else if (subexperiment_name == "constant")
    {
        total_time_simulating += pop_size * get_currentMaxEvalTime();
        write_results();
        updateNoveltyEnergybudgetArchive();
        cma_iteration();
        print_fitness_iteration();
        return;
    }

    else if (subexperiment_name == "measure_ranks")
    {   
        const int REEVALUATE_EVERY_N_GENS = 10;
        const int N_LINSPACE_SAMPLES_RUNTIME = 6;
        const double RUNTIMES[N_LINSPACE_SAMPLES_RUNTIME] = {0.9, 1.8, 3.7, 7.5, 15.0, 30.0};
        if (generation % REEVALUATE_EVERY_N_GENS == 0)
        {
            std::cout << "MaxEvalTime: " << RUNTIMES[n_iterations_isReevaluating] << std::endl;
            write_measure_ranks_to_results();
            set_currentMaxEvalTime(RUNTIMES[n_iterations_isReevaluating]);
            isReevaluating = n_iterations_isReevaluating < N_LINSPACE_SAMPLES_RUNTIME;
            if(isReevaluating)
            {
                std::cout << "- Reevaluating... " << std::endl;
            }
        }
        if (isReevaluating)
        {
            set_generation(get_generation() - 1);
            generation = get_generation();
            numberEvaluation -= population.size();
            n_iterations_isReevaluating++;
        }
        else
        {
            n_iterations_isReevaluating = 0;
            updateNoveltyEnergybudgetArchive();
            cma_iteration();
            set_currentMaxEvalTime(RUNTIMES[N_LINSPACE_SAMPLES_RUNTIME-1]);
        }
        print_fitness_iteration();
        return;

    }







}

void NIPES::init_next_pop(){

    if (!isReevaluating)
    {
        new_samples = cmaStrategy->ask();
        nbr_weights = std::dynamic_pointer_cast<NNParamGenome>(population[0]->get_ctrl_genome())->get_weights().size();
        nbr_bias = std::dynamic_pointer_cast<NNParamGenome>(population[0]->get_ctrl_genome())->get_biases().size();
        weights.resize(nbr_weights);
        biases.resize(nbr_bias);
    }
    pop_size = cmaStrategy->get_parameters().lambda();
    double tmp_currentMaxEvalTime = get_currentMaxEvalTime(); 
    population.clear();
    for(int i = 0; i < pop_size ; i++){

        for(int j = 0; j < nbr_weights; j++)
            weights[j] = new_samples(j,i);
        for(int j = nbr_weights; j < nbr_weights+nbr_bias; j++)
            biases[j-nbr_weights] = new_samples(j,i);

        EmptyGenome::Ptr morph_gen(new EmptyGenome);
        NNParamGenome::Ptr ctrl_gen(new NNParamGenome);
        ctrl_gen->set_weights(weights);
        ctrl_gen->set_biases(biases);
        Individual::Ptr ind(new NIPESIndividual(morph_gen,ctrl_gen));
        ind->set_parameters(parameters);
        ind->set_randNum(randomNum);
        population.push_back(ind);
    }
    set_currentMaxEvalTime(tmp_currentMaxEvalTime);
    savefCheckpoints();
}

void NIPES::setObjectives(size_t indIdx, const std::vector<double> &objectives){
    population[indIdx]->setObjectives(objectives);
}


std::string NIPES::getIndividualHash(Individual::Ptr ind)
{
    auto v = std::dynamic_pointer_cast<NNParamGenome>(ind->get_ctrl_genome())->get_full_genome();
    long unsigned int num = static_cast<long unsigned int> (average(v) * 10000000000.0) % 890000L + 100000L;
    std::stringstream ss;
    ss << num;
    return ss.str();
}


bool NIPES::update(const Environment::Ptr & env){
    std::cout << "update() " << sw.toc() << std::endl;
    sw.tic();
    numberEvaluation++;
    if(simulator_side){
        Individual::Ptr ind = population[currentIndIndex];
        std::cout << "- Evaluated genome with hash #" << getIndividualHash(ind);
        std::dynamic_pointer_cast<NIPESIndividual>(ind)->set_final_position(env->get_final_position());
        std::dynamic_pointer_cast<NIPESIndividual>(ind)->set_trajectory(env->get_trajectory());
        if(env->get_name() == "obstacle_avoidance"){
            std::dynamic_pointer_cast<NIPESIndividual>(ind)->set_visited_zones(std::dynamic_pointer_cast<sim::ObstacleAvoidance>(env)->get_visited_zone_matrix());
            std::dynamic_pointer_cast<NIPESIndividual>(ind)->set_descriptor_type(VISITED_ZONES);
        }
    std:: cout << ", fitness: " << ind->getObjectives()[0] << ", runtime: " << get_currentMaxEvalTime()<< ", traj of length " ;
    
    std::string traj = "";
    for (size_t i = 0; i < env->get_trajectory().size(); i++)
    {
        traj+= env->get_trajectory()[i].to_string();

    }
    std::cout << env->get_trajectory().size() << " and hash #" << hash_string(traj) << std::endl;
    }
    sw.tic();

    bestasrefGetfCheckpointsFromIndividual(currentIndIndex);
    

    return true;
}


void NIPES::write_results()
{
    static long int lastNumberEvaluationWrite = -1;

    if (numberEvaluation == lastNumberEvaluationWrite)
    {
        return;
    }
    lastNumberEvaluationWrite = numberEvaluation;

    double total_time_according_to_sw = total_time_sw.toc();
    std::stringstream res_to_write;
    res_to_write << std::setprecision(28);
    res_to_write << settings::getParameter<settings::String>(parameters, "#preTextInResultFile").value;
    res_to_write << ",";
    res_to_write << best_fitness;
    res_to_write << ",";
    res_to_write << total_time_according_to_sw;
    res_to_write << ",";
    res_to_write << total_time_simulating;
    res_to_write << ",";
    res_to_write << get_currentMaxEvalTime();
    res_to_write << ",";
    res_to_write << numberEvaluation;

    if(subexperiment_name == "bestasref")
    {
        res_to_write << ",(";
        for (size_t j = 0; j < pop_size; j++)
        {
            auto ind = population[j];
            auto NIPESind = std::dynamic_pointer_cast<NIPESIndividual>(ind);
            res_to_write <<  NIPESind->consumed_runtime << ";";
        }
        res_to_write << ")";
    }

    res_to_write << std::endl;
    append_line_to_file(result_filename, res_to_write.str());
}


bool NIPES::is_finish(){
    int maxNbrEval = settings::getParameter<settings::Integer>(parameters,"#maxNbrEval").value;
    double maxComputationTime = settings::getParameter<settings::Double>(parameters,"#maxComputationTime").value;
    double total_time = total_time_sw.toc();


    if ((total_time > maxComputationTime || numberEvaluation > maxNbrEval + population.size()) && !isReevaluating)
    {
        std::cout << "numberEvaluation: " << numberEvaluation << std::endl;
        std::cout << "maxNbrEval: " << maxNbrEval << std::endl;
        std::cout << "Best fitness: " << best_fitness << std::endl;
        std::cout << "Total runtime: " << total_time_sw.toc() << std::endl;
        std::cout << "maxComputationTime: " << maxComputationTime << std::endl;
        write_results();
        return true;
    }
    else
    {
        return false;
    }
}

bool NIPES::finish_eval(const Environment::Ptr &env){

    // std::cout << "simGetSimulationTime()" << simGetSimulationTime() << std::endl;

    // we need an offset of 0.3 seconds, because the simulation will halt the third
    // time true is returned.
    // static long unsigned int checks = 0;
    // checks++;

    // Save fitness and check if stopping is necessary
    if (subexperiment_name == "bestasref")
    {
        static float time_delta = settings::getParameter<settings::Float>(parameters, "#timeStep").value;

        // in the first iteration
        if (simGetSimulationTime() < time_delta*1.5)
        {
            finish_eval_array[currentIndIndex] = false;
            tick = 0;
        }

        // need to return true 3 times to really stop.
        if (finish_eval_array[currentIndIndex])
        {
            return true;
        }

        static int instance_type = settings::getParameter<settings::Integer>(parameters, "#instanceType").value;
        static float bestasrefGrace = settings::getParameter<settings::Float>(parameters, "#bestasrefGrace").value;
        static long unsigned bestasrefGraceTicks = lround(bestasrefGrace / time_delta);
        

        // in the first iteration
        if (simGetSimulationTime() < time_delta * 0.5)
        {   
            if (instance_type == 1)
            {
                loadfCheckpoints();
            }
        }

        // save fitness
        auto NIPESind = std::dynamic_pointer_cast<NIPESIndividual>(population[currentIndIndex]);
        NIPESind->bestasref_observed_fitnesses[tick] = getFitness(env);
        NIPESind->consumed_runtime = simGetSimulationTime() + 0.3;


        if (tick >= bestasrefGraceTicks && getFitness(env) < bestasref_ref_fitnesses[tick - bestasrefGraceTicks])
        {
            std::cout << "Finish bestasref. " << std::endl;
            // PrintArray(bestasref_ref_fitnesses, tick);
            // PrintArray(NIPESind->bestasref_observed_fitnesses, tick);
            finish_eval_array[currentIndIndex] = true;
            return true;
        }
    }

    if ((double) simGetSimulationTime() + 0.3 > get_currentMaxEvalTime())
    {
        // checks = 0;
        // std::cout << "True returned in finish_eval()" << std::endl;
        // std::cout << "checks=" << checks <<  ", simGetSimulationTime() = " << simGetSimulationTime() << std::endl;
        return true;
    }

    float tPos[3];
    tPos[0] = settings::getParameter<settings::Double>(parameters,"#target_x").value;
    tPos[1] = settings::getParameter<settings::Double>(parameters,"#target_y").value;
    tPos[2] = settings::getParameter<settings::Double>(parameters,"#target_z").value;
    double fTarget = settings::getParameter<settings::Double>(parameters,"#FTarget").value;
    double arenaSize = settings::getParameter<settings::Double>(parameters,"#arenaSize").value;

    auto distance = [](float* a,float* b) -> double
    {
        return std::sqrt((a[0] - b[0])*(a[0] - b[0]) +
                         (a[1] - b[1])*(a[1] - b[1]) +
                         (a[2] - b[2])*(a[2] - b[2]));
    };

    tick++;

    int handle = std::dynamic_pointer_cast<sim::Morphology>(population[currentIndIndex]->get_morphology())->getMainHandle();
    float pos[3];
    simGetObjectPosition(handle,-1,pos);
    double dist = distance(pos,tPos)/sqrt(2*arenaSize*arenaSize);

    if(dist < fTarget){
        std::cout << "STOP !" << std::endl;
    }

    return  dist < fTarget;
}

