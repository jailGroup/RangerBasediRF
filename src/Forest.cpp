/*-------------------------------------------------------------------------------
 This file is part of Ranger.

 Ranger is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Ranger is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with Ranger. If not, see <http://www.gnu.org/licenses/>.

 Written by:

 Marvin N. Wright
 Institut für Medizinische Biometrie und Statistik
 Universität zu Lübeck
 Ratzeburger Allee 160
 23562 Lübeck
 Germany

 http://www.imbs-luebeck.de
 #-------------------------------------------------------------------------------*/

#include <math.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <ctime>
#include <math.h>
#ifndef OLD_WIN_R_BUILD
#include <thread>
#include <chrono>
#endif

#include <unistd.h>

#include <mpi.h>

#include "utility.h"
#include "Forest.h"
#include "DataChar.h"
#include "DataDouble.h"
#include "DataFloat.h"


Forest::Forest() :
    verbose_out(0), num_trees(DEFAULT_NUM_TREE), mtry(0), min_node_size(0), num_variables(0), num_independent_variables(
        0), seed(0), dependent_varID(0), num_samples(0), prediction_mode(false), memory_mode(MEM_DOUBLE), sample_with_replacement(
        true), memory_saving_splitting(false), splitrule(DEFAULT_SPLITRULE), predict_all(false), keep_inbag(false), sample_fraction(
        1), holdout(false), prediction_type(DEFAULT_PREDICTIONTYPE), num_random_splits(DEFAULT_NUM_RANDOM_SPLITS), alpha(
        DEFAULT_ALPHA), minprop(DEFAULT_MINPROP), num_threads(DEFAULT_NUM_THREADS), data(0), overall_prediction_error(
        0), importance_mode(DEFAULT_IMPORTANCE_MODE), progress(0) {
}

Forest::~Forest() {
  for (auto& tree : trees) {
    delete tree;
  }
}

// #nocov start
void Forest::initCpp(std::string dependent_variable_name, MemoryMode memory_mode, std::string input_file, uint mtry,
    std::string output_prefix, uint num_trees, std::ostream* verbose_out, uint seed, uint num_threads,
    std::string load_forest_filename, ImportanceMode importance_mode, uint min_node_size,
    std::string split_select_weights_file, std::vector<std::string>& always_split_variable_names,
    std::string status_variable_name, bool sample_with_replacement, std::vector<std::string>& unordered_variable_names,
    bool memory_saving_splitting, SplitRule splitrule, std::string case_weights_file, bool predict_all,
    double sample_fraction, double alpha, double minprop, bool holdout, PredictionType prediction_type,
    uint num_random_splits, int useMPI, int rank, int size, std::string outputDirectory, std::ostream* verbose_time) {

            //std::cout << "In Forest initCpp with rank: " << rank << "\n";
        this->verbose_out = verbose_out;
        this->rank = rank;
        this->outputDirectory = outputDirectory;
        std::cout << "initCPP directory: " << outputDirectory << "\n";
        // Initialize data with memmode
        switch (memory_mode) {
        case MEM_DOUBLE:
          data = new DataDouble();
          break;
        case MEM_FLOAT:
          data = new DataFloat();
          break;
        case MEM_CHAR:
          data = new DataChar();
          break;
        }
        //std::cout << "In Forest initCpp, after memory mode\n";
        //std::cout << "Rank: " << rank << '\n';
        // Load data
        *verbose_out << "Loading input file: " << input_file << "." << std::endl;
        std::cout << "In Forest initCpp, after verbose out\n";
        bool rounding_error;

        std::chrono::time_point<std::chrono::system_clock> startRead, endRead;
				startRead = std::chrono::system_clock::now();

        //if (useMPI == 1){
          //for (int i = 0; i < size; i++){
            //MPI_Barrier(MPI_COMM_WORLD);
            //if (i == rank){
              rounding_error = data->loadFromFile(input_file);
            //}
          //}
        //}
        //else{
          //rounding_error = data->loadFromFile(input_file);
        //}

        endRead = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsedRead = endRead - startRead;
				*verbose_out << "Data read processing time: " << elapsedRead.count() << "s." << std::endl;
        *verbose_time << "File read time: " << elapsedRead.count() << "s." << std::endl;

        if (rounding_error) {
          *verbose_out << "Warning: Rounding or Integer overflow occurred. Use FLOAT or DOUBLE precision to avoid this."
              << std::endl;
        }
        *verbose_out << "Finished loading input file\n";
        //std::cout << "In forest initCpp, after rounding error\n";
        // Set prediction mode
        bool prediction_mode = false;
        if (!load_forest_filename.empty()) {
          prediction_mode = true;
        }
        //std::cout << "In Forest.cpp, before call to init\n";
        std::cout << std::flush;
        // Call other init function
        init(dependent_variable_name, memory_mode, data, mtry, output_prefix, num_trees, seed, num_threads, importance_mode,
            min_node_size, status_variable_name, prediction_mode, sample_with_replacement, unordered_variable_names,
            memory_saving_splitting, splitrule, predict_all, sample_fraction, alpha, minprop, holdout, prediction_type,
            num_random_splits, useMPI);
        //std::cout << "In Forest.cpp, directly after call to init\n";
        std::cout << std::flush;

        if (prediction_mode) {
          loadFromFile(load_forest_filename);
        }
        // Set variables to be always considered for splitting
        if (!always_split_variable_names.empty()) {
          setAlwaysSplitVariables(always_split_variable_names);
        }

        // TODO: Read 2d weights for tree-wise split select weights
        // Load split select weights from file
        // if (useMPI == 1){
        //   for (int i = 0; i < size; i++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     if (i == rank){
              // if (!split_select_weights_file.empty()) {
              //   std::vector<std::vector<double>> split_select_weights;
              //   split_select_weights.resize(1);
              //   loadDoubleVectorFromFile(split_select_weights[0], split_select_weights_file);
              //   if (split_select_weights[0].size() != num_variables - 1) {
              //     throw std::runtime_error("Number of split select weights is not equal to number of independent variables.");
              //   }
              //   setSplitWeightVector(split_select_weights);
              // }
        //     }
        //   }
        // }
        //else{
        if (!split_select_weights_file.empty()) {
          std::vector<std::vector<double>> split_select_weights;
          split_select_weights.resize(1);
          loadDoubleVectorFromFile(split_select_weights[0], split_select_weights_file);
          if (split_select_weights[0].size() != num_variables - 1) {
            throw std::runtime_error("Number of split select weights is not equal to number of independent variables.");
          }
          setSplitWeightVector(split_select_weights);
        }
      //}

        // Load case weights from file
        if (!case_weights_file.empty()) {
          loadDoubleVectorFromFile(case_weights, case_weights_file);
          if (case_weights.size() != num_samples - 1) {
            //throw std::runtime_error("Number of case weights is not equal to number of samples.");
          }
        }

        // Sample from non-zero weights in holdout mode
        if (holdout && !case_weights.empty()) {
          size_t nonzero_weights = 0;
          for (auto& weight : case_weights) {
            if (weight > 0) {
              ++nonzero_weights;
            }
          }
          this->sample_fraction = this->sample_fraction * ((double) nonzero_weights / (double) num_samples);
        }

        // Check if all catvars are coded in integers starting at 1
        if (!unordered_variable_names.empty()) {
          std::string error_message = checkUnorderedVariables(data, unordered_variable_names);
          if (!error_message.empty()) {
            throw std::runtime_error(error_message);
          }
        }
      }
      // #nocov end







      bool isZero (double i) {return (i!=0.0);}


      void Forest::initCppData(std::string dependent_variable_name, MemoryMode memory_mode, std::string input_file, uint mtry,
          std::string output_prefix, uint num_trees, std::ostream* verbose_out, uint seed, uint num_threads,
          std::string load_forest_filename, ImportanceMode importance_mode, uint min_node_size,
          std::string split_select_weights_file, std::vector<std::string>& always_split_variable_names,
          std::string status_variable_name, bool sample_with_replacement, std::vector<std::string>& unordered_variable_names,
          bool memory_saving_splitting, SplitRule splitrule, std::string case_weights_file, bool predict_all,
          double sample_fraction, double alpha, double minprop, bool holdout, PredictionType prediction_type,
          uint num_random_splits, int useMPI, int rank, int size, Data* oldData, std::string outputDirectory) {

                  //std::cout << "In Forest initCpp with rank: " << rank << "\n";
              this->verbose_out = verbose_out;
              this->num_trees = num_trees;
              this->data = oldData;
              this->rank = rank;
              this->outputDirectory = outputDirectory;
              std::cout << "initCPPData directory: " << outputDirectory << "\n";
              // Initialize data with memmode
              // switch (memory_mode) {
              // case MEM_DOUBLE:
              //   data = new DataDouble();
              //   break;
              // case MEM_FLOAT:
              //   data = new DataFloat();
              //   break;
              // case MEM_CHAR:
              //   data = new DataChar();
              //   break;
              // }
              //std::cout << "In Forest initCpp, after memory mode\n";
              //std::cout << "Rank: " << rank << '\n';
              // Load data
              //*verbose_out << "Loading input file: " << input_file << "." << std::endl;
              // std::cout << "In Forest initCpp, after verbose out\n";
              // bool rounding_error;
              //
              // std::chrono::time_point<std::chrono::system_clock> startRead, endRead;
      				// startRead = std::chrono::system_clock::now();
              //
              // if (useMPI == 1){
              //   for (int i = 0; i < size; i++){
              //     MPI_Barrier(MPI_COMM_WORLD);
              //     if (i == rank){
              //       rounding_error = data->loadFromFile(input_file);
              //     }
              //   }
              // }
              // else{
              //   rounding_error = data->loadFromFile(input_file);
              // }
              //
              // endRead = std::chrono::system_clock::now();
      				// std::chrono::duration<double> elapsedRead = endRead - startRead;
      				// *verbose_out << "Data read processing time: " << elapsedRead.count() << "s." << std::endl;
              //
              // if (rounding_error) {
              //   *verbose_out << "Warning: Rounding or Integer overflow occurred. Use FLOAT or DOUBLE precision to avoid this."
              //       << std::endl;
              // }
               //*verbose_out << "Finished loading input file\n";
              // std::cout << "In forest initCpp, after rounding error\n";
              // // Set prediction mode
              // bool prediction_mode = false;
              // if (!load_forest_filename.empty()) {
              //   prediction_mode = true;
              // }
               std::cout << "In initCppData, before call to init\n";
               num_samples = oldData->getNumRows();
               std::cout << "number rows: " << num_samples << '\n';
               num_variables = oldData->getNumCols();
               std::cout << "number columns: " << num_variables << '\n';
               num_independent_variables = num_variables - oldData->getNoSplitVariables().size();
               split_select_weights.push_back(std::vector<double>());
              // std::cout << std::flush;
              // Call other init function
              int nonzero_Weights;
              if (!split_select_weights_file.empty()) {
                std::vector<std::vector<double>> split_select_weights;
                split_select_weights.resize(1);
                loadDoubleVectorFromFile(split_select_weights[0], split_select_weights_file);
                if (split_select_weights[0].size() != num_variables - 1) {
                  throw std::runtime_error("Number of split select weights is not equal to number of independent variables.");
                }

                //size_t numColumns = data->getNumCols();
                //std::cout << "After get num cols in initCppData\n" <<std::flush;
                //std::vector<size_t> noSplits = data->getNoSplitVariables();
                //std::cout << "After noSplits in initCppData\n" << std::flush;

                std::cout << "After loadDoubleVector\n" << std::flush;
                setSplitWeightVector(split_select_weights);
                std::cout << "After setSplitWeightVector\n" << std::flush;
                nonzero_Weights = count_if (split_select_weights[0].begin(), split_select_weights[0].end(), isZero);
              }
              std::cout << "After set weight loop\n" << std::flush;
              unsigned long temp = sqrt((double) nonzero_Weights);
              std::cout << "After setting temp\n" << std::flush;
              mtry = std::max((unsigned long) 1, temp);
              std::cout << "After setting mtry\n" << std::flush;


              initData(dependent_variable_name, memory_mode, oldData, mtry, output_prefix, num_trees, seed, num_threads, importance_mode,
                  min_node_size, status_variable_name, prediction_mode, sample_with_replacement, unordered_variable_names,
                  memory_saving_splitting, splitrule, predict_all, sample_fraction, alpha, minprop, holdout, prediction_type,
                  num_random_splits, useMPI);
              std::cout << "In Forest.cpp, directly after call to init\n";
              std::cout << std::flush;

              if (prediction_mode) {
                loadFromFile(load_forest_filename);
              }
              // Set variables to be always considered for splitting
              if (!always_split_variable_names.empty()) {
                setAlwaysSplitVariables(always_split_variable_names);
              }

              // TODO: Read 2d weights for tree-wise split select weights
              // Load split select weights from file
              // if (useMPI == 1){
              //   for (int i = 0; i < size; i++){
              //     MPI_Barrier(MPI_COMM_WORLD);
              //     if (i == rank){
              //       if (!split_select_weights_file.empty()) {
              //         std::vector<std::vector<double>> split_select_weights;
              //         split_select_weights.resize(1);
              //         std::cout << "Right before load vector\n";
              //         std::cout << std::flush;
              //         loadDoubleVectorFromFile(split_select_weights[0], split_select_weights_file);
              //         // std::cout << "split select weights size: " << split_select_weights[0].size() << '\n';
              //         // std::cout << "num_variables: " << num_variables << '\n';
              //         // std::cout << std::flush;
              //         //sleep(10);
              //         if (split_select_weights[0].size() != (num_variables - 1)) {
              //           throw std::runtime_error("Number of split select weights is not equal to number of independent variables.");
              //         }
              //         setSplitWeightVector(split_select_weights);
              //       }
              //     }
              //   }
              // }
              //else{
              // if (!split_select_weights_file.empty()) {
              //   std::vector<std::vector<double>> split_select_weights;
              //   split_select_weights.resize(1);
              //   loadDoubleVectorFromFile(split_select_weights[0], split_select_weights_file);
              //   if (split_select_weights[0].size() != num_variables - 1) {
              //     throw std::runtime_error("Number of split select weights is not equal to number of independent variables.");
              //   }
              //   setSplitWeightVector(split_select_weights);
              //   int nonzero_Weights = count_if (split_select_weights[0].begin(), split_select_weights[0].end(), isZero);
              // }
            //}

            // initData(dependent_variable_name, memory_mode, oldData, mtry, output_prefix, num_trees, seed, num_threads, importance_mode,
            //     min_node_size, status_variable_name, prediction_mode, sample_with_replacement, unordered_variable_names,
            //     memory_saving_splitting, splitrule, predict_all, sample_fraction, alpha, minprop, holdout, prediction_type,
            //     num_random_splits, useMPI);

              // Load case weights from file
              if (!case_weights_file.empty()) {
                loadDoubleVectorFromFile(case_weights, case_weights_file);
                if (case_weights.size() != num_samples - 1) {
                  throw std::runtime_error("Number of case weights is not equal to number of samples.");
                }
              }

              // Sample from non-zero weights in holdout mode
              if (holdout && !case_weights.empty()) {
                size_t nonzero_weights = 0;
                for (auto& weight : case_weights) {
                  if (weight > 0) {
                    ++nonzero_weights;
                  }
                }
                this->sample_fraction = this->sample_fraction * ((double) nonzero_weights / (double) num_samples);
              }

              // Check if all catvars are coded in integers starting at 1
              if (!unordered_variable_names.empty()) {
                std::string error_message = checkUnorderedVariables(oldData, unordered_variable_names);
                if (!error_message.empty()) {
                  throw std::runtime_error(error_message);
                }
              }
            }













void Forest::initR(std::string dependent_variable_name, Data* input_data, uint mtry, uint num_trees,
    std::ostream* verbose_out, uint seed, uint num_threads, ImportanceMode importance_mode, uint min_node_size,
    std::vector<std::vector<double>>& split_select_weights, std::vector<std::string>& always_split_variable_names,
    std::string status_variable_name, bool prediction_mode, bool sample_with_replacement,
    std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
    std::vector<double>& case_weights, bool predict_all, bool keep_inbag, double sample_fraction, double alpha,
    double minprop, bool holdout, PredictionType prediction_type, uint num_random_splits) {

  this->verbose_out = verbose_out;

  // Call other init function
  init(dependent_variable_name, MEM_DOUBLE, input_data, mtry, "", num_trees, seed, num_threads, importance_mode,
      min_node_size, status_variable_name, prediction_mode, sample_with_replacement, unordered_variable_names,
      memory_saving_splitting, splitrule, predict_all, sample_fraction, alpha, minprop, holdout, prediction_type,
      num_random_splits, useMPI);

  // Set variables to be always considered for splitting
  if (!always_split_variable_names.empty()) {
    setAlwaysSplitVariables(always_split_variable_names);
  }

  // Set split select weights
  if (!split_select_weights.empty()) {
    setSplitWeightVector(split_select_weights);
  }

  // Set case weights
  if (!case_weights.empty()) {
    if (case_weights.size() != num_samples) {
      throw std::runtime_error("Number of case weights not equal to number of samples.");
    }
    this->case_weights = case_weights;
  }

  // Keep inbag counts
  this->keep_inbag = keep_inbag;
}

void Forest::init(std::string dependent_variable_name, MemoryMode memory_mode, Data* input_data, uint mtry,
    std::string output_prefix, uint num_trees, uint seed, uint num_threads, ImportanceMode importance_mode,
    uint min_node_size, std::string status_variable_name, bool prediction_mode, bool sample_with_replacement,
    std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
    bool predict_all, double sample_fraction, double alpha, double minprop, bool holdout,
    PredictionType prediction_type, uint num_random_splits, int useMPI) {

  std::cout << "Starting init within Forest.cpp\n";
  // Initialize data with memmode
  this->data = input_data;

  // Initialize random number generator and set seed
  if (seed == 0) {
    std::random_device random_device;
    random_number_generator.seed(random_device());
  } else {
    random_number_generator.seed(seed);
  }

  // Set number of threads
  if (num_threads == DEFAULT_NUM_THREADS) {
#ifdef OLD_WIN_R_BUILD
    this->num_threads = 1;
#else
    this->num_threads = std::thread::hardware_concurrency();
#endif
  } else {
    this->num_threads = num_threads;
  }

  // Set member variables
  this->num_trees = num_trees;
  this->mtry = mtry;
  this->seed = seed;
  this->output_prefix = output_prefix;
  this->importance_mode = importance_mode;
  this->min_node_size = min_node_size;
  this->memory_mode = memory_mode;
  this->prediction_mode = prediction_mode;
  this->sample_with_replacement = sample_with_replacement;
  this->memory_saving_splitting = memory_saving_splitting;
  this->splitrule = splitrule;
  this->predict_all = predict_all;
  this->sample_fraction = sample_fraction;
  this->holdout = holdout;
  this->alpha = alpha;
  this->minprop = minprop;
  this->prediction_type = prediction_type;
  this->num_random_splits = num_random_splits;
  this->useMPI = useMPI;

  // Set number of samples and variables
  num_samples = data->getNumRows();
  std::cout << "number rows: " << num_samples << '\n';
  num_variables = data->getNumCols();
  std::cout << "number columns: " << num_variables << '\n';

  // Convert dependent variable name to ID
  if (!prediction_mode && !dependent_variable_name.empty()) {
    dependent_varID = data->getVariableID(dependent_variable_name);
  }

  // Set unordered factor variables
  if (!prediction_mode) {
    data->setIsOrderedVariable(unordered_variable_names);
  }

  data->addNoSplitVariable(dependent_varID);

  initInternal(status_variable_name);

  num_independent_variables = num_variables - data->getNoSplitVariables().size();
  std::cout << "Num vars in Init: " << num_variables << '\n';
  std::cout << "Num independent vars in Init: " << num_independent_variables << '\n';
  std::cout << std::flush;

  // Init split select weights
   split_select_weights.push_back(std::vector<double>());
   std::cout << "in Forest init, after split_select_weights push\n" << std::flush;

  // Check if mtry is in valid range
  if (this->mtry > num_variables - 1) {
    throw std::runtime_error("mtry can not be larger than number of variables in data.");
  }

  // Check if any observations samples
  if ((size_t) num_samples * sample_fraction < 1) {
    throw std::runtime_error("sample_fraction too small, no observations sampled.");
  }

  // Permute samples for corrected Gini importance
  if (importance_mode == IMP_GINI_CORRECTED) {
    data->permuteSampleIDs(random_number_generator);
  }

}



void Forest::initData(std::string dependent_variable_name, MemoryMode memory_mode, Data* input_data, uint mtry,
    std::string output_prefix, uint num_trees, uint seed, uint num_threads, ImportanceMode importance_mode,
    uint min_node_size, std::string status_variable_name, bool prediction_mode, bool sample_with_replacement,
    std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
    bool predict_all, double sample_fraction, double alpha, double minprop, bool holdout,
    PredictionType prediction_type, uint num_random_splits, int useMPI) {

  std::cout << "Starting initData within Forest.cpp\n";
  // Initialize data with memmode
  //this->data = input_data;

  // Initialize random number generator and set seed
  if (seed == 0) {
    std::random_device random_device;
    random_number_generator.seed(random_device());
  } else {
    random_number_generator.seed(seed);
  }

  // Set number of threads
  if (num_threads == DEFAULT_NUM_THREADS) {
#ifdef OLD_WIN_R_BUILD
    this->num_threads = 1;
#else
    this->num_threads = std::thread::hardware_concurrency();
#endif
  } else {
    this->num_threads = num_threads;
  }

  // Set member variables
  this->num_trees = num_trees;
  this->mtry = mtry;
  this->seed = seed;
  this->output_prefix = output_prefix;
  this->importance_mode = importance_mode;
  this->min_node_size = min_node_size;
  this->memory_mode = memory_mode;
  this->prediction_mode = prediction_mode;
  this->sample_with_replacement = sample_with_replacement;
  this->memory_saving_splitting = memory_saving_splitting;
  this->splitrule = splitrule;
  this->predict_all = predict_all;
  this->sample_fraction = sample_fraction;
  this->holdout = holdout;
  this->alpha = alpha;
  this->minprop = minprop;
  this->prediction_type = prediction_type;
  this->num_random_splits = num_random_splits;
  this->useMPI = useMPI;

  // Set number of samples and variables
  // num_samples = data->getNumRows();
  // std::cout << "number rows: " << num_samples << '\n';
  // num_variables = data->getNumCols();
  // std::cout << "number columns: " << num_variables << '\n';

  // Convert dependent variable name to ID
  if (!prediction_mode && !dependent_variable_name.empty()) {
    dependent_varID = data->getVariableID(dependent_variable_name);
  }
  //std::cout << "Finished setting Dep Var in initData \n";
  //std::cout << std::flush;
  // Set unordered factor variables
  if (!prediction_mode) {
    //data->setIsOrderedVariable(unordered_variable_names);
  }
  //std::cout << "Finished setting ordered variables in initData \n";
  std::cout << std::flush;
  //data->addNoSplitVariable(dependent_varID);

  //initInternal(status_variable_name);
  initInternalData(status_variable_name);
  std::cout << "Finished Init Internal Data\n";
  std::cout << std::flush;

  //num_independent_variables = num_variables - data->getNoSplitVariables().size();
  //std::cout << "Num vars in Init: " << num_variables << '\n';
  //std::cout << "Num independent vars in Init: " << num_independent_variables << '\n';
  //std::cout << std::flush;

  // Init split select weights
  //split_select_weights.push_back(std::vector<double>());
  //std::cout << "In initData, before checking mtry\n" << std::flush;
  // Check if mtry is in valid range
  if (this->mtry > num_variables - 1) {
    throw std::runtime_error("mtry can not be larger than number of variables in data.");
  }

  // Check if any observations samples
  //std::cout << "In initData, before checking num_samples\n" << std::flush;
  if ((size_t) num_samples * sample_fraction < 1) {
    throw std::runtime_error("sample_fraction too small, no observations sampled.");
  }

  // Permute samples for corrected Gini importance
  //std::cout << "In initData, before importancce_mode check\n" << std::flush;
  if (importance_mode == IMP_GINI_CORRECTED) {
    data->permuteSampleIDs(random_number_generator);
  }

}



void Forest::run(bool verbose) {
  std::cout << "Starting run within Forest.cpp\n";
  if (prediction_mode) {
    if (verbose) {
      *verbose_out << "Predicting .." << std::endl;
    }
    predict();
  } else {
    if (verbose) {
      *verbose_out << "Growing trees .." << std::endl;
    }

    std::chrono::time_point<std::chrono::system_clock> startGrow, endGrow;
    startGrow = std::chrono::system_clock::now();

    grow();

    endGrow = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsedGrow = endGrow - startGrow;
    *verbose_out << "Forest grow processing time: " << elapsedGrow.count() << "s." << std::endl;

    // if (verbose) {
    //   *verbose_out << "Computing prediction error .." << std::endl;
    // }
    //computePredictionError();

    if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW || importance_mode == IMP_PERM_RAW) {
      if (verbose) {
        *verbose_out << "Computing permutation variable importance .." << std::endl;
      }
      computePermutationImportance();
    }
  }
}

// #nocov start
void Forest::writeOutput() {

  *verbose_out << std::endl;
  writeOutputInternal();
  *verbose_out << "Dependent variable name:           " << data->getVariableNames()[dependent_varID] << std::endl;
  *verbose_out << "Dependent variable ID:             " << dependent_varID << std::endl;
  *verbose_out << "Number of trees:                   " << num_trees << std::endl;
  *verbose_out << "Sample size:                       " << num_samples << std::endl;
  *verbose_out << "Number of independent variables:   " << num_independent_variables << std::endl;
  *verbose_out << "Mtry:                              " << mtry << std::endl;
  *verbose_out << "Target node size:                  " << min_node_size << std::endl;
  *verbose_out << "Variable importance mode:          " << importance_mode << std::endl;
  *verbose_out << "Memory mode:                       " << memory_mode << std::endl;
  *verbose_out << "Seed:                              " << seed << std::endl;
  *verbose_out << "Number of threads:                 " << num_threads << std::endl;
  *verbose_out << std::endl;

  if (prediction_mode) {
    writePredictionFile(rank);
  }
  //else {
//     *verbose_out << "Overall OOB prediction error:      " << overall_prediction_error << std::endl;
//     *verbose_out << std::endl;
//
//     if (!split_select_weights.empty() & !split_select_weights[0].empty()) {
//       *verbose_out
//           << "Warning: Split select weights used. Variable importance measures are only comparable for variables with equal weights."
//           << std::endl;
//     }
//
//     if (importance_mode != IMP_NONE) {
//       writeImportanceFile();
//     }
//
//     writeConfusionFile();
//   }
 }

void Forest::writeOutputNewForest() {
    std::cout << "In writeoutput New Forest\n" << std::flush;
    if (prediction_mode) {
      //writePredictionFile(rank);
    } else {
      //*verbose_out << "Overall OOB prediction error:      " << overall_prediction_error << std::endl;
      //*verbose_out << std::endl;
      std::cout << "In writeOutputNewForest\n" << std::flush;
      //if (!split_select_weights.empty() & !split_select_weights[0].empty()) {
        //*verbose_out
            //<< "Warning: Split select weights used. Variable importance measures are only comparable for variables with equal weights."
            //<< std::endl;
      //}
      std::cout << "In writeOutputNewForest, right before write Importance\n" << std::flush;
      if (importance_mode != IMP_NONE) {
        writeImportanceFile();
      }
      std::cout << "In writeOutputNewForest, right before write Confusion\n" << std::flush;
      //writeConfusionFile();
      std::cout << "In writeOutputNewForest, after write confusion\n" << std::flush;
    }
}

void Forest::writeImportanceFile() {
  std::cout << "In writeImportance, called from writeOutputNewForest \n" << std::flush;
  // Open importance file for writing
  std::string directory = outputDirectory;
  std::string filename = directory + "/" + output_prefix + ".importance";
  std::cout << "Filename: " << filename << "\n";
  std::string rankStr = std::to_string(rank);
  std::ofstream importance_file;
  std::cout << "After creating ofstream in writeImportance\n" << std::flush;
  importance_file.open(filename, std::ios::out);
  if (!importance_file.good()) {
    throw std::runtime_error("Could not write to importance file: " + filename + ".");
  }

  // Write importance to file
  std::cout << "Right before write in writeImportance\n" << std::flush;
  for (size_t i = 0; i < variable_importance.size(); ++i) {
    size_t varID = i;
    //std::cout << "Right before data get splitVars\n" << std::flush;
    //std::cout << "NoSplitVar Size: " << data->getNoSplitVariables().size() << '\n' << std::flush;
    //std::cout << "NoSplitVar size: "  << data->no_split_variables.size();
    std::vector<size_t> noSplitVars = data->getNoSplitVariables();
    // for (auto& skip : data->getNoSplitVariables()) {
    for (auto& skip : noSplitVars) {
      //std::cout << "After get noSplitVars\n" << std::flush;
      if (varID >= skip) {
        ++varID;
      }
    }

    std::string variable_name = data->getVariableNames()[varID];
    //std::cout << variable_name << ": " << variable_importance[i] << " Rank: " << rankStr << '\n';
    std::cout << std::flush;
    //importance_file << variable_name << ": " << variable_importance[i] << std::endl;
    importance_file << variable_name << ": " << variable_importance[i] << '\n';
  }

  importance_file.close();
  std::cout << "Right before verbose in writeImportance\n" << std::flush;
  *verbose_out << "Saved variable importance to file " << filename << "." << std::endl;
}

void Forest::writeSplitWeightsFile(std::string outputPrefix){
  std::string directory = outputDirectory;
  std::string filename = directory + "/" + outputPrefix + ".splitWeights";
  std::ofstream split_file;
  split_file.open(filename, std::ios::out);
  if (!split_file.good()){
    throw std::runtime_error("Cout not write to split weight file: " + filename + ".");
  }


  //Calculate scaled splitWeights
  double total = 0;
  for (size_t i = 0; i < variable_importance.size(); i++) {
    if (variable_importance[i] >= 0){
      total += variable_importance[i];
    }
    //std::cout << "Var Imp[" << i << "]: " <<  variable_importance[i] << '\n';
    //std::cout << "Total at i: " << total << '\n';
    std::cout << std::flush;
  }
  std::cout << "Total: " << total << '\n';
  std::cout << "varImp size: " << variable_importance.size() << '\n';

  std::vector<double> scaledVarImp;
  for (size_t i = 0; i < variable_importance.size(); i++) {
    double scaledVar = variable_importance[i]/total;
    if (scaledVar < 0){
      scaledVar = 0;
      //std::cout << "In write splitweight, scaledVar: " << scaledVar << " VarID: " << i << '\n' << std::flush;
    }
    //std::cout << "I: " << i << " scaledVar: " << scaledVar << '\n';
    std::cout << std::flush;
    scaledVarImp.push_back(scaledVar);
  }
  std::cout << "After push back" << '\n';
  std::cout << std::flush;
  //Write split weight file
  for (size_t i = 0; i < variable_importance.size(); i++) {
    size_t varID = i;
    // std::cout << "VarID: " << i << '\n';
    // std::cout << std::flush;
    // for (auto& skip : data->getNoSplitVariables()) {
    //   if (varID >= skip) {
    //     ++varID;
    //   }
    // }
    //std::cout << "After splitVar check. I: " << i << " VarID: " << varID << '\n';
    std::cout << std::flush;
    //std::string variable_name = data->getVariableNames()[varID];
    if (i < (variable_importance.size()-1)){
      split_file << scaledVarImp[i] << " ";
    }
    else {
      split_file << scaledVarImp[i];
    }
  }
  std::cout << "After write to file\n";
  std::cout << std::flush;
  split_file << std::endl;

  std::cout << "After adding endl to file\n";
  std::cout << std::flush;

  split_file.close();
  std::cout << "After file close\n";
  std::cout << std::flush;
  //*verbose_out << "Saved variable split weights to file " << filename << "." << std::endl;
  std::cout << "After verbose write\n";
  std::cout << std::flush;

}

void Forest::saveToFile(int rank) {

  // Open file for writing
  std::string directory = outputDirectory;
  std::string filename = directory + "/" + output_prefix + ".forest";
  std::ofstream outfile;
  outfile.open(filename, std::ios::binary);
  //outfile.open(filename);
  if (!outfile.good()) {
    throw std::runtime_error("Could not write to output file: " + filename + ".");
  }

  // Write dependent_varID
  outfile.write((char*) &dependent_varID, sizeof(dependent_varID));

  // Write num_trees
  outfile.write((char*) &num_trees, sizeof(num_trees));

  // Write is_ordered_variable
  saveVector1D(data->getIsOrderedVariable(), outfile);

  saveToFileInternal(outfile);

  // Write tree data for each tree
  for (auto& tree : trees) {
    tree->appendToFile(outfile);
  }

  // Close file
  outfile.close();
  *verbose_out << "Saved forest to file " << filename << "." << std::endl;
}
// #nocov end

void Forest::grow() {
  std::cout << "Starting grow() in Forest.cpp\n";
  // Create thread ranges
  std::cout << "Num Trees: " << num_trees << '\n';
  equalSplit(thread_ranges, 0, num_trees - 1, num_threads);

  // Call special grow functions of subclasses. There trees must be created.
  *verbose_out << "Growing trees internal .." << std::endl;
  growInternal();

  // Init trees, create a seed for each tree, based on main seed
  std::uniform_int_distribution<uint> udist;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();


  for (size_t i = 0; i < num_trees; ++i) {
    uint tree_seed;
    if (seed == 0) {
      tree_seed = udist(random_number_generator);
    } else {
      tree_seed = (i + 1) * seed;
    }

    // Get split select weights for tree
    std::vector<double>* tree_split_select_weights;
    if (split_select_weights.size() > 1) {
      tree_split_select_weights = &split_select_weights[i];
    } else {
      tree_split_select_weights = &split_select_weights[0];
    }

    trees[i]->init(data, mtry, dependent_varID, num_samples, tree_seed, &deterministic_varIDs, &split_select_varIDs,
        tree_split_select_weights, importance_mode, min_node_size, sample_with_replacement, memory_saving_splitting,
        splitrule, &case_weights, keep_inbag, sample_fraction, alpha, minprop, holdout, num_random_splits);
  }

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  //time_t tt = std::chrono::system_clock::to_time_t(end);
  *verbose_out << "Trees Init processing time, in Forest.cpp " << elapsed.count() << "s." << std::endl;

  std::cout << "After init each tree in Forest Grow\n" << std::flush;

// Init variable importance
  variable_importance.resize(num_independent_variables, 0);

  std::cout << "After varImportance resize in Forest Grow\n" << std::flush;
// Grow trees in multiple threads
#ifdef OLD_WIN_R_BUILD
  progress = 0;
  clock_t start_time = clock();
  clock_t lap_time = clock();

  for (size_t i = 0; i < num_trees; ++i) {
    trees[i]->grow(&variable_importance);
    progress++;
    showProgress("Growing trees..", start_time, lap_time);
  }

#else
  progress = 0;
#ifdef R_BUILD
  aborted = false;
  aborted_threads = 0;
#endif

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  //std::cout << "After thread reserve in Forest Grow\n" << std::flush;
// Initailize importance per thread
  std::vector<std::vector<double>> variable_importance_threads(num_threads);
  std::chrono::time_point<std::chrono::system_clock> startTree, endTree;
  startTree = std::chrono::system_clock::now();
  //std::cout << "before importance loop in Forest Grow\n" << std::flush;
  for (uint i = 0; i < num_threads; ++i) {
    //std::cout << "In thread loop\n" << std::flush;
    if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
      variable_importance_threads[i].resize(num_independent_variables, 0);
      //std::cout << "In thread loop, after setting var imp thread\n" << std::flush;
    }
    threads.push_back(std::thread(&Forest::growTreesInThread, this, i, &(variable_importance_threads[i])));
    //std::cout << "In thread loop, after push back thread\n" << std::flush;
  }
  showProgress("Growing trees..");
  for (auto &thread : threads) {
    thread.join();
  }

  endTree = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsedTree = endTree - startTree;
  //tt = std::chrono::system_clock::to_time_t(endTree);
  *verbose_out << "Trees grow processing time, in Forest.cpp " << elapsedTree.count() << "s." << std::endl;

#ifdef R_BUILD
  if (aborted_threads > 0) {
    throw std::runtime_error("User interrupt.");
  }
#endif

  // Sum thread importances
  if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
    variable_importance.resize(num_independent_variables, 0);
    for (size_t i = 0; i < num_independent_variables; ++i) {
      for (uint j = 0; j < num_threads; ++j) {
        variable_importance[i] += variable_importance_threads[j][i];
        if (variable_importance[i] < 0){
          //std::cout << "In Forest Grow, variable_importance[i]: " << variable_importance[i] << " I: " << i << '\n' << std::flush;
        }
      }
    }
    variable_importance_threads.clear();
  }

#endif

// Divide importance by number of trees
  if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
    for (auto& v : variable_importance) {
      v /= num_trees;
    }
  }
}

void Forest::predict() {

// Predict trees in multiple threads and join the threads with the main thread
#ifdef OLD_WIN_R_BUILD
  progress = 0;
  clock_t start_time = clock();
  clock_t lap_time = clock();
  for (size_t i = 0; i < num_trees; ++i) {
    trees[i]->predict(data, false);
    progress++;
    showProgress("Predicting..", start_time, lap_time);
  }
#else
  progress = 0;
#ifdef R_BUILD
  aborted = false;
  aborted_threads = 0;
#endif

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (uint i = 0; i < num_threads; ++i) {
    threads.push_back(std::thread(&Forest::predictTreesInThread, this, i, data, false));
  }
  showProgress("Predicting..");
  for (auto &thread : threads) {
    thread.join();
  }

#ifdef R_BUILD
  if (aborted_threads > 0) {
    throw std::runtime_error("User interrupt.");
  }
#endif
#endif

// Call special functions for subclasses
  predictInternal();
}

void Forest::computePredictionError() {

// Predict trees in multiple threads
#ifdef OLD_WIN_R_BUILD
  progress = 0;
  clock_t start_time = clock();
  clock_t lap_time = clock();
  for (size_t i = 0; i < num_trees; ++i) {
    trees[i]->predict(data, true);
    progress++;
    showProgress("Predicting..", start_time, lap_time);
  }
#else
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (uint i = 0; i < num_threads; ++i) {
    threads.push_back(std::thread(&Forest::predictTreesInThread, this, i, data, true));
  }
  showProgress("Computing prediction error..");
  for (auto &thread : threads) {
    thread.join();
  }

#ifdef R_BUILD
  if (aborted_threads > 0) {
    throw std::runtime_error("User interrupt.");
  }
#endif
#endif

  // Call special function for subclasses
  computePredictionErrorInternal();
}

void Forest::computePermutationImportance() {

// Compute tree permutation importance in multiple threads
#ifdef OLD_WIN_R_BUILD
  progress = 0;
  clock_t start_time = clock();
  clock_t lap_time = clock();

// Initailize importance and variance
  variable_importance.resize(num_independent_variables, 0);
  std::vector<double> variance;
  if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW) {
    variance.resize(num_independent_variables, 0);
  }

// Compute importance
  for (size_t i = 0; i < num_trees; ++i) {
    trees[i]->computePermutationImportance(&variable_importance, &variance);
    progress++;
    showProgress("Computing permutation importance..", start_time, lap_time);
  }
#else
  progress = 0;
#ifdef R_BUILD
  aborted = false;
  aborted_threads = 0;
#endif

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

// Initailize importance and variance
  std::vector<std::vector<double>> variable_importance_threads(num_threads);
  std::vector<std::vector<double>> variance_threads(num_threads);

// Compute importance
  for (uint i = 0; i < num_threads; ++i) {
    variable_importance_threads[i].resize(num_independent_variables, 0);
    if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW) {
      variance_threads[i].resize(num_independent_variables, 0);
    }
    threads.push_back(
        std::thread(&Forest::computeTreePermutationImportanceInThread, this, i, &(variable_importance_threads[i]),
            &(variance_threads[i])));
  }
  showProgress("Computing permutation importance..");
  for (auto &thread : threads) {
    thread.join();
  }

#ifdef R_BUILD
  if (aborted_threads > 0) {
    throw std::runtime_error("User interrupt.");
  }
#endif

// Sum thread importances
  variable_importance.resize(num_independent_variables, 0);
  for (size_t i = 0; i < num_independent_variables; ++i) {
    for (uint j = 0; j < num_threads; ++j) {
      variable_importance[i] += variable_importance_threads[j][i];
    }
  }
  variable_importance_threads.clear();

// Sum thread variances
  std::vector<double> variance(num_independent_variables, 0);
  if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW) {
    for (size_t i = 0; i < num_independent_variables; ++i) {
      for (uint j = 0; j < num_threads; ++j) {
        variance[i] += variance_threads[j][i];
      }
    }
    variance_threads.clear();
  }
#endif

  for (size_t i = 0; i < variable_importance.size(); ++i) {
    variable_importance[i] /= num_trees;

    // Normalize by variance for scaled permutation importance
    if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW) {
      if (variance[i] != 0) {
        variance[i] = variance[i] / num_trees - variable_importance[i] * variable_importance[i];
        variable_importance[i] /= sqrt(variance[i] / num_trees);
      }
    }
  }
}

#ifndef OLD_WIN_R_BUILD
void Forest::growTreesInThread(uint thread_idx, std::vector<double>* variable_importance) {
  if (thread_ranges.size() > thread_idx + 1) {
    for (size_t i = thread_ranges[thread_idx]; i < thread_ranges[thread_idx + 1]; ++i) {
      trees[i]->grow(variable_importance);

      // Check for user interrupt
#ifdef R_BUILD
      if (aborted) {
        std::unique_lock<std::mutex> lock(mutex);
        ++aborted_threads;
        condition_variable.notify_one();
        return;
      }
#endif

      // Increase progress by 1 tree
      std::unique_lock<std::mutex> lock(mutex);
      ++progress;
      condition_variable.notify_one();
    }
  }
}

void Forest::predictTreesInThread(uint thread_idx, const Data* prediction_data, bool oob_prediction) {
  if (thread_ranges.size() > thread_idx + 1) {
    for (size_t i = thread_ranges[thread_idx]; i < thread_ranges[thread_idx + 1]; ++i) {
      trees[i]->predict(prediction_data, oob_prediction);

      // Check for user interrupt
#ifdef R_BUILD
      if (aborted) {
        std::unique_lock<std::mutex> lock(mutex);
        ++aborted_threads;
        condition_variable.notify_one();
        return;
      }
#endif

      // Increase progress by 1 tree
      std::unique_lock<std::mutex> lock(mutex);
      ++progress;
      condition_variable.notify_one();
    }
  }
}

void Forest::computeTreePermutationImportanceInThread(uint thread_idx, std::vector<double>* importance,
    std::vector<double>* variance) {
  if (thread_ranges.size() > thread_idx + 1) {
    for (size_t i = thread_ranges[thread_idx]; i < thread_ranges[thread_idx + 1]; ++i) {
      trees[i]->computePermutationImportance(importance, variance);

      // Check for user interrupt
#ifdef R_BUILD
      if (aborted) {
        std::unique_lock<std::mutex> lock(mutex);
        ++aborted_threads;
        condition_variable.notify_one();
        return;
      }
#endif

      // Increase progress by 1 tree
      std::unique_lock<std::mutex> lock(mutex);
      ++progress;
      condition_variable.notify_one();
    }
  }
}
#endif

// #nocov start
void Forest::loadFromFile(std::string filename) {
  *verbose_out << "Loading forest from file " << filename << "." << std::endl;

// Open file for reading
  std::ifstream infile;
  infile.open(filename, std::ios::binary);
  if (!infile.good()) {
    throw std::runtime_error("Could not read from input file: " + filename + ".");
  }

// Read dependent_varID and num_trees
  infile.read((char*) &dependent_varID, sizeof(dependent_varID));
  infile.read((char*) &num_trees, sizeof(num_trees));

// Read is_ordered_variable
  readVector1D(data->getIsOrderedVariable(), infile);

// Read tree data. This is different for tree types -> virtual function
  loadFromFileInternal(infile);

  infile.close();

// Create thread ranges
  equalSplit(thread_ranges, 0, num_trees - 1, num_threads);
}
// #nocov end

void Forest::setSplitWeightVector(std::vector<std::vector<double>>& split_select_weights) {
  //std::cout << "In setPlitWeightVector\n" << std::flush;
// Size should be 1 x num_independent_variables or num_trees x num_independent_variables
  if (split_select_weights.size() != 1 && split_select_weights.size() != num_trees) {
    throw std::runtime_error("Size of split select weights not equal to 1 or number of trees.");
  }
  //std::cout << "In setPlitWeightVector, after throw\n" << std::flush;
// Reserve space
  if (split_select_weights.size() == 1) {
    this->split_select_weights[0].resize(num_independent_variables);
    //std::cout << "In setPlitWeightVector, after if\n" << std::flush;
  }else {
    //std::cout << "In setPlitWeightVector, after else\n" << std::flush;
    this->split_select_weights.clear();
    //std::cout << "In setPlitWeightVector, after clear\n" << std::flush;
    this->split_select_weights.resize(num_trees, std::vector<double>(num_independent_variables));
  }
  //std::cout << "In setSplitWeightVector, after splitWeight resize\n" << std::flush;
  this->split_select_varIDs.resize(num_independent_variables);
  deterministic_varIDs.reserve(num_independent_variables);
  //std::cout << "In setPlitWeightVector, after deterministic var reserve\n" << std::flush;

// Split up in deterministic and weighted variables, ignore zero weights
  for (size_t i = 0; i < split_select_weights.size(); ++i) {
    //std::cout << "split select weights size: " << split_select_weights[i].size() << '\n';
    //std::cout << "num independent vars: " << num_independent_variables << '\n';
    //std::cout << std::flush;
    // Size should be 1 x num_independent_variables or num_trees x num_independent_variables
    if (split_select_weights[i].size() != num_independent_variables) {
      throw std::runtime_error("Number of split select weights not equal to number of independent variables.");
    }
    //std::cout << "in setSplitWeightVector, after throw\n" << std::flush;
    for (size_t j = 0; j < split_select_weights[i].size(); ++j) {
      double weight = split_select_weights[i][j];
      //std::cout << "In setSplitWeightVector, after setting weight\n" << std::flush;
      if (i == 0) {
        size_t varID = j;
        for (auto& skip : data->getNoSplitVariables()) {
          //std::cout << "In setSplitWeightVector, after getNoSplitVars\n" << std::flush;
          if (varID >= skip) {
            ++varID;
          }
        }
        //std::cout << "In setSplitWeightVector, before checking weight equals one\n" << std::flush;
        if (weight == 1) {
          deterministic_varIDs.push_back(varID);
        } else if (weight < 1 && weight > 0) {
          this->split_select_varIDs[j] = varID;
          this->split_select_weights[i][j] = weight;
        } else if (weight < 0 || weight > 1) {
          std::cout << "Column: " << j << " Value: " << weight << '\n' <<std::flush;
          throw std::runtime_error("One or more split select weights not in range [0,1].");
        }

      } else {
        //std::cout << "In setSplitWeightVector, inside else\n" << std::flush;
        if (weight < 1 && weight > 0) {
          this->split_select_weights[i][j] = weight;
        } else if (weight < 0 || weight > 1) {
          throw std::runtime_error("One or more split select weights not in range [0,1].");
        }
      }
    }
    //std::cout << "In setSplitWeightVector, after for loop\n" << std::flush;
  }

  if (deterministic_varIDs.size() > this->mtry) {
    throw std::runtime_error("Number of ones in split select weights cannot be larger than mtry.");
  }
  if (deterministic_varIDs.size() + split_select_varIDs.size() < mtry) {
    throw std::runtime_error("Too many zeros in split select weights. Need at least mtry variables to split at.");
  }
}

void Forest::setAlwaysSplitVariables(std::vector<std::string>& always_split_variable_names) {

  deterministic_varIDs.reserve(num_independent_variables);

  for (auto& variable_name : always_split_variable_names) {
    size_t varID = data->getVariableID(variable_name);
    deterministic_varIDs.push_back(varID);
  }

  if (deterministic_varIDs.size() + this->mtry > num_independent_variables) {
    throw std::runtime_error(
        "Number of variables to be always considered for splitting plus mtry cannot be larger than number of independent variables.");
  }
}

#ifdef OLD_WIN_R_BUILD
void Forest::showProgress(std::string operation, clock_t start_time, clock_t& lap_time) {

// Check for user interrupt
  if (checkInterrupt()) {
    throw std::runtime_error("User interrupt.");
  }

  double elapsed_time = (clock() - lap_time) / CLOCKS_PER_SEC;
  if (elapsed_time > STATUS_INTERVAL) {
    double relative_progress = (double) progress / (double) num_trees;
    double time_from_start = (clock() - start_time) / CLOCKS_PER_SEC;
    uint remaining_time = (1 / relative_progress - 1) * time_from_start;
    *verbose_out << operation << " Progress: " << round(100 * relative_progress) << "%. Estimated remaining time: "
    << beautifyTime(remaining_time) << "." << std::endl;
    lap_time = clock();
  }
}
#else
void Forest::showProgress(std::string operation) {
  using std::chrono::steady_clock;
  using std::chrono::duration_cast;
  using std::chrono::seconds;

  steady_clock::time_point start_time = steady_clock::now();
  steady_clock::time_point last_time = steady_clock::now();
  std::unique_lock<std::mutex> lock(mutex);

// Wait for message from threads and show output if enough time elapsed
  while (progress < num_trees) {
    condition_variable.wait(lock);
    seconds elapsed_time = duration_cast<seconds>(steady_clock::now() - last_time);

    // Check for user interrupt
#ifdef R_BUILD
    if (!aborted && checkInterrupt()) {
      aborted = true;
    }
    if (aborted && aborted_threads >= num_threads) {
      return;
    }
#endif

    if (progress > 0 && elapsed_time.count() > STATUS_INTERVAL) {
      double relative_progress = (double) progress / (double) num_trees;
      seconds time_from_start = duration_cast<seconds>(steady_clock::now() - start_time);
      uint remaining_time = (1 / relative_progress - 1) * time_from_start.count();
      *verbose_out << operation << " Progress: " << round(100 * relative_progress) << "%. Estimated remaining time: "
          << beautifyTime(remaining_time) << "." << std::endl;
      last_time = steady_clock::now();
    }
  }
}
#endif







//void Forest::sendTreesMPI(std::vector<double> &fData, std::vector<int> &tLocations, int &nTrees, std::vector<int> &nNodes, std::vector<double> &predicts, std::vector<double> var_imp){
void Forest::sendTreesMPI(std::vector<double> &fData, std::vector<int> &tLocations, int &nTrees, std::vector<int> &nNodes, std::vector<double> &var_imp){
  std::cout << "Starting sendTreesMPI() in Forest.cpp\n";
  //If using MPI, send Forest to rank 0
  //Tree data must be sent as an array/vector
  std::vector<double> forestData;
  //Resize to standard size for sendArray
  //forestData.resize();

  //Save the number of trees in the forest
  int numTrees = getNumTrees();
  nTrees = numTrees;
  //Save the index location of the start of each tree, split_varIDs vector, and split_values in forestData
  std::vector<int> treeLocations;
  int sizeNum = numTrees * 3;
  //treeLocations.resize(sizeNum);

  std::vector<int> numNodes;
  int treeCount = 0;

  std::cout << "In sendTreesMPI(), starting iteration through trees\n";
  for (auto& tree : trees){
    //std::cout << "In sendTreesMPI(), in Tree loop\n";
    std::vector<std::vector<size_t> > child_nodeIDs = tree->getChildNodeIDs();


    //std::cout << "In sendTreesMPI(), after grabbing child nodes\n";
    std::vector<size_t> split_varIDs = tree->getSplitVarIDs();
    //std::cout << "In sendTreesMPI(), after grabbing split var IDs\n";
    std::vector<double> split_values = tree->getSplitValues();
    //std::cout << "In sendTreesMPI(), after grabbing split values\n";
    //std::vector<double> treeVec;
    //std::vector<std::vector<int> >::iterator col;
    //std::vector<int>::iterator row;
    //numNodes[treeCount] = child_nodeIDs.size();
    numNodes.push_back(child_nodeIDs[0].size());
    //std::cout << "In sendTreesMPI(), after setting numNodes size\n";
    //std::cout << "Child node ids in forest \n";
    // for (int i = 0; i < child_nodeIDs.size(); i++){
    //   for (int j = 0; j < child_nodeIDs[0].size(); j++){
    //     std::cout << child_nodeIDs[i][j] << ' ';
    //   }
    //   std::cout << '\n';
    // }
    //Before adding the next tree to forestData, add the starting location to treeLocations
    treeLocations.push_back(forestData.size());
    //std::cout << " In sendTreesMPI(), before childNode iteration\n";
    int nodeCount = 0;
    //Iterate through the child_nodeIDs matrix, add to forestData in pattern L1,R1,L2,R2....Ln,Rn
    // for (std::vector<std::vector<size_t> >::const_iterator col = child_nodeIDs.begin(); col != child_nodeIDs.end(); ++col){
    //   for (std::vector<size_t>::const_iterator row = col->begin(); row != col->end(); ++row){
    //     forestData.push_back(*row);
    //     nodeCount++;
    //   }
    // }
    for (int i = 0; i < child_nodeIDs.size(); i++){
      for (int j = 0; j < child_nodeIDs[0].size(); j++){
        forestData.push_back(child_nodeIDs[i][j]);
        nodeCount++;
      }
    }
    //std::cout << "nodeCount: " <<  nodeCount << "childNodeIds[0].size: " << child_nodeIDs[0].size() << '\n';
    //add start location for split_varIDs to treeLocations
    treeLocations.push_back(forestData.size());
    //std::cout << " In sendTreesMPI(), before splitVarIDs iteration\n";
    //Iterate through split_varIDs, add to forestData
    for (std::vector<size_t>::const_iterator row = split_varIDs.begin(); row != split_varIDs.end(); row++){
      forestData.push_back(*row);
    }

    //add start location for split_values to treeLocations
    treeLocations.push_back(forestData.size());
    //std::cout << " In sendTreesMPI(), before splitValues iteration\n";
    for (std::vector<double>::const_iterator row = split_values.begin(); row != split_values.end(); row++){
      forestData.push_back(*row);
    }
    treeCount++;

  }
  //std::cout << "In sendTreesMPI, after iteration through trees\n";
  var_imp = getVariableImportance();

  // std::cout << "Variable Importance in Forest.cpp: \n";
  // for (int i = 0; i < var_imp.size(); i++){
  //   std::cout << var_imp[i] << '\n';
  // }
  std::cout << std::flush;
  //std::vector<double> preds;
  //sendPredictionsInternal(preds);
  //predicts = preds;
  fData = forestData;
  //std::cout << "In sendTreesMPI, printing forest\n";
  // for (int i = 0; i < forestData.size(); i++) {
  //   std::cout << "I: " << i << " " << forestData[i] << " ";
  //   if (i % 10 == 0){
  //     std::cout << '\n';
  //   }
  // }
  std::cout << '\n';
  tLocations = treeLocations;
  nNodes = numNodes;
  //std::cout << "sendTreesMPI, forestData size: " << fData.size() << '\n';
  //std::cout << "sendTreesMPI, treeLocations size: " << tLocations.size() << '\n';
  //std::cout << "sendTreesMPI, nNodes size: " << nNodes.size() << '\n';
  //std::cout << "sendTreesMPI, var Importance size: " << var_imp.size() << '\n';
  //if Classification, send std::vector<double> class_values;
  //and std::vector<uint> response_classIDs;
}

void Forest::sendTrees(std::vector<double> &classValues, std::vector<uint> &responseClassIDs){
  std::vector<double> classVals;
  std::vector<uint> respClassIDs;

  sendTreesInternal(classVals, responseClassIDs);
  classValues = classVals;
  responseClassIDs = respClassIDs;
}
