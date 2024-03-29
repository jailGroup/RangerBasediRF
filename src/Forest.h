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

#ifndef FOREST_H_
#define FOREST_H_

#include <vector>
#include <iostream>
#include <random>
#include <ctime>
#ifndef OLD_WIN_R_BUILD
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#endif

#include "globals.h"
#include "Tree.h"
#include "Data.h"

class Forest {
public:
  Forest();
  virtual ~Forest();

  std::vector<Tree*> trees;
  //std::vector<Tree*> trees;
  Data* data;

  // Init from c++ main or Rcpp from R
  void initCpp(std::string dependent_variable_name, MemoryMode memory_mode, std::string input_file, std::string y_file, uint mtry, MtryType mtryType,
      std::string output_prefix, uint num_trees, std::ostream* verbose_out, uint seed, uint num_threads,
      std::string load_forest_filename, ImportanceMode importance_mode, uint min_node_size,
      std::string split_select_weights_file, std::vector<std::string>& always_split_variable_names,
      std::string status_variable_name, bool sample_with_replacement,
      std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
      std::string case_weights_file, bool predict_all, double sample_fraction, double alpha, double minprop,
      bool holdout, PredictionType prediction_type, uint num_random_splits, int useMPI, int rank, int size, std::string outputDirectory, std::ostream* vermbose_time);
  void initCppData(std::string dependent_variable_name, MemoryMode memory_mode, std::string input_file, std::string y_file, uint mtry, MtryType mtryType,
      std::string output_prefix, uint num_trees, std::ostream* verbose_out, uint seed, uint num_threads,
      std::string load_forest_filename, ImportanceMode importance_mode, uint min_node_size,
      std::string split_select_weights_file, std::vector<std::string>& always_split_variable_names,
      std::string status_variable_name, bool sample_with_replacement,
      std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
      std::string case_weights_file, bool predict_all, double sample_fraction, double alpha, double minprop,
      bool holdout, PredictionType prediction_type, uint num_random_splits, int useMPI, int rank, int size, Data* oldData, std::string outputDirectory);
  void initR(std::string dependent_variable_name, Data* input_data, uint mtry, uint num_trees,
      std::ostream* verbose_out, uint seed, uint num_threads, ImportanceMode importance_mode, uint min_node_size,
      std::vector<std::vector<double>>& split_select_weights, std::vector<std::string>& always_split_variable_names,
      std::string status_variable_name, bool prediction_mode, bool sample_with_replacement,
      std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
      std::vector<double>& case_weights, bool predict_all, bool keep_inbag, double sample_fraction, double alpha,
      double minprop, bool holdout, PredictionType prediction_type, uint num_random_splits);
  void init(std::string dependent_variable_name, MemoryMode memory_mode, Data* input_data, uint mtry, MtryType mtryType,
      std::string output_prefix, uint num_trees, uint seed, uint num_threads, ImportanceMode importance_mode,
      uint min_node_size, std::string status_variable_name, bool prediction_mode, bool sample_with_replacement,
      std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
      bool predict_all, double sample_fraction, double alpha, double minprop, bool holdout,
      PredictionType prediction_type, uint num_random_splits, int useMPI);
void initData(std::string dependent_variable_name, MemoryMode memory_mode, Data* input_data, uint mtry, MtryType mtryType,
      std::string output_prefix, uint num_trees, uint seed, uint num_threads, ImportanceMode importance_mode,
      uint min_node_size, std::string status_variable_name, bool prediction_mode, bool sample_with_replacement,
      std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
      bool predict_all, double sample_fraction, double alpha, double minprop, bool holdout,
      PredictionType prediction_type, uint num_random_splits, int useMPI);
  virtual void initInternal(std::string status_variable_name) = 0;
  virtual void initInternalData(std::string status_variable_name) = 0;

  // Grow or predict
  void run(bool verbose);

  // Write results to output files
  void writeOutput();
  void writeOutputNewForest(int loop);
  virtual void writeOutputInternal() = 0;
  virtual void writeConfusionFile() = 0;
  virtual void writePredictionFile(int rank) = 0;
  void writeImportanceFile(int loop);
  void writeSplitWeightsFile(std::string outputPrefix);

  // Save forest to file
  void saveToFile(int rank);
  virtual void saveToFileInternal(std::ofstream& outfile) = 0;

  std::vector<std::vector<std::vector<size_t>>>getChildNodeIDs() {
    std::vector<std::vector<std::vector<size_t>>> result;
    for (auto& tree : trees) {
      result.push_back(tree->getChildNodeIDs());
    }
    return result;
  }
  std::vector<std::vector<size_t>> getSplitVarIDs() {
    std::vector<std::vector<size_t>> result;
    for (auto& tree : trees) {
      result.push_back(tree->getSplitVarIDs());
    }
    return result;
  }
  std::vector<std::vector<double>> getSplitValues() {
    std::vector<std::vector<double>> result;
    for (auto& tree : trees) {
      result.push_back(tree->getSplitValues());
    }
    return result;
  }
  const std::vector<double>& getVariableImportance() const {
    return variable_importance;
  }
  double getOverallPredictionError() const {
    return overall_prediction_error;
  }
  const std::vector<std::vector<std::vector<double>> >& getPredictions() const {
    return predictions;
  }
  size_t getDependentVarId() const {
    return dependent_varID;
  }
  size_t getNumTrees() const {
    return num_trees;
  }
  uint getMtry() const {
    return mtry;
  }
  uint getMinNodeSize() const {
    return min_node_size;
  }
  size_t getNumIndependentVariables() const {
    return num_independent_variables;
  }

  const std::vector<bool>& getIsOrderedVariable() const {
    return data->getIsOrderedVariable();
  }

  std::vector<std::vector<size_t>> getInbagCounts() const {
    std::vector<std::vector<size_t>> result;
    for (auto& tree : trees) {
      result.push_back(tree->getInbagCounts());
    }
    return result;
  }
  // Variable importance for all variables in forest
  std::vector<double> variable_importance;
  ImportanceMode importance_mode;
  std::ostream* verbose_out;
  std::string output_prefix;
  std::string outputDirectory;
  std::vector<std::vector<std::vector<double>>> predictions;
  void sendTreesMPI(std::vector<double> &fData, std::vector<int> &tLocations, int &nTrees, std::vector<int> &nNodes, std::vector<double> &var_imp);
  void sendTrees(std::vector<double> &classValues, std::vector<uint> &responseClassIDs);
  virtual void sendTreesInternal(std::vector<double> &classValues, std::vector<uint> &responseClassIDs) = 0;
protected:
  void grow();
  virtual void growInternal() = 0;

  // Predict using existing tree from file and data as prediction data
  void predict();
  virtual void predictInternal() = 0;

  void computePredictionError();
  virtual void computePredictionErrorInternal() = 0;

  void computePermutationImportance();

  // Multithreading methods for growing/prediction/importance, called by each thread
  void growTreesInThread(uint thread_idx, std::vector<double>* variable_importance);
  void predictTreesInThread(uint thread_idx, const Data* prediction_data, bool oob_prediction);
  void computeTreePermutationImportanceInThread(uint thread_idx, std::vector<double>* importance, std::vector<double>* variance);

  // Load forest from file
  void loadFromFile(std::string filename);
  virtual void loadFromFileInternal(std::ifstream& infile) = 0;

  // Set split select weights and variables to be always considered for splitting
  void setSplitWeightVector(std::vector<std::vector<double>>& split_select_weights);
  void setAlwaysSplitVariables(std::vector<std::string>& always_split_variable_names);

  // Show progress every few seconds
#ifdef OLD_WIN_R_BUILD
  void showProgress(std::string operation, clock_t start_time, clock_t& lap_time);
#else
  void showProgress(std::string operation);
#endif



  // Verbose output stream, cout if verbose==true, logfile if not
  // std::ostream* verbose_out;

  size_t num_trees;
  uint mtry;
  MtryType mtryType;
  uint min_node_size;
  size_t num_variables;
  size_t num_independent_variables;
  uint seed;
  size_t dependent_varID;
  size_t num_samples;
  bool prediction_mode;
  MemoryMode memory_mode;
  bool sample_with_replacement;
  bool memory_saving_splitting;
  SplitRule splitrule;
  bool predict_all;
  bool keep_inbag;
  double sample_fraction;
  bool holdout;
  PredictionType prediction_type;
  uint num_random_splits;
  int useMPI;
  int rank;
  int size;
  // MAXSTAT splitrule
  double alpha;
  double minprop;

  // Multithreading
  uint num_threads;
  std::vector<uint> thread_ranges;
#ifndef OLD_WIN_R_BUILD
  std::mutex mutex;
  std::condition_variable condition_variable;
#endif



  // std::vector<std::vector<std::vector<double>>> predictions;
  double overall_prediction_error;

  // Weight vector for selecting possible split variables, one weight between 0 (never select) and 1 (always select) for each variable
  // Deterministic variables are always selected
  std::vector<size_t> deterministic_varIDs;
  std::vector<size_t> split_select_varIDs;
  std::vector<std::vector<double>> split_select_weights;

  // Bootstrap weights
  std::vector<double> case_weights;

  // Random number generator
  std::mt19937_64 random_number_generator;

  // std::string output_prefix;
  // ImportanceMode importance_mode;

  // Variable importance for all variables in forest
  //std::vector<double> variable_importance;

  // Computation progress (finished trees)
  size_t progress;
#ifdef R_BUILD
  size_t aborted_threads;
  bool aborted;
#endif

private:
  DISALLOW_COPY_AND_ASSIGN(Forest);
};

#endif /* FOREST_H_ */
