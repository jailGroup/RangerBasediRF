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


#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <chrono>
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>

#include <mpi.h>
#include <sys/stat.h>

#include "globals.h"
#include "ArgumentHandler.h"
#include "ForestClassification.h"
#include "ForestRegression.h"
#include "ForestSurvival.h"
#include "ForestProbability.h"
#include "TreeRegression.h"
#include "TreeClassification.h"


int main(int argc, char **argv) {

	std::chrono::time_point<std::chrono::system_clock> startCode, endCode;
	startCode = std::chrono::system_clock::now();

	ArgumentHandler arg_handler(argc, argv);

	if (arg_handler.processArguments() != 0) {
		return 0;
	}
	arg_handler.checkArguments();
	std::cout << "ArgHandler useMPI value: " << arg_handler.useMPI << "\n";

    //If useMPI is true - parallel forest creation

		if (arg_handler.useMPI == 1 && !arg_handler.predict.empty()){
			throw std::runtime_error("MPI cannot be used with prediction, single node only");
		}

		//Test output directory
		struct stat sb;
		if (stat(arg_handler.outputDirectory.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)){
			std::cout << "Directory is good\n";
		}
		else{
			//Directory either isn't accessible or doesn't exist
			const int dir_err = mkdir(arg_handler.outputDirectory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
			if (-1 == dir_err){
				std::cout << "Error creating directory\n";
				exit(1);
			}
		}

    if (arg_handler.useMPI == 1){
    	int rank;
			int size;

			MPI_Init(&argc,&argv);

    	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    	MPI_Comm_size(MPI_COMM_WORLD, &size);
			char processorName[20];
			int nameLength;
			MPI_Get_processor_name(processorName,&nameLength);
			std::cout << "This is rank: " << rank << "\n";
			for (int i = 0; i < nameLength; i++){
				std::cout << processorName[i];
			}
			std::cout << '\n';

			Data* oldForestData;
			std::ostream* verbose_out;
			std::ostream* verbose_time;
			Forest* newForest;
			int numLoops = arg_handler.numIterations;
			Forest* forest2;

//Beginning of First loop
			std::chrono::time_point<std::chrono::system_clock> startFirstLoop, endFirstLoop;
			startFirstLoop = std::chrono::system_clock::now();
			Forest* forest = 0;
			try {

				// Handle command line arguments
				if (arg_handler.processArguments() != 0) {
					return 0;
				}
				arg_handler.checkArguments();

				// Create forest object
				switch (arg_handler.treetype) {
					case TREE_CLASSIFICATION:
						if (arg_handler.probability) {
							forest = new ForestProbability;
			  		}
			  		else {
							forest = new ForestClassification;
			  		}
			  		break;
					case TREE_REGRESSION:
						forest = new ForestRegression;
			  		break;
					case TREE_SURVIVAL:
			  		forest = new ForestSurvival;
			  		break;
					case TREE_PROBABILITY:
			  		forest = new ForestProbability;
			  		break;
				}

				if (arg_handler.verbose) {
					verbose_out = &std::cout;
				}
				else {
					std::ofstream* logfile = new std::ofstream();
					std::string strRank = std::to_string(rank);
					std::string directory = arg_handler.outputDirectory;
					logfile->open(directory+ "/" + arg_handler.outprefix + strRank + ".log");
			  	if (!logfile->good()) {
						throw std::runtime_error("Could not write to logfile.");
			  	}
			  	verbose_out = logfile;

					std::ofstream* timefile = new std::ofstream();
					timefile->open(directory+ "/" + arg_handler.outprefix + strRank + ".time");
			  	if (!timefile->good()) {
						throw std::runtime_error("Could not write to timefile.");
			  	}
			  	verbose_time = timefile;
				}
				std::cout << "About to initCpp with rank: " << rank << "\n";
				std::cout << std::flush;
				int iteration = 0;

				std::chrono::time_point<std::chrono::system_clock> startInit, endInit;
				startInit = std::chrono::system_clock::now();


				forest->initCpp(arg_handler.depvarname, arg_handler.memmode, arg_handler.file, arg_handler.yfile, arg_handler.mtry, arg_handler.mtryType,
					arg_handler.outprefix, arg_handler.ntree, verbose_out, arg_handler.seed, arg_handler.nthreads,
					arg_handler.predict, arg_handler.impmeasure, arg_handler.targetpartitionsize, arg_handler.splitweights,
					arg_handler.alwayssplitvars, arg_handler.statusvarname, arg_handler.replace, arg_handler.catvars,
					arg_handler.savemem, arg_handler.splitrule, arg_handler.caseweights, arg_handler.predall, arg_handler.fraction,
					arg_handler.alpha, arg_handler.minprop, arg_handler.holdout, arg_handler.predictiontype,
					arg_handler.randomsplits, arg_handler.useMPI, rank, size, arg_handler.outputDirectory, verbose_time);


				endInit = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsedInit = endInit - startInit;
				time_t tt = std::chrono::system_clock::to_time_t(endInit);
				*verbose_out << "Forest Init processing time (" << ctime(&tt) << "): " << elapsedInit.count() << "s." << std::endl;

				std::chrono::time_point<std::chrono::system_clock> startRun, endRun;
				startRun = std::chrono::system_clock::now();

				forest->run(true);

				endRun = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsedRun = endRun - startRun;
				tt = std::chrono::system_clock::to_time_t(endRun);
				*verbose_out << "Forest Run processing time (" << ctime(&tt) << "): " << elapsedRun.count() << "s." << std::endl;

				int varImportanceLength = forest->getNumIndependentVariables();

		//MPI Variables
				//forests contains nodeValues, splitVarIDs, and splitValues for each tree
				//in that order
				double *forests;

				//numTreesTotalbuf contains the number of trees created at each rank
				//numTreesTotalbuf[i] where i goes from 0 to size
				int *numTreesTotalbuf;

				//numNodesTotal contains the number of nodes for each tree from every rank
				//numNodesTotal.size = sum(numTreesTotalbuf[i]) for all i
				int *numNodesTotal;

				//allLocationsbuf contains the index locations for the start of each set
				//of nodeValues, splitVarIDs, and splitValues in forests
				//allLocationsbuf.size = 3*sum(numTreesTotalbuf[i]) for all i
				int *allLocationsbuf;

				//allVariableImportance contains the variable importance for each variable
				//from each node
				//allVariableImportance.size = size*varImportanceLength
				double *allVariableImportance;

				if (rank == 0){
						numTreesTotalbuf = (int *)malloc(size*sizeof(int));
				}

					//Gather all information to rank 0
					std::cout << "Start data gather with rank: " << rank << "\n";
					std::vector<double> forestData;
					std::vector<int> treeLocations;
					int numTrees;
					std::vector<int> numNodes;
					std::vector<double> variableImportance;
					std::vector<double> classVals;
					std::vector<uint> respClassIDs;

					forest->sendTreesMPI(forestData, treeLocations, numTrees, numNodes, variableImportance);
					std::chrono::time_point<std::chrono::system_clock> sendTreesTime;
					sendTreesTime = std::chrono::system_clock::now();
					tt = std::chrono::system_clock::to_time_t(sendTreesTime);
					std::cout << "SendTrees Time: " << ctime(&tt) << " Rank: " << rank;
					std::cout << std::flush;

					switch (arg_handler.treetype) {
						case TREE_CLASSIFICATION:
							if (arg_handler.probability) {
								//No extra information needed than sendTreesMPI
							}
							else {
								//More data than basic sendTreesMPI needed
								forest->sendTrees(classVals, respClassIDs);
							}
							break;
						case TREE_REGRESSION:
							//No extra information needed than sendTreesMPI
							break;
						case TREE_SURVIVAL:
						//Not currently implemented
							break;
						case TREE_PROBABILITY:
						//Not currently implemented
							break;
					}

					std::cout << "Sent forest data to Main with rank: " << rank << "\n";


		//Send numTrees from each rank to rank 0 - stored in numTreesTotalbuf
					MPI_Gather(&numTrees,1,MPI_INT,numTreesTotalbuf,1,MPI_INT,0,MPI_COMM_WORLD);
					std::cout << "First MPIGather - numTrees with rank: " << rank << "\n";
					std::cout << std::flush;


		//Send treeLocations from each rank to rank 0
		//Will be stored in allLocationsbuf at rank 0
					int *locationCounts;
					//For each rank i: locationCounts[i] contains the size of treeLocations
					//sent from rank i to rank 0
					locationCounts = (int *)malloc(size*sizeof(int));
					if (rank == 0){
						for (int i = 0; i < size; i++){
							locationCounts[i] = numTreesTotalbuf[i] * 3;
						}
					}
					int *displacements;
					//For each rank i: displacements[i] contains the offset from index 0
					//in allLocationsbuf or each treeLocations data chunk being sent to rank 0
					displacements = (int *)malloc(size*sizeof(int));
					int locationSize = 0;
					if (rank == 0){
						displacements[0] = 0;
						for (int i = 1; i < size; i++){
							displacements[i] = displacements[i-1] + locationCounts[i-1];
						}
						for (int i = 0; i < size; i++){
							locationSize += numTreesTotalbuf[i] * 3;
						}
						allLocationsbuf = (int *)malloc(locationSize*sizeof(int));
					}
					int numLocations = numTrees*3;
					//Gather treeLocation vectors from each rank, store in allLocations vector, with offset displacements[rank] and vector lengths locationCounts[rank]
					MPI_Gatherv(&treeLocations[0],numLocations,MPI_INT,allLocationsbuf,locationCounts,displacements,MPI_INT,0,MPI_COMM_WORLD);
					std::cout << "Finished first MPI_Gatherv in main() with rank: " << rank << "\n";
					std::cout << std::flush;
					std::chrono::time_point<std::chrono::system_clock> sendTreeLocations;
					sendTreeLocations = std::chrono::system_clock::now();
					tt = std::chrono::system_clock::to_time_t(sendTreeLocations);
					std::cout << "MPI Gatherv TreeLocations Time: " << ctime(&tt) << " Rank: " << rank;
					std::cout << std::flush;
					free(locationCounts);
					free(displacements);
		//Send numNodes from each rank to rank 0
		//Store in numNodesTotal
					//Displacement from beginning of numNodesTotal that each node count per tree starts, at each displacement a new 'forest' starts
					//Use numTreesTotal at [rank] offset to determine number of trees associated with that forest
					int *displacementsNodes;
					//For each rank i: displacementsNodes[i] contains the offset from index 0
					//for each chunk of data being sent to rank 0
					displacementsNodes = (int *)malloc(size*sizeof(int));

					int nodeSize = 0;
					if (rank == 0){
						displacementsNodes[0] = 0;
						for (int i = 1; i < size; i++){
							displacementsNodes[i] = displacementsNodes[i-1] + numTreesTotalbuf[i-1];
						}
						for (int i = 0; i < size; i++){
							nodeSize += numTreesTotalbuf[i];
						}
						numNodesTotal = (int *)malloc(nodeSize*sizeof(int));
					}

						int *numTreesArray;
						numTreesArray = (int *)malloc(size*sizeof(int));

					if (rank == 0){
							for (int i = 0; i < size; i++){
								numTreesArray[i] = numTreesTotalbuf[i];
							}
					}

					std::cout << "About to start MPI_Gatherv for numNodesTotal, in main() with rank: " << rank << "\n";
					//Gather numNodes vector from each rank, store in numNodesTotal vector, with offset displacementsNodes[rank], and vector lengths numTreesTotal[rank]
				  MPI_Gatherv(&numNodes[0],numTrees,MPI_INT,numNodesTotal,numTreesArray,displacementsNodes,MPI_INT,0,MPI_COMM_WORLD);
					//Calculate forestData size from numTrees and numNodes
					std::cout << "Finished MPI_Gatherv for numNodes in main() with rank: " << rank << "\n";
					std::cout << std::flush;
					free(displacementsNodes);
					free(numTreesArray);






				//If Classification
			// 	if (arg_handler.treetype == TREE_CLASSIFICATION & !arg_handler.probability ){
			// 		int predSize = preds.size();
			// 		//Gather pred vector size from each rank, store in numPredictionsTotal[rank]
			// 		MPI_Gather(&predSize,1,MPI_INT,&numPredictionsTotal[0],1,MPI_INT,MPI_COMM_WORLD)
			// 		std::vector<int> predDisplacements;
			// 		predDisplacements[0] = 0;
			// 		for (int i = 1; i < size; i++){
			// 			predDisplacements[i] = predDisplacements[i-1] + numPredictionsTotal[i-1];
			// 		}
			// 		//Gather preds vector from each rank, store in allPredictions vector, with offset predDisplacements[rank], and vector lengths numPredictionsTotal[rank]
			// 		MPI_Gatherv(&preds[0],predSize,MPI_DOUBLE,&allPredictions[0],numPredictionsTotal,predDisplacements,MPI_DOUBLE,0,MPI_COMM_WORLD);
			//
			// //Add section about std::vector<double> class_values;
			// //and std::vector<uint> response_classIDs;
			// 	}
			// 	//Or if Regression
			// 	else if (arg_handler.treetype == TREE_REGRESSION) {
			// 		int predSize = preds.size();
			// 		//Gather pred vector size from each rank, store in numPredictionsTotal[rank]
			// 		MPI_Gather(&predSize,1,MPI_INT,&numPredictionsTotal[0],1,MPI_INT,MPI_COMM_WORLD
			// 		std::vector<int> predDisplacements;
			// 		predDisplacements[0] = 0;
			// 		for (int i = 1; i < size; i++){
			// 			predDisplacements[i] = predDisplacements[i-1] + numPredictionsTotal[i-1];
			// 		}
			// 		//Gather preds vector from each rank, store in allPredictions vector, with offset predDisplacements[rank], and vector lengths numPredictionsTotal[rank]
			// 		MPI_Gatherv(&preds[0],predSize,MPI_DOUBLE,&allPredictions[0],numPredictionsTotal,predDisplacements,MPI_DOUBLE,0,MPI_COMM_WORLD);
			// 	}




		//send variableImportance from each rank to rank 0
		//stored in allVariableImportance at rank 0
					int allVarSize = varImportanceLength*size;
					if (rank == 0){
						//Update allLocationsbuf to contain continuous indices
						//Rather than restarting at 0 for each rank-chunk
						//Ex: [0 2 5 7 9 11 | 0 4 9 12 15 17] -> [0 2 5 7 9 11 | 13 17 20 23 26 28]
						for (int i = 0; i < (size-1); i++){
							int totNumTrees = numTreesTotalbuf[i];
							int allLocShift = totNumTrees * 3 * (i+1);
							int diffCount = allLocationsbuf[allLocShift - 1] - allLocationsbuf[allLocShift - 2];
							int update = diffCount + allLocationsbuf[allLocShift - 1];
							//propage update forward through array
							for (int j = allLocShift; j < (allLocShift+totNumTrees*3); j++){
								allLocationsbuf[j] = allLocationsbuf[j] + update;
							}
						}
						std::cout << std::flush;
						allVariableImportance = (double *)malloc(allVarSize*sizeof(double));
					}

					MPI_Gather(&variableImportance[0],varImportanceLength,MPI_DOUBLE,allVariableImportance,varImportanceLength,MPI_DOUBLE,0,MPI_COMM_WORLD);
					std::cout << "Finished MPI_Gather for variableImportance with rank: " << rank << "\n" << std::flush;


		//Send forestData from each rank to rank 0
		//Stored in forests at rank 0
					//Length of forestData at each compute node
					int forestDataLength = 0;
					for (int i = 0; i < numNodes.size(); i++){
						forestDataLength += (numNodes[i]*2 + numNodes[i] + numNodes[i]);
					}

					int *forestLengthArray;
					forestLengthArray = (int *)malloc(size*sizeof(int));
					int *forestDisplacement;
					forestDisplacement = (int *)malloc(size*sizeof(int));
					int forestLength = 0;
					if (rank == 0){
						forestDisplacement[0] = 0;
						for (int i = 0; i < nodeSize; i++){
							forestLength = forestLength + (numNodesTotal[i]*2 + numNodesTotal[i] + numNodesTotal[i]);
						}
						int offset = 0;
						int numberTrees = 0;
						for (int i = 0; i < size; i++){
							numberTrees = numTreesTotalbuf[i];
							forestLengthArray[i] = 0;
							for (int j = offset; j < (numberTrees+offset); j++){
								forestLengthArray[i] += (numNodesTotal[j]*2 + numNodesTotal[j] + numNodesTotal[j]);
							}

							offset += numberTrees;
						}

						std::cout << "forestLength: " << forestLength << '\n';
						//forests->resize(forestLength);
						forests = (double *)malloc(forestLength*sizeof(double));
						std::cout << "After forest resize\n" << std::flush;
						int offsetDisp = 0;
						int numberTreesDisp = 0;
						forestDisplacement[0] = 0;

						for (int i = 1; i < size; i++){
							forestDisplacement[i] = forestDisplacement[i-1] + forestLengthArray[i-1];
						}
					}

					//Gather forestData vector from each rank, store in forests vector, with offset forestDisplacement[rank], and vector lengths forestLength[rank]
					MPI_Gatherv(&forestData[0],forestDataLength,MPI_DOUBLE,forests,forestLengthArray,forestDisplacement,MPI_DOUBLE,0,MPI_COMM_WORLD);
					std::cout << "Finished MPI_Gatherv for forestData with rank: " << rank << "\n";
					std::cout << std::flush;
					std::chrono::time_point<std::chrono::system_clock> sendForestDataTime;
					sendForestDataTime = std::chrono::system_clock::now();
					tt = std::chrono::system_clock::to_time_t(sendForestDataTime);
					std::cout << "Send ForestData Time: " << ctime(&tt) << " Rank: " << rank;
					std::cout << std::flush;
					free(forestLengthArray);
					free(forestDisplacement);
					//Forest* newForest;


//Rebuild trees
			if (rank == 0){

				//While not at allLocations.end(), iterate and pull data from Forests
				std::cout << "Starting Rebuild Section\n";
				newForest = 0;

					// Create forest object
					switch (arg_handler.treetype) {
						case TREE_CLASSIFICATION:
							if (arg_handler.probability) {
								newForest = new ForestProbability;
				  		}
				  		else {
								newForest = new ForestClassification;
				  		}
				  		break;
						case TREE_REGRESSION:
							newForest = new ForestRegression;
				  		break;
						case TREE_SURVIVAL:
				  		newForest = new ForestSurvival;
				  		break;
						case TREE_PROBABILITY:
				  		newForest = new ForestProbability;
				  		break;
					}
					int totalTreeCount = 0;
					for (int i = 0; i < size; i++){
						totalTreeCount = totalTreeCount + numTreesTotalbuf[i];
					}
					newForest->trees.reserve(totalTreeCount);
					std::cout << "newForest reserve\n";
					std::cout << std::flush;

				std::vector<std::vector<size_t> > child_nodeIDsNew;

				//std::cout << "new Child nodes created\n";
				std::vector<size_t> split_varIDsNew;
				std::vector<double> split_valuesNew;

				//Iterate through allLocations to get index locations in Forests
				//Pattern: TreeLocation, split_varIDsLocation, split_valuesLocation, repeat
				//0,3,6,9 etc are treeLocations
				//1,4,7,10 etc are split_varIDsLocations
				//2,5,8,11 etc are split_valuesLocations
				//std::cout << "Start rebuilding Forest at rank 0\n";
				//for (int i = 0; i < allLocations->size()-1; i++){
				int allLocCount = 0;
				int numTreesCount = 0;
				// std::cout << "allLocationsbuf.size(): " << locationSize << '\n';
				// for (int i = 0; i <locationSize; i++){
				// 	//std::cout << allLocationsbuf[i] << " ";
				// 	if (i % 15 == 0){
				// 		//std::cout << '\n';
				// 	}
				// }
				// std::cout << '\n';
				while (allLocCount < (locationSize-2)){
					std::vector<size_t> newColumn;
					child_nodeIDsNew.push_back(newColumn);
					child_nodeIDsNew.push_back(newColumn);
					std::cout << std::flush;

					//Grab the childNodes
					int x = 0;
					while (x < numNodesTotal[numTreesCount]){
						child_nodeIDsNew[0].push_back(forests[allLocationsbuf[allLocCount]+x]);
						x++;
					}
					int y = 0;
					while (y < numNodesTotal[numTreesCount]){
						child_nodeIDsNew[1].push_back(forests[allLocationsbuf[allLocCount]+x + y]);
						y++;
					}
					//std::cout << "After x while loop\n";
					numTreesCount++;
					allLocCount++;
					//Grab the split_varIDs
					int k = 0;
					int diff = allLocationsbuf[allLocCount+1] - allLocationsbuf[allLocCount];
					while (k < (diff)){
						split_varIDsNew.push_back((size_t) forests[allLocationsbuf[allLocCount]+k]);
						k++;
					}
					allLocCount++;
					//Grab the split_values
					int m = 0;
					while (m < (diff)){
						split_valuesNew.push_back((double) forests[allLocationsbuf[allLocCount]+m]);
						m++;
						std::cout << std::flush;
					}
					allLocCount++;

				//Save childNodes, splitVars, and splitValues as new tree in Node 0's forest

				std::cout << std::flush;
				switch (arg_handler.treetype) {
					case TREE_CLASSIFICATION:
						if (arg_handler.probability) {

						}
						else {
								newForest->trees.push_back(new TreeClassification(child_nodeIDsNew, split_varIDsNew, split_valuesNew, &classVals, &respClassIDs));
						}
						break;

					case TREE_REGRESSION:
							newForest->trees.push_back(new TreeRegression(child_nodeIDsNew, split_varIDsNew, split_valuesNew));
						break;
					// case TREE_SURVIVAL:
					//
					// 	break;
					// case TREE_PROBABILITY:
					//
					// 	break;


				}
				//Reset variables to empty before next loop
				child_nodeIDsNew[0].clear();
				child_nodeIDsNew[1].clear();
				child_nodeIDsNew.clear();
				split_varIDsNew.clear();
				split_valuesNew.clear();

			}




			//Average variable importances for each variable
			//std::vector<double> allVariableImportance;
			std::vector<double> varImportance(varImportanceLength,0);
			for (int i = 0; i < varImportanceLength; i++) {
				for (int j = 0; j < size; j++){
					varImportance[i] = varImportance[i] + allVariableImportance[i+j*varImportanceLength];
				}
				varImportance[i] = varImportance[i]/size;
			}
			std::cout << "set varImportance\n";
			std::cout << std::flush;
			newForest->variable_importance = varImportance;
			std::cout << "assign varImportance to newForest\n";
			std::cout << std::flush;

		}


			std::cout << "Before forest Write\n" << std::flush;
			std::cout << "Forest mtry var: " << forest->getMtry() << '\n' << std::flush;
    	forest->writeOutput();
			std::cout << "After forest write\n" << std:: flush;
			std::cout << "about to set forest data to oldForest with rank: " << rank << "\n";
			std::cout << std::flush;
			std::chrono::time_point<std::chrono::system_clock> sendoldForestTime;
			sendoldForestTime = std::chrono::system_clock::now();
			tt = std::chrono::system_clock::to_time_t(sendoldForestTime);
			std::cout << "Set forest data to oldForest Time: " << ctime(&tt) << " Rank: " << rank;
			std::cout << std::flush;
			oldForestData = forest->data;
			std::cout << "oldForestData var names[0]: " << oldForestData->getVariableNames()[0] << '\n';
			std::cout << "set forest data to old Forest with rank: " << rank << "\n";
			std::cout << std::flush;
			if (rank == 0){
				newForest->verbose_out = verbose_out;
				newForest->output_prefix = arg_handler.outprefix;
				newForest->predictions = forest->getPredictions();
				newForest->importance_mode = arg_handler.impmeasure;
				newForest->data = oldForestData;
				std::cout << "In main, before setting split vars\n" << std::flush;
				newForest->data->no_split_variables = oldForestData->no_split_variables;
				std::cout << "In main, after setting split vars, before size\n";
				newForest->outputDirectory = arg_handler.outputDirectory;

				std::cout << "In main, splitVar size: " << oldForestData->no_split_variables.size() << '\n' << std::flush;
				newForest->writeOutputNewForest(0);
				if (arg_handler.write) {
					newForest->saveToFile(0);
				}
			}
    	*verbose_out << "Finished Ranger." << std::endl;

			if (rank == 0){
				newForest->writeSplitWeightsFile(arg_handler.outprefix);
				std::cout << "write split weights file\n";
				std::cout << std::flush;
			}
			std::cout << "before delete forest with rank: " << rank << "\n";
			std::cout << std::flush;
			if (arg_handler.numIterations != 1){
				delete forest;
			}
			std::cout << "after delete Forest with rank: " << rank << "\n";
			std::cout << std::flush;
			if (rank == 0 && arg_handler.numIterations != 1){
				delete newForest;
			}
			if (rank == 0){
				free(numTreesTotalbuf);
				free(allLocationsbuf);
				free(numNodesTotal);
				free(allVariableImportance);
				free(forests);
			}
}

	catch (std::exception& e) {
		if (rank == 0){
			std::cerr << "Error: " << e.what() << " Ranger will EXIT now." << std::endl;
			delete forest;
			return -1;
		}
	}

	//End of first loop







			MPI_Barrier(MPI_COMM_WORLD);
			endFirstLoop = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsedFirstLoop = endFirstLoop - startFirstLoop;
			time_t tt = std::chrono::system_clock::to_time_t(endFirstLoop);
			*verbose_time << "First Loop time: " << elapsedFirstLoop.count() << "s." << std::endl;


			std::cout << "End of first loop with rank: " << rank << '\n';
			std::cout << std::flush;

			//Duplicate everything within first loop - inside a for loop of n-1 iterations
			int loop;
			std::cout << "After creating loop var\n";
			std::cout << std::flush;



			for (loop = 1; loop < numLoops; loop++){
				std::chrono::time_point<std::chrono::system_clock> startNextLoop, endNextLoop;
				startNextLoop = std::chrono::system_clock::now();

				std::cout << "LOOP: " << loop << '\n';
				std::cout << std::flush;
				forest2 = 0;
				try {

					// Handle command line arguments
					if (arg_handler.processArguments() != 0) {
						return 0;
					}
					arg_handler.checkArguments();

					// Create forest object
					switch (arg_handler.treetype) {
						std::cout << "Create new forest2 with rank: " << rank << '\n';
						std::cout << std::flush;
						case TREE_CLASSIFICATION:
							if (arg_handler.probability) {
								forest2 = new ForestProbability;
							}
							else {
								forest2 = new ForestClassification;
							}
							break;
						case TREE_REGRESSION:
							forest2 = new ForestRegression;
							break;
						case TREE_SURVIVAL:
							forest2 = new ForestSurvival;
							break;
						case TREE_PROBABILITY:
							forest2 = new ForestProbability;
							break;
					}


					std::cout << "About to initCppData with rank: " << rank << "\n";
					std::cout << std::flush;
					int iteration = 0;

					//MPI_Barrier(MPI_COMM_WORLD);
					std::chrono::time_point<std::chrono::system_clock> startInit2, endInit2;
					startInit2 = std::chrono::system_clock::now();

					//Grab data from previous run rather than reading in again
					std::string weightFile = arg_handler.outputDirectory + "/" + arg_handler.outprefix + ".splitWeights";
					forest2->initCppData(arg_handler.depvarname, arg_handler.memmode, arg_handler.file, arg_handler.yfile, arg_handler.mtry, arg_handler.mtryType,
						arg_handler.outprefix, arg_handler.ntree, verbose_out, arg_handler.seed, arg_handler.nthreads,
						arg_handler.predict, arg_handler.impmeasure, arg_handler.targetpartitionsize, weightFile,
						arg_handler.alwayssplitvars, arg_handler.statusvarname, arg_handler.replace, arg_handler.catvars,
						arg_handler.savemem, arg_handler.splitrule, arg_handler.caseweights, arg_handler.predall, arg_handler.fraction,
						arg_handler.alpha, arg_handler.minprop, arg_handler.holdout, arg_handler.predictiontype,
						arg_handler.randomsplits, arg_handler.useMPI, rank, size, oldForestData, arg_handler.outputDirectory);

					//MPI_Barrier(MPI_COMM_WORLD);
					endInit2 = std::chrono::system_clock::now();
					std::chrono::duration<double> elapsedInit2 = endInit2 - startInit2;
					time_t tt = std::chrono::system_clock::to_time_t(endInit2);
					*verbose_out << "Forest2 Init processing time (All) (" << ctime(&tt) << "): " << elapsedInit2.count() << "s." << std::endl;


					std::cout << "Initialized Forest2 with rank: " << rank << "\n";
					std::cout << std::flush;


					//MPI_Barrier(MPI_COMM_WORLD);
					std::chrono::time_point<std::chrono::system_clock> startRun2, endRun2;
					startRun2 = std::chrono::system_clock::now();

					forest2->run(true);

					//MPI_Barrier(MPI_COMM_WORLD);
					endRun2 = std::chrono::system_clock::now();
					std::chrono::duration<double> elapsedRun2 = endRun2 - startRun2;
					tt = std::chrono::system_clock::to_time_t(endRun2);
					*verbose_out << "Forest2 Run processing time (All) (" << ctime(&tt) << "): " << elapsedRun2.count() << "s." << std::endl;


					std::cout << "Forest2 Ran with rank: " << rank << "\n";
					std::cout << std::flush;


					int varImportanceLength = forest2->getNumIndependentVariables();

					//forests contains nodeValues, splitVarIDs, and splitValues for each tree
					//in that order
					double *forests;

					//numTreesTotalbuf contains the number of trees created at each rank
					//numTreesTotalbuf[i] where i goes from 0 to size
					int *numTreesTotalbuf;

					//numNodesTotal contains the number of nodes for each tree from every node
					//numNodesTotal.size = sum(numTreesTotalbuf[i]) for all i
					int *numNodesTotal;

					//allLocationsbuf contains the index locations for the start of each set
					//of nodeValues, splitVarIDs, and splitValues in forests
					//allLocationsbuf.size = 3*sum(numTreesTotalbuf[i]) for all i
					int *allLocationsbuf;

					//allVariableImportance contains the variable importance for each variable
					//from each node
					//allVariableImportance.size = size*varImportanceLength
					double *allVariableImportance;

					if (rank == 0){
							numTreesTotalbuf = (int *)malloc(size*sizeof(int));
					}
					std::cout << "After vector init in Main()\n";

						//Gather all information to rank 0
						std::cout << "Start data gather with rank: " << rank << "\n";
						std::vector<double> forestData;
						std::vector<int> treeLocations;
						int numTrees;
						std::vector<int> numNodes;
						std::vector<double> variableImportance;
						std::vector<double> classVals;
						std::vector<uint> respClassIDs;

						forest2->sendTreesMPI(forestData, treeLocations, numTrees, numNodes, variableImportance);

						switch (arg_handler.treetype) {
							case TREE_CLASSIFICATION:
								if (arg_handler.probability) {

								}
								else {
									forest2->sendTrees(classVals, respClassIDs);
								}
								break;
							case TREE_REGRESSION:

								break;
							case TREE_SURVIVAL:
								break;
							case TREE_PROBABILITY:
								break;
						}




						std::cout << "Sent forest2 data to Main with rank: " << rank << "\n";

			//Send numTrees from each rank to rank 0 - stored in numTreesTotalbuf
						MPI_Gather(&numTrees,1,MPI_INT,numTreesTotalbuf,1,MPI_INT,0,MPI_COMM_WORLD);
						std::cout << "First MPIGather - numTrees with rank: " << rank << "\n";
						std::cout << std::flush;


			//Send treeLocations from each rank to rank 0
			//Will be stored in allLocationsbuf at rank 0
						std::cout << "About to malloc locationCounts in main()\n";
						int *locationCounts;
						//For each rank i: locationCounts[i] contains the size of treeLocations
						//sent from rank i to rank 0
						locationCounts = (int *)malloc(size*sizeof(int));
						if (rank == 0){
							for (int i = 0; i < size; i++){
								locationCounts[i] = numTreesTotalbuf[i] * 3;
							}
						}
						int *displacements;
						//For each rank i: displacements[i] contains the offset from index 0
						//in allLocationsbuf or each treeLocations data chunk being sent to rank 0
						displacements = (int *)malloc(size*sizeof(int));
						int locationSize = 0;
						if (rank == 0){
							displacements[0] = 0;
							for (int i = 1; i < size; i++){
								displacements[i] = displacements[i-1] + locationCounts[i-1];
							}
							for (int i = 0; i < size; i++){
								locationSize += numTreesTotalbuf[i] * 3;
							}
							allLocationsbuf = (int *)malloc(locationSize*sizeof(int));
						}
						int numLocations = numTrees*3;
						std::cout << "displacements values set\n";
						std::cout << "About to start MPI_Gatherv for allLocationsbuf, in main() with rank: " << rank << "\n";
						//Gather treeLocation vectors from each rank, store in allLocations vector, with offset displacements[rank] and vector lengths locationCounts[rank]
						MPI_Gatherv(&treeLocations[0],numLocations,MPI_INT,allLocationsbuf,locationCounts,displacements,MPI_INT,0,MPI_COMM_WORLD);
						std::cout << "Finished first MPI_Gatherv in main() with rank: " << rank << "\n";
						std::cout << std::flush;
						free(locationCounts);
						free(displacements);

			//Send numNodes from each rank to rank 0
			//Store in numNodesTotal
						//Displacement from beginning of numNodesTotal that each node count per tree starts, at each displacement a new 'forest' starts
						//Use numTreesTotal at [rank] offset to determine number of trees associated with that forest
						int *displacementsNodes;
						//For each rank i: displacementsNodes[i] contains the offset from index 0
						//for each chunk of data being sent to rank 0
						displacementsNodes = (int *)malloc(size*sizeof(int));

						int nodeSize = 0;
						if (rank == 0){
							displacementsNodes[0] = 0;
							for (int i = 1; i < size; i++){
								displacementsNodes[i] = displacementsNodes[i-1] + numTreesTotalbuf[i-1];
							}
							std::cout << "DisplacementsNodes values set\n";
							for (int i = 0; i < size; i++){
								nodeSize += numTreesTotalbuf[i];
							}
							numNodesTotal = (int *)malloc(nodeSize*sizeof(int));
						}

							int *numTreesArray;
							numTreesArray = (int *)malloc(size*sizeof(int));

						if (rank == 0){
								for (int i = 0; i < size; i++){
									numTreesArray[i] = numTreesTotalbuf[i];
								}
						}

						std::cout << "About to start MPI_Gatherv for numNodesTotal, in main() with rank: " << rank << "\n";
						//Gather numNodes vector from each rank, store in numNodesTotal vector, with offset displacementsNodes[rank], and vector lengths numTreesTotal[rank]
						MPI_Gatherv(&numNodes[0],numTrees,MPI_INT,numNodesTotal,numTreesArray,displacementsNodes,MPI_INT,0,MPI_COMM_WORLD);
						//Calculate forestData size from numTrees and numNodes
						std::cout << "Finished MPI_Gatherv for numNodes in main() with rank: " << rank << "\n";
						std::cout << std::flush;
						free(displacementsNodes);
						free(numTreesArray);





					//If Classification
				// 	if (arg_handler.treetype == TREE_CLASSIFICATION & !arg_handler.probability ){
				// 		int predSize = preds.size();
				// 		//Gather pred vector size from each rank, store in numPredictionsTotal[rank]
				// 		MPI_Gather(&predSize,1,MPI_INT,&numPredictionsTotal[0],1,MPI_INT,MPI_COMM_WORLD)
				// 		std::vector<int> predDisplacements;
				// 		predDisplacements[0] = 0;
				// 		for (int i = 1; i < size; i++){
				// 			predDisplacements[i] = predDisplacements[i-1] + numPredictionsTotal[i-1];
				// 		}
				// 		//Gather preds vector from each rank, store in allPredictions vector, with offset predDisplacements[rank], and vector lengths numPredictionsTotal[rank]
				// 		MPI_Gatherv(&preds[0],predSize,MPI_DOUBLE,&allPredictions[0],numPredictionsTotal,predDisplacements,MPI_DOUBLE,0,MPI_COMM_WORLD);
				//
				// //Add section about std::vector<double> class_values;
				// //and std::vector<uint> response_classIDs;
				// 	}
				// 	//Or if Regression
				// 	else if (arg_handler.treetype == TREE_REGRESSION) {
				// 		int predSize = preds.size();
				// 		//Gather pred vector size from each rank, store in numPredictionsTotal[rank]
				// 		MPI_Gather(&predSize,1,MPI_INT,&numPredictionsTotal[0],1,MPI_INT,MPI_COMM_WORLD
				// 		std::vector<int> predDisplacements;
				// 		predDisplacements[0] = 0;
				// 		for (int i = 1; i < size; i++){
				// 			predDisplacements[i] = predDisplacements[i-1] + numPredictionsTotal[i-1];
				// 		}
				// 		//Gather preds vector from each rank, store in allPredictions vector, with offset predDisplacements[rank], and vector lengths numPredictionsTotal[rank]
				// 		MPI_Gatherv(&preds[0],predSize,MPI_DOUBLE,&allPredictions[0],numPredictionsTotal,predDisplacements,MPI_DOUBLE,0,MPI_COMM_WORLD);
				// 	}




			//send variableImportance from each rank to rank 0
			//stored in allVariableImportance at rank 0
						int allVarSize = varImportanceLength*size;
						if (rank == 0){
							std::cout << std::flush;
							//Update allLocationsbuf to contain continuous indices
								//Rather than restarting at 0 for each rank-chunk
							//Ex: [0 2 5 7 9 11 | 0 4 9 12 15 17] -> [0 2 5 7 9 11 | 13 17 20 23 26 28]
							for (int i = 0; i < (size-1); i++){
								int totNumTrees = numTreesTotalbuf[i];
								int allLocShift = totNumTrees * 3 * (i+1);
								int diffCount = allLocationsbuf[allLocShift - 1] - allLocationsbuf[allLocShift - 2];
								int update = diffCount + allLocationsbuf[allLocShift - 1];
								//propage update forward through array
								for (int j = allLocShift; j < (allLocShift+totNumTrees*3); j++){
									allLocationsbuf[j] = allLocationsbuf[j] + update;
								}
							}
							std::cout << std::flush;
							allVariableImportance = (double *)malloc(allVarSize*sizeof(double));
						}
						MPI_Gather(&variableImportance[0],varImportanceLength,MPI_DOUBLE,allVariableImportance,varImportanceLength,MPI_DOUBLE,0,MPI_COMM_WORLD);
						std::cout << "Finished MPI_Gather for variableImportance with rank: " << rank << "\n";


			//Send forestData from each rank to rank 0
			//Stored in forests at rank 0
						//Length of forestData at each compute node
						int forestDataLength = 0;
						for (int i = 0; i < numNodes.size(); i++){
							forestDataLength += (numNodes[i]*2 + numNodes[i] + numNodes[i]);
						}

						int *forestLengthArray;
						forestLengthArray = (int *)malloc(size*sizeof(int));
						int *forestDisplacement;
						forestDisplacement = (int *)malloc(size*sizeof(int));
						std::cout << "Variable initialization for forest2 data done\n";
						int forestLength = 0;
						if (rank == 0){
							forestDisplacement[0] = 0;
							for (int i = 0; i < nodeSize; i++){
								forestLength = forestLength + (numNodesTotal[i]*2 + numNodesTotal[i] + numNodesTotal[i]);
							}
							int offset = 0;
							int numberTrees = 0;
							for (int i = 0; i < size; i++){
								numberTrees = numTreesTotalbuf[i];
								forestLengthArray[i] = 0;
								for (int j = offset; j < (numberTrees+offset); j++){
									forestLengthArray[i] += (numNodesTotal[j]*2 + numNodesTotal[j] + numNodesTotal[j]);
								}

								offset += numberTrees;
							}

							std::cout << "Right before forest2 resize\n";
							std::cout << "forestLength: " << forestLength << '\n';
							//forests->resize(forestLength);
							forests = (double *)malloc(forestLength*sizeof(double));
							std::cout << "After forest2 resize\n";
							int offsetDisp = 0;
							int numberTreesDisp = 0;
							forestDisplacement[0] = 0;
							for (int i = 1; i < size; i++){
								forestDisplacement[i] = forestDisplacement[i-1] + forestLengthArray[i-1];
							}
						}
						std::cout << "Right before forest2 data mpi transfer\n";
						std::chrono::time_point<std::chrono::system_clock> startGather, endGather;
						startGather = std::chrono::system_clock::now();
						//Gather forestData vector from each rank, store in forests vector, with offset forestDisplacement[rank], and vector lengths forestLength[rank]
						MPI_Gatherv(&forestData[0],forestDataLength,MPI_DOUBLE,forests,forestLengthArray,forestDisplacement,MPI_DOUBLE,0,MPI_COMM_WORLD);
						endGather = std::chrono::system_clock::now();
						std::chrono::duration<double> elapsedGather = endGather - startGather;
						tt = std::chrono::system_clock::to_time_t(endGather);
						std::cout << "Finished MPI_Gatherv for forestData with rank: " << rank << " time ("<< ctime(&tt)<< "): "<< elapsedGather.count() << "\n";
						std::cout << std::flush;
						free(forestLengthArray);
						free(forestDisplacement);

						// Forest* newForest;
	//Rebuild trees
				if (rank == 0){
					//While not at allLocations.end(), iterate and pull data from Forests
					std::cout << "Starting Rebuild Section\n";
					newForest = 0;

						// Create forest object
						switch (arg_handler.treetype) {
							case TREE_CLASSIFICATION:
								if (arg_handler.probability) {
									newForest = new ForestProbability;
								}
								else {
									newForest = new ForestClassification;
								}
								break;
							case TREE_REGRESSION:
								newForest = new ForestRegression;
								break;
							case TREE_SURVIVAL:
								newForest = new ForestSurvival;
								break;
							case TREE_PROBABILITY:
								newForest = new ForestProbability;
								break;
						}
						int totalTreeCount = 0;
						for (int i = 0; i < size; i++){
							totalTreeCount = totalTreeCount + numTreesTotalbuf[i];
						}
						newForest->trees.reserve(totalTreeCount);

					std::vector<std::vector<size_t> > child_nodeIDsNew;

					std::cout << "new Child nodes created\n";
					std::vector<size_t> split_varIDsNew;
					std::vector<double> split_valuesNew;
					std::cout << "New split var and values create\n";
					//Iterate through allLocations to get index locations in Forests
					//Pattern: TreeLocation, split_varIDsLocation, split_valuesLocation, repeat
					//0,3,6,9 etc are treeLocations
					//1,4,7,10 etc are split_varIDsLocations
					//2,5,8,11 etc are split_valuesLocations
					std::cout << "Start rebuilding Forest at rank 0\n";
					//for (int i = 0; i < allLocations->size()-1; i++){
					int allLocCount = 0;
					int numTreesCount = 0;
					while (allLocCount < (locationSize-2)){
						std::vector<size_t> newColumn;
						child_nodeIDsNew.push_back(newColumn);
						child_nodeIDsNew.push_back(newColumn);

						//Grab the childNodes
						int x = 0;
						while (x < numNodesTotal[numTreesCount]){
							child_nodeIDsNew[0].push_back(forests[allLocationsbuf[allLocCount]+x]);
							x++;
						}
						int y = 0;
						while (y < numNodesTotal[numTreesCount]){
							child_nodeIDsNew[1].push_back(forests[allLocationsbuf[allLocCount]+x + y]);
							y++;
						}
						numTreesCount++;
						allLocCount++;
						//Grab the split_varIDs
						int k = 0;
						int diff = allLocationsbuf[allLocCount+1] - allLocationsbuf[allLocCount];
						while (k < (diff)){
							split_varIDsNew.push_back((size_t) forests[allLocationsbuf[allLocCount]+k]);
							k++;
						}
						allLocCount++;
						//Grab the split_values
						int m = 0;
						while (m < (diff)){
							split_valuesNew.push_back((double) forests[allLocationsbuf[allLocCount]+m]);
							m++;
							std::cout << std::flush;
						}
						allLocCount++;

					//Save childNodes, splitVars, and splitValues as new tree in Node 0's forest

					std::cout << std::flush;
					switch (arg_handler.treetype) {
						case TREE_CLASSIFICATION:
							if (arg_handler.probability) {

							}
							else {
									newForest->trees.push_back(new TreeClassification(child_nodeIDsNew, split_varIDsNew, split_valuesNew, &classVals, &respClassIDs));
							}
							break;

						case TREE_REGRESSION:
								newForest->trees.push_back(new TreeRegression(child_nodeIDsNew, split_varIDsNew, split_valuesNew));
							break;
						// case TREE_SURVIVAL:
						//
						// 	break;
						// case TREE_PROBABILITY:
						//
						// 	break;


					}
					//Reset variables to empty before next loop
					child_nodeIDsNew[0].clear();
					child_nodeIDsNew[1].clear();
					child_nodeIDsNew.clear();
					split_varIDsNew.clear();
					split_valuesNew.clear();

				}




				//Average variable importances for each variable
				//std::vector<double> allVariableImportance;
				std::vector<double> varImportance(varImportanceLength,0);
				for (int i = 0; i < varImportanceLength; i++) {
					for (int j = 0; j < size; j++){
						varImportance[i] = varImportance[i] + allVariableImportance[i+j*varImportanceLength];
					}
					varImportance[i] = varImportance[i]/size;
				}
				newForest->variable_importance = varImportance;

				std::cout << "Finished rebuilding Forest at rank 0\n";




				if (arg_handler.write && loop == (numLoops - 1)) {
					forest2->saveToFile(0);
				}
		}
			std::cout << "Forest mtry var: " << forest2->getMtry() << '\n' << std::flush;
			forest2->writeOutput();
			*verbose_out << "Finished Ranger." << std::endl;



			if (rank == 0){
				newForest->verbose_out = verbose_out;
				newForest->output_prefix = arg_handler.outprefix;
				newForest->importance_mode = arg_handler.impmeasure;
				newForest->outputDirectory = arg_handler.outputDirectory;
				newForest->data = forest2->data;
				newForest->data->no_split_variables = forest2->data->no_split_variables;
				newForest->writeOutputNewForest(loop);
				newForest->writeSplitWeightsFile(arg_handler.outprefix);
			}

			if (loop < (numLoops-1)){
				delete forest2;
				if (rank == 0){
					delete newForest;
				}
			}
			if (rank == 0){
				free(numTreesTotalbuf);
				free(allLocationsbuf);
				free(numNodesTotal);
				free(allVariableImportance);
				free(forests);
			}
	}

		catch (std::exception& e) {
			if (rank == 0){
				std::cerr << "Error: " << e.what() << " Ranger will EXIT now." << std::endl;
				delete forest2;
				return -1;
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);

		endNextLoop = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsedNextLoop = endNextLoop - startNextLoop;
		time_t tt = std::chrono::system_clock::to_time_t(endNextLoop);
		*verbose_time << "Loop " << loop << " time: " << elapsedNextLoop.count() << "s." << std::endl;

	}
			//End of Final Loop


			//After last loop - create paths
			std::cout << "After last loop with rank: " << rank << '\n';
			std::cout << std::flush;



					//Create paths
			if (arg_handler.printPathfile == 0){
					int num_ind_vars;
					if (arg_handler.numIterations == 1){
						num_ind_vars = forest->getNumIndependentVariables();
					}
					else{
						num_ind_vars = forest2->getNumIndependentVariables();
					}

					std::ofstream fileOut;
					int numVars = num_ind_vars + 1;
					std::vector<std::string> varNames(numVars);
					for (int i = 0; i < (num_ind_vars+1); i++){
						varNames[i] = oldForestData->getVariableNames()[i];
					}

						std::string strRank = std::to_string(rank);
						std::string directory = arg_handler.outputDirectory;
						std::string pathFile = directory + "/" + arg_handler.outprefix + strRank + ".pathfile";
						fileOut.open(pathFile);
						std::cout << "Right after opening PathFile\n";
						std::cout << std::flush;

					std::chrono::time_point<std::chrono::system_clock> startPath, endPath;
					startPath = std::chrono::system_clock::now();
					Forest* currentForest;
					std::cout << "In main, create currentForest\n" << std::flush;
					if (arg_handler.numIterations == 1){
						currentForest = forest;
					}
					else{
						currentForest = forest2;
					}
					std::vector<double> allDepVarVals;
					double pathDepVarAvg = 0.0;
					int depVarID = oldForestData->getVariableID(arg_handler.depvarname);
					std::cout << "After getting depVarID\n" << std::flush;
					for (auto& tree : currentForest->trees){
						std::cout << "In next tree loop\n" << std::flush;
						const std::vector<double> splitVec = tree->getSplitValues();
						std::cout << "In tree traversal\n" << std::flush;
						//Find the paths
						std::vector<int> localPath;
						std::vector<int> localPathDirection;
						std::cout << "In tree traversal, after localPathDirection\n" << std::flush;
						size_t num_open_nodes = 1;
						int left_childID = 0;
						int right_childID = 0;
						std::vector<std::vector<size_t> > childNodes = tree->getChildNodeIDs();
						const std::vector<size_t> splitIDs = tree->getSplitVarIDs();
						std::cout << std::flush;


						std::vector<int> visited(childNodes[0].size());
						std::fill(visited.begin(),visited.end(),0);
						//Once the root is visited three times all nodes have been seen
						int previousNode = 0;
						int currentNode = 0;
						localPath.push_back(currentNode);
						visited[currentNode]++;
						left_childID = childNodes[0][currentNode];
						right_childID = childNodes[1][currentNode];
					   while (visited[0] < 3) {
							if (visited[currentNode] < 2) {
								//traverse left until terminal
								while (left_childID != 0 && right_childID != 0){
									previousNode = currentNode;
									currentNode = left_childID;
								  localPath.push_back(currentNode);
									localPathDirection.push_back(0);
									visited[currentNode]++;
									left_childID = childNodes[0][currentNode];
									right_childID = childNodes[1][currentNode];
								}

								//check if terminal node - assend back up tree
					    	if(left_childID == 0 && right_childID == 0){
									int numSamples = tree->sampleIDs[currentNode].size();
									oldForestData->getAllValuesTotal(allDepVarVals, tree->sampleIDs[currentNode], depVarID);
									for (int i = 0; i < numSamples; i++){
										pathDepVarAvg += allDepVarVals[i];
									}
									if (pathDepVarAvg == 0){
										pathDepVarAvg = 0;
									} else{
										//std::cout << "pathDepVarTotal: " << pathDepVarAvg << " numSamples: " << numSamples << '\n';
										pathDepVarAvg = pathDepVarAvg / numSamples;
										//std::cout << "pathDepVarAvg: " << pathDepVarAvg << '\n' << std::flush;
									}
									fileOut << numSamples << " ";
									fileOut << pathDepVarAvg << " ";
									for (int i = 0; i < (localPath.size() - 1); i++){
										fileOut << varNames[splitIDs[localPath[i]]] << " " << splitVec[localPath[i]] << " " << localPathDirection[i] << " ";
									}
									fileOut << '\n';
									fileOut << std::flush;


									// //reset pathVars
									// //reset num samples and allDepVarVals
									// //numSamples = 0;
									 allDepVarVals.clear();
									 pathDepVarAvg = 0;
								}
								//Go back up tree 1 node
								if (!localPath.empty() && localPath.size() > 1){
									//pop off the leaf
									localPath.pop_back();
									localPathDirection.pop_back();
								}
								else if (localPath.size() == 1){
									//pop off leaf but do not pop direction
									localPath.pop_back();
								}
								else if (localPath.empty()){
									//std::cout << "localPath Empty, left" << " rank:" << rank <<'\n' << std::flush;
								}
								currentNode = previousNode;
								left_childID = childNodes[0][currentNode];
								right_childID = childNodes[1][currentNode];
								if (!localPath.empty() && localPath.size() > 1){
									localPath.pop_back();
									localPathDirection.pop_back();
									previousNode = localPath.back();
								}
								else if (localPath.size() == 1){
									localPath.pop_back();
									previousNode = 0;
								}
								else if (localPath.empty()){
									//std::cout << "localPath empty, left 2\n" << std::flush;
									previousNode = 0;
								}

								visited[currentNode]++;
							}
					 		else if (visited[currentNode] == 2){
					 			//go right 1

					 			localPath.push_back(currentNode);
					 			//localPathDirection.push_back(1);
									if (localPath.size() == 1){
										//Re-added root node, do not add to direction yet
									}else{
										if (childNodes[0][previousNode] == currentNode){
											localPathDirection.push_back(0);
										}else if ( childNodes[1][previousNode] == currentNode){
											localPathDirection.push_back(1);
										}
									}

								previousNode = currentNode;
								currentNode = right_childID;
					 			localPath.push_back(currentNode);
					 			localPathDirection.push_back(1);
								visited[currentNode]++;
								left_childID = childNodes[0][currentNode];
								right_childID = childNodes[1][currentNode];

					 			if(left_childID == 0 && right_childID == 0){
				 					int numSamples = tree->sampleIDs[currentNode].size();
				 					oldForestData->getAllValuesTotal(allDepVarVals, tree->sampleIDs[currentNode], depVarID);
				   				for (int i = 0; i < numSamples; i++){
										pathDepVarAvg += allDepVarVals[i];
									}
									if (pathDepVarAvg == 0){
										pathDepVarAvg = 0;
									} else{
										pathDepVarAvg = pathDepVarAvg / numSamples;
									}
									fileOut << numSamples << " ";
									fileOut << pathDepVarAvg << " ";
									for (int i = 0; i < localPath.size() - 1; i++){
										fileOut << varNames[splitIDs[localPath[i]]] << " " << splitVec[localPath[i]] << " " << localPathDirection[i] << " ";
									}
									fileOut << '\n';
									fileOut << std::flush;
					 				allDepVarVals.clear();
					   			pathDepVarAvg = 0;

									if (!localPath.empty() && localPath.size() > 1){
										//pop off the leaf
										localPath.pop_back();
										localPathDirection.pop_back();
									}
									else if (localPath.size() == 1){
										//pop off leaf but do not pop direction
										localPath.pop_back();
									}
									else if (localPath.empty()){
										//std::cout << "localPath Empty, right" << " rank:" << rank <<'\n' << std::flush;
									}
									currentNode = previousNode;
									left_childID = childNodes[0][currentNode];
									right_childID = childNodes[1][currentNode];
									if (!localPath.empty() && localPath.size() > 1){
										localPath.pop_back();
										localPathDirection.pop_back();
										previousNode = localPath.back();
									}
									else if (localPath.size() == 1){
										localPath.pop_back();
										previousNode = 0;
									}
									else if (localPath.empty()){
										//std::cout << "localPath empty, right 2" << " rank:" << rank <<'\n' << std::flush;
										previousNode = 0;
									}
										visited[currentNode]++;
								}
					 		}

				 			else if (visited[currentNode] >= 3){
				 				//Go back up tree 1 node
				 				currentNode = previousNode;
								left_childID = childNodes[0][currentNode];
								right_childID = childNodes[1][currentNode];
								//std::cout << "Current Node (visited 3, after update) : " << currentNode << '\n';
				 				std::cout << std::flush;
								if (!localPath.empty() && localPath.size() > 1){
									localPath.pop_back();
									localPathDirection.pop_back();
									previousNode = localPath.back();
								}
								else if (localPath.size() == 1){
									localPath.pop_back();
									previousNode = 0;
								}
								else if (localPath.empty()){
									//std::cout << "localPath empty, up" << " rank:" << rank <<'\n' << std::flush;
									previousNode = 0;
								}
								visited[currentNode]++;
					 		}
					   }
					 	 localPath.clear();
						 localPathDirection.clear();
					}
					//if (arg_handler.printPathfile == 0){
						fileOut.close();



					endPath = std::chrono::system_clock::now();
					std::chrono::duration<double> elapsedPath = endPath - startPath;
					time_t tt = std::chrono::system_clock::to_time_t(endPath);
					*verbose_out << "Path creation processing time (" << ctime(&tt) <<"): " << elapsedPath.count() << "s." << std::endl;
					*verbose_time << "Pathfile creation time: " << elapsedPath.count() << "s." << std::endl;
			}

				std::cout << "Before final delete forest2 with rank: " << rank << '\n';
				std::cout << std::flush;
				if (numLoops > 1){
					delete forest2;
				}

				std::cout << "After final delete forest with rank: " << rank << '\n';
				std::cout << std::flush;
				if (rank == 0){
					delete newForest;
				}

				std::cout << "Before MPI_finalize() with rank: " << rank << '\n';
				std::cout << std::flush;

				endCode = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsedCode = endCode - startCode;

				*verbose_time << "Total run time: " << elapsedCode.count() << "s." << std::endl;

			MPI_Finalize();
			return 0;
    }











    //If not using MPI
    else {
		//ArgumentHandler arg_handler(argc, argv);
  		Forest* forest = 0;
  		try {



		// Create forest object
		switch (arg_handler.treetype) {
		case TREE_CLASSIFICATION:
		  if (arg_handler.probability) {
			forest = new ForestProbability;
		  } else {
			forest = new ForestClassification;
		  }
		  break;
		case TREE_REGRESSION:
		  forest = new ForestRegression;
		  break;
		case TREE_SURVIVAL:
		  forest = new ForestSurvival;
		  break;
		case TREE_PROBABILITY:
		  forest = new ForestProbability;
		  break;
		}

		// Verbose output to logfile if non-verbose mode
		std::ostream* verbose_out;
		std::ostream* verbose_time;
		if (arg_handler.verbose) {
		  verbose_out = &std::cout;
		} else {
			std::ofstream* logfile = new std::ofstream();
			std::string directory = arg_handler.outputDirectory;
			logfile->open(directory+ "/" + arg_handler.outprefix + ".log");
		  if (!logfile->good()) {
			throw std::runtime_error("Could not write to logfile.");
		  }
		  verbose_out = logfile;

			std::ofstream* timefile = new std::ofstream();
			timefile->open(directory+ "/" + arg_handler.outprefix + ".time");
			if (!timefile->good()) {
				throw std::runtime_error("Could not write to timefile.");
			}
			verbose_time = timefile;
		}

			std::chrono::time_point<std::chrono::system_clock> startTime, endTime;
			startTime = std::chrono::system_clock::now();

    	*verbose_out << "Starting Ranger." << std::endl;
    	forest->initCpp(arg_handler.depvarname, arg_handler.memmode, arg_handler.file, arg_handler.yfile, arg_handler.mtry, arg_handler.mtryType,
        	arg_handler.outprefix, arg_handler.ntree, verbose_out, arg_handler.seed, arg_handler.nthreads,
        	arg_handler.predict, arg_handler.impmeasure, arg_handler.targetpartitionsize, arg_handler.splitweights,
        	arg_handler.alwayssplitvars, arg_handler.statusvarname, arg_handler.replace, arg_handler.catvars,
        	arg_handler.savemem, arg_handler.splitrule, arg_handler.caseweights, arg_handler.predall, arg_handler.fraction,
        	arg_handler.alpha, arg_handler.minprop, arg_handler.holdout, arg_handler.predictiontype,
        	arg_handler.randomsplits, arg_handler.useMPI,0,0,arg_handler.outputDirectory,verbose_time);

    	forest->run(true);
    	if (arg_handler.write) {
      		forest->saveToFile(0);
    	}
    	forest->writeOutput();
    	*verbose_out << "Finished Ranger." << std::endl;

			endTime = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsedTime = endTime - startTime;
			time_t tt = std::chrono::system_clock::to_time_t(endTime);
			*verbose_time << "Total run time: " << elapsedTime.count() << "s." << std::endl;
    	delete forest;
  }
  catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << " Ranger will EXIT now." << std::endl;
    delete forest;
    return -1;
  }

  return 0;

	}
}
