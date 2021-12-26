#pragma once

#include <vector>
using namespace System;

namespace NNLib21 {
	public class NeuralNetwork
	{
		class Node
		{
			friend class NeuralNetwork;
			double Error;
		public:
			double Output;
			Node() {
				Output = 0;
				Error = 0;
			}
		};
		std::vector<std::vector<Node>*> Layers;//Neuron layers
		std::vector<std::vector<double>*> Weights;//Weights of neuron threads 
		double CoefficientOfStudy;//read this name again
		double EqualFunction(double);//education function
		void SetInput(double* InputData, int NumberOfInputData);//set input
		void LayerInputStep(std::vector<Node>* PreviousLayer, \
			std::vector<Node>* NextLayer, \
			std::vector<double>* PNWeights);//one step from layer to layerф
		void LayerInputWalk();//Go through all layers of neuron
		void SetOutputErrors(double* IdleOutput);//set errors of output
		void LayerErrorStep(std::vector<Node>* RPrevious, \
			std::vector<Node>* RNext, \
			std::vector<double>* RWeights);
		void LayerErrorWalk();//Reverse go through all layers
		void LayerWeightStep(std::vector<Node>* PreviousLayer, \
			std::vector<Node>* NextLayer, \
			std::vector<double>* PNWeights);//Calculate new weight on one step of layers
		void LayerWeightWalk();//Go through all layers and calculate new weights
	public:
		NeuralNetwork(int NumberOfInput, \
			int NumberOfHiddenLayers, \
			int NumberOfOutput, \
			int TypeOfNetwork, \
			double CoefficientOfPacking, \
			double CoefficientOfStudy);/*0 - triangel; 1 - rhombus, 2 - line*/
		NeuralNetwork(int NumberOfLayers, int* NumbersOfNueron, double CoefficientOfStudy);
		NeuralNetwork(char* FileName);//Create studied network
		~NeuralNetwork();
		void Study(double* Data, double* Idle, int NumberOfInputData);//Study proccess
		double AverageErr();//Average Error for all outputs
		double MaxError();//retrun maximum error value
		void Work(double* Data, int NumberOfInputData);//work after study, just work
		void Save(char* FileName);//Save network in file
		std::vector<Node>* LayOut;//Output
	};
}
