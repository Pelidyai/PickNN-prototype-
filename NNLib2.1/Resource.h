//{{NO_DEPENDENCIES}}
// Включаемый файл, созданный в Microsoft Visual C++.
// Используется app.rc
#include <math.h>
#include "pch.h"
#include <fstream>

using namespace std;
using namespace NNLib21;

NeuralNetwork::NeuralNetwork(int Inputs, int NLayers, int Outputs, int Type, double Coef, double CoS)
{
	if (Inputs < 1)
		throw std::exception("Exception: Input is NULL.");
	if (Outputs < 1)
		throw std::exception("Exception: Output is NULL.");
	int NNodes = 0;
	switch (Type)
	{
	case 0:
		//triangel
		NNodes = int(Inputs / Coef);
		break;
	case 1:
		//rhombus
		NNodes = Inputs * 2;
		break;
	case 2:
		//line
		NNodes = Inputs;
		break;
	}
	if (NNodes < 2)
		NNodes = 2;
	std::vector<Node>* BufVector = new std::vector<Node>();
	for (int j = 0; j < Inputs; j++)
	{//Create input nodes
		BufVector->push_back(Node());
	}
	Layers.push_back(BufVector);
	int i = 1;
	for (; i < NLayers + 1; i++, NNodes = int(NNodes / Coef))
	{//Create hidden nodes
		if (NNodes <= 1)
		{
			//NLayers = i - 1;
			//break;
			NNodes = 2;
		}
		BufVector = new std::vector<Node>();
		for (int j = 0; j < NNodes; j++)
		{
			BufVector->push_back(Node());
		}
		Layers.push_back(BufVector);
	}
	BufVector = new std::vector<Node>();
	for (int j = 0; j < Outputs; j++)
	{//Create output nodes
		BufVector->push_back(Node());
	}
	Layers.push_back(BufVector);
	LayOut = BufVector;
	std::vector<double>* BufVector2;
	for (int j = 0; j < NLayers + 1; j++)
	{//Create and set default weights
		int CountOfEdges = (int)Layers[j]->size() * (int)Layers[(size_t)j + 1]->size();
		BufVector2 = new std::vector<double>();
		for (int k = 0; k < CountOfEdges; k++)
		{
			double W = (double)(rand() % 100) + 1;
			BufVector2->push_back(W / 1000);
		}
		Weights.push_back(BufVector2);
	}
	CoefficientOfStudy = CoS;
}
NeuralNetwork::NeuralNetwork(int NLayers, int* Numbers, double CoS)
{
	std::vector<Node>* BufVector = NULL;
	for (int i = 0; i < NLayers; i++)
	{
		BufVector = new std::vector<Node>();
		for (int j = 0; j < Numbers[i]; j++)
			BufVector->push_back(Node());
		Layers.push_back(BufVector);
	}
	LayOut = BufVector;
	std::vector<double>* BufVector2;
	for (int j = 0; j < NLayers - 1; j++)
	{//Create and set default weights
		int CountOfEdges = (int)Layers[j]->size() * (int)Layers[(size_t)j + 1]->size();
		BufVector2 = new std::vector<double>();
		for (int k = 0; k < CountOfEdges; k++)
		{
			double W = (double)(rand() % 100) + 1;
			BufVector2->push_back(W / 1000);
		}
		Weights.push_back(BufVector2);
	}
	CoefficientOfStudy = CoS;
}
NeuralNetwork::NeuralNetwork(char* Name)
{
	ifstream File;
	File.open(Name);
	if (!File.is_open())
	{
		throw std::exception("Exception: file no open.");
	}
	File >> CoefficientOfStudy;
	int LayersSize = 0;
	File >> LayersSize;
	vector<Node>* Buf = NULL;
	for (int i = 0; i < LayersSize; i++)
	{
		int CurLaySize = 0;
		File >> CurLaySize;
		Buf = new vector<Node>();
		for (int j = 0; j < CurLaySize; j++)
			Buf->push_back(Node());
		Layers.push_back(Buf);
	}
	LayOut = Buf;
	int WeightsCount = 0;
	File >> WeightsCount;
	vector<double>* Buf2 = NULL;
	for (int i = 0; i < WeightsCount; i++)
	{
		int WeightsSize = 0;
		File >> WeightsSize;
		Buf2 = new vector<double>();
		for (int j = 0; j < WeightsSize; j++)
		{
			double BufDoub = 0;
			File >> BufDoub;
			Buf2->push_back(BufDoub);
		}
		Weights.push_back(Buf2);
	}
	File.close();
}
NeuralNetwork::~NeuralNetwork()
{
	for (int i = 0; i < Layers.size(); i++)
		delete Layers[i];
	for (int i = 0; i < Weights.size(); i++)
		delete Weights[i];
}
double NeuralNetwork::EqualFunction(double x)
{
	return (1 / (1 + exp(-x)));
}
void NeuralNetwork::SetInput(double* InputData, int N)
{
	int k = 0;
	for (int i = 0; i < Layers[0]->size(); i++, k++)
	{
		if (k > N - 1)
			k = 0;
		Layers[0]->at(i).Output = InputData[k];
	}
}
void NeuralNetwork::LayerInputStep(std::vector<NeuralNetwork::Node>* PreviousLayer, std::vector<NeuralNetwork::Node>* NextLayer, std::vector<double>* PNWeights)
{
	for (int i = 0; i < NextLayer->size(); i++)
	{
		NextLayer->at(i).Output = 0;
		for (int j = 0; j < PreviousLayer->size(); j++)
		{
			NextLayer->at(i).Output += PreviousLayer->at(j).Output\
				* PNWeights->at(j * NextLayer->size() + i);
		}
		NextLayer->at(i).Output = EqualFunction(NextLayer->at(i).Output);
	}
}
void NeuralNetwork::SetOutputErrors(double* IdleOutput)
{
	for (int i = 0; i < LayOut->size(); i++)
	{
		LayOut->at(i).Error = IdleOutput[i] - LayOut->at(i).Output;
	}
}
void  NeuralNetwork::LayerErrorStep(std::vector<Node>* RPrevious, std::vector<Node>* RNext, std::vector<double>* RWeights)
{
	for (int j = 0; j < RNext->size(); j++)
	{
		RNext->at(j).Error = 0;
	}
	int PrevSize = (int)RPrevious->size();
	int NextSize = (int)RNext->size();
	for (int i = 0; i < PrevSize; i++)
	{
		for (int j = 0; j < NextSize; j++)
		{
			RNext->at(j).Error += RPrevious->at(i).Error \
				* RWeights->at((size_t)(j * PrevSize + i));
		}
	}
}
void NeuralNetwork::LayerInputWalk()
{
	for (int i = 0; i < Layers.size() - 1; i++)
	{
		LayerInputStep(Layers[i], Layers[(size_t)i + 1], Weights[i]);
	}
}
void NeuralNetwork::LayerErrorWalk()
{
	for (int i = (int)Layers.size() - 1; i > 1; i--)
	{
		LayerErrorStep(Layers[i], Layers[(size_t)i - 1], Weights[(size_t)i - 1]);
	}
}
void NeuralNetwork::LayerWeightStep(std::vector<NeuralNetwork::Node>* PreviousLayer, std::vector<NeuralNetwork::Node>* NextLayer, std::vector<double>* PNWeights)
{
	for (int i = 0; i < NextLayer->size(); i++)
	{
		for (int j = 0; j < PreviousLayer->size(); j++)
		{
			PNWeights->at(j * NextLayer->size() + i) += CoefficientOfStudy\
				* NextLayer->at(i).Error * NextLayer->at(i).Output\
				* (1 - NextLayer->at(i).Output)\
				* PreviousLayer->at(j).Output;
		}
	}
}
void NeuralNetwork::LayerWeightWalk()
{
	for (int i = 0; i < Layers.size() - 1; i++)
	{
		LayerWeightStep(Layers[i], Layers[(size_t)i + 1], Weights[i]);
	}
}

void NeuralNetwork::Study(double* Data, double* Idle, int N)
{
	SetInput(Data, N);
	LayerInputWalk();
	SetOutputErrors(Idle);
	LayerErrorWalk();
	LayerWeightWalk();
}
double NeuralNetwork::AverageErr()
{
	double Sum = 0;
	for (int i = 0; i < LayOut->size(); i++)
		Sum += abs(LayOut->at(i).Error);
	return Sum / LayOut->size();
}
void NeuralNetwork::Work(double* Data, int N)
{
	SetInput(Data, N);
	LayerInputWalk();
}
void NeuralNetwork::Save(char* Name)
{
	std::ofstream File;
	File.open(Name);
	if (!File.is_open())
	{
		throw std::exception("Exception: file no open.");
	}
	File << CoefficientOfStudy << endl;
	File << Layers.size() << endl;
	for (int i = 0; i < Layers.size(); i++)
		File << Layers[i]->size() << endl;
	File << Weights.size() << endl;
	for (int i = 0; i < Weights.size(); i++)
	{
		File << Weights[i]->size() << endl;
		for (int j = 0; j < Weights[i]->size(); j++)
		{
			File << Weights[i]->at(j) << endl;
		}
	}
	File.close();
}
double NeuralNetwork::MaxError()
{
	double Max = 0;
	for (int i = 0; i < LayOut->size(); i++)
		if (LayOut->at(i).Error > Max)
			Max = LayOut->at(i).Error;
	return Max;
}