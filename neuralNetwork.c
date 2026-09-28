#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// This is a simple neural network implementation in C using the C11 standard. The code defines the 
// structure of a neural network, including layers, neurons, and activation functions. It also includes functions for initializing the network, performing forward propagation, and training the network using backpropagation.

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
} 

double dSigmoid(double x) {
    return x * (1.0 - x);
}

void shuffle(int *array, size_t n){
    if(n>1){
        size_t i;
        for(i = 0; i < n - 1; i++){
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

double init_weights() {
    return ((double)rand())/((double)RAND_MAX); //random number between 0 and 1
}

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4


int main() {
    const double lr = 0.1f;

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];

    double training_inputs[numTrainingSets][numInputs] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    double training_outputs[numTrainingSets][numOutputs] = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };
   // Initialize weights between input layer and hidden layer, and between hidden layer and output layer
    for(int i = 0; i < numInputs; i++) {
        for(int j = 0; j < numHiddenNodes; j++) {
            hiddenWeights[i][j] = init_weights();
        }
    }
    // Initialize weights between hidden layer and output layer
    for(int i = 0; i < numHiddenNodes; i++) {
        
        for(int j = 0; j < numOutputs; j++) {
            outputWeights[i][j] = init_weights();
        }
    }

    // Initialize biases for output layer
    for(int i = 0; i < numOutputs; i++) {
        outputLayerBias[i] = init_weights();
    }
    // Initialize biases for hidden layer
    for(int i = 0; i < numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weights();
    }

    int trainingSetOrder[]={0,1,2,3};

    
    int numberOfEpochs = 10000;

    // Training loop
    for(int epoch = 0; epoch < numberOfEpochs; epoch++) {
        shuffle(trainingSetOrder, numTrainingSets);
        for(int x = 0; x < numTrainingSets; x++) {
            int i = trainingSetOrder[x];

            // Forward propagation


            //Compute hidden layer activations
            
            for(int j = 0; j < numHiddenNodes; j++) {
                double activation = hiddenLayerBias[j];
                //x=bias + sum(input * weight);
                for(int k = 0; k < numInputs; k++) {
                    // Compute the weighted sum of inputs for the hidden layer
                    activation += training_inputs[i][k] * hiddenWeights[k][j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }

            //Compute output layer activations
            for(int j = 0; j < numOutputs; j++) {
                double activation = outputLayerBias[j];
                //x=bias + sum(input * weight);
                for(int k = 0; k < numHiddenNodes; k++) {
                    // Compute the weighted sum of hidden layer outputs for the output layer
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }

            printf("Input: %g %g output: %g Predicted Output: %g\n", training_inputs[i][0], training_inputs[i][1], training_outputs[i][0], outputLayer[0]);



            // Backpropagation

            // Compute change in output weights

            double deltaOutput[numOutputs];
            for(int j = 0; j < numOutputs; j++) {
                double error = training_outputs[i][j] - outputLayer[j];
                deltaOutput[j] = error * dSigmoid(outputLayer[j]);
            }

            // Compute change in hidden weights
            double deltaHidden[numHiddenNodes];
            for(int j = 0; j < numHiddenNodes; j++) {
                double error = 0.0;
                for(int k = 0; k < numOutputs; k++) {
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
            } 


            //Apply change in output bias and weights
            for(int j = 0; j < numOutputs; j++) {
                
                outputLayerBias[j] += lr * deltaOutput[j];
                for(int k = 0; k < numHiddenNodes; k++) {
                    outputWeights[k][j] += lr * deltaOutput[j] * hiddenLayer[k];
                }
            }
            //Apply change in hidden bias and weights
            for(int j = 0; j < numHiddenNodes; j++) {
                hiddenLayerBias[j] += lr * deltaHidden[j];
                for(int k = 0; k < numInputs; k++) {
                    hiddenWeights[k][j] += lr * deltaHidden[j] * training_inputs[i][k];
                }
            }




        }
    }
                fputs("Final Hidden Weights: [", stdout);
            for(int j = 0; j < numInputs; j++) {
                fputs("[", stdout);
                for(int k = 0; k < numHiddenNodes; k++) {
                    printf("%g ", hiddenWeights[j][k]);
                }
                fputs("] ", stdout);
            }

            fputs("]\nFinal Output Weights: [", stdout);
            for(int j = 0; j < numHiddenNodes; j++) {   
                fputs("[", stdout);
                for(int k = 0; k < numOutputs; k++) {
                    printf("%g ", outputWeights[j][k]);
                }
                fputs("] ", stdout);
            }   

            fputs("]\nFinal Hidden Biases: [", stdout);
            for(int j = 0; j < numHiddenNodes; j++) {
                printf("%g ", hiddenLayerBias[j]);
            }
            
            fputs("]\nFinal Output Biases: [", stdout);
            for(int j = 0; j < numOutputs; j++) {
                printf("%g ", outputLayerBias[j]);
            }
            fputs("]\n", stdout);
}