#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Structure to hold all weights and biases for the XOR network
// Network architecture: OR + NAND -> AND
typedef struct {
    float or_w1;    // Weight 1 for OR neuron
    float or_w2;    // Weight 2 for OR neuron
    float or_b;     // Bias for OR neuron
    
    float nand_w1;  // Weight 1 for NAND neuron
    float nand_w2;  // Weight 2 for NAND neuron
    float nand_b;   // Bias for NAND neuron
    
    float and_w1;   // Weight 1 for final AND neuron
    float and_w2;   // Weight 2 for final AND neuron
    float and_b;    // Bias for final AND neuron
} XorNetwork;

// Sigmoid activation function
float sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// Forward propagation through the network
// Implements: (OR gate output) AND (NAND gate output) = XOR
float forward(XorNetwork net, float x1, float x2)
{
    // First layer: OR and NAND gates
    float or_output = sigmoidf(net.or_w1 * x1 + net.or_w2 * x2 + net.or_b);
    float nand_output = sigmoidf(net.nand_w1 * x1 + net.nand_w2 * x2 + net.nand_b);
    
    // Second layer: AND gate combines OR and NAND outputs
    return sigmoidf(or_output * net.and_w1 + nand_output * net.and_w2 + net.and_b);
}

// Training samples format: [input1, input2, expected_output]
typedef float sample[3];

// XOR truth table
sample xor_train[] = {
    {0, 0, 0},  // 0 XOR 0 = 0
    {1, 0, 1},  // 1 XOR 0 = 1
    {0, 1, 1},  // 0 XOR 1 = 1
    {1, 1, 0},  // 1 XOR 1 = 0
};

// Other logic gates (for reference/debugging)
sample or_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

sample and_train[] = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 1},
};

sample nand_train[] = {
    {0, 0, 1},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};

sample nor_train[] = {
    {0, 0, 1},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 0},
};

// Current training dataset
sample *train_data = xor_train;
size_t train_count = 4;

// Cost function: Mean Squared Error
float cost_function(XorNetwork net)
{
    float total_error = 0.0f;
    
    for (size_t i = 0; i < train_count; ++i) {
        float x1 = train_data[i][0];
        float x2 = train_data[i][1];
        float expected = train_data[i][2];
        
        float predicted = forward(net, x1, x2);
        float error = predicted - expected;
        
        total_error += error * error;
    }
    
    return total_error / train_count;
}

// Generate random float between 0 and 1
float rand_float(void)
{
    return (float)rand() / (float)RAND_MAX;
}

// Initialize network with random weights and biases
XorNetwork create_random_network(void)
{
    XorNetwork net;
    
    // Random initialization for OR neuron
    net.or_w1 = rand_float();
    net.or_w2 = rand_float();
    net.or_b = rand_float();
    
    // Random initialization for NAND neuron
    net.nand_w1 = rand_float();
    net.nand_w2 = rand_float();
    net.nand_b = rand_float();
    
    // Random initialization for final AND neuron
    net.and_w1 = rand_float();
    net.and_w2 = rand_float();
    net.and_b = rand_float();
    
    return net;
}

// Print all network parameters
void print_network(XorNetwork net)
{
    printf("=== Network Parameters ===\n");
    printf("OR Neuron:   w1=%.4f, w2=%.4f, b=%.4f\n", 
           net.or_w1, net.or_w2, net.or_b);
    printf("NAND Neuron: w1=%.4f, w2=%.4f, b=%.4f\n", 
           net.nand_w1, net.nand_w2, net.nand_b);
    printf("AND Neuron:  w1=%.4f, w2=%.4f, b=%.4f\n", 
           net.and_w1, net.and_w2, net.and_b);
    printf("=========================\n");
}

// Update network parameters using gradients
XorNetwork update_network(XorNetwork net, XorNetwork gradients, float learning_rate)
{
    net.or_w1 -= learning_rate * gradients.or_w1;
    net.or_w2 -= learning_rate * gradients.or_w2;
    net.or_b -= learning_rate * gradients.or_b;
    
    net.nand_w1 -= learning_rate * gradients.nand_w1;
    net.nand_w2 -= learning_rate * gradients.nand_w2;
    net.nand_b -= learning_rate * gradients.nand_b;
    
    net.and_w1 -= learning_rate * gradients.and_w1;
    net.and_w2 -= learning_rate * gradients.and_w2;
    net.and_b -= learning_rate * gradients.and_b;
    
    return net;
}

// Compute gradients using finite difference method
XorNetwork compute_gradients(XorNetwork net, float epsilon)
{
    XorNetwork gradients;
    float original_cost = cost_function(net);
    float saved_value;
    
    // Compute gradient for OR neuron weights and bias
    saved_value = net.or_w1;
    net.or_w1 += epsilon;
    gradients.or_w1 = (cost_function(net) - original_cost) / epsilon;
    net.or_w1 = saved_value;
    
    saved_value = net.or_w2;
    net.or_w2 += epsilon;
    gradients.or_w2 = (cost_function(net) - original_cost) / epsilon;
    net.or_w2 = saved_value;
    
    saved_value = net.or_b;
    net.or_b += epsilon;
    gradients.or_b = (cost_function(net) - original_cost) / epsilon;
    net.or_b = saved_value;
    
    // Compute gradient for NAND neuron weights and bias
    saved_value = net.nand_w1;
    net.nand_w1 += epsilon;
    gradients.nand_w1 = (cost_function(net) - original_cost) / epsilon;
    net.nand_w1 = saved_value;
    
    saved_value = net.nand_w2;
    net.nand_w2 += epsilon;
    gradients.nand_w2 = (cost_function(net) - original_cost) / epsilon;
    net.nand_w2 = saved_value;
    
    saved_value = net.nand_b;
    net.nand_b += epsilon;
    gradients.nand_b = (cost_function(net) - original_cost) / epsilon;
    net.nand_b = saved_value;
    
    // Compute gradient for final AND neuron weights and bias
    saved_value = net.and_w1;
    net.and_w1 += epsilon;
    gradients.and_w1 = (cost_function(net) - original_cost) / epsilon;
    net.and_w1 = saved_value;
    
    saved_value = net.and_w2;
    net.and_w2 += epsilon;
    gradients.and_w2 = (cost_function(net) - original_cost) / epsilon;
    net.and_w2 = saved_value;
    
    saved_value = net.and_b;
    net.and_b += epsilon;
    gradients.and_b = (cost_function(net) - original_cost) / epsilon;
    net.and_b = saved_value;
    
    return gradients;
}

int main(void)
{
    // Seed random number generator
    srand((unsigned int)time(NULL));
    
    // Initialize network with random weights
    XorNetwork network = create_random_network();
    
    // Hyperparameters
    float epsilon = 1e-1;      // For finite difference approximation
    float learning_rate = 1e-1; // Step size for gradient descent
    size_t iterations = 100000; // Number of training iterations
    
    printf("Training XOR Neural Network...\n");
    printf("Initial cost: %.6f\n", cost_function(network));
    
    // Training loop
    for (size_t i = 0; i < iterations; ++i) {
        XorNetwork gradients = compute_gradients(network, epsilon);
        network = update_network(network, gradients, learning_rate);
        
        // Print progress occasionally
        if (i % 20000 == 0) {
            printf("Iteration %zu, Cost: %.6f\n", i, cost_function(network));
        }
    }
    
    printf("\nFinal cost: %.6f\n", cost_function(network));
    print_network(network);
    
    // Test the trained network
    printf("\n=== XOR Results ===\n");
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            float result = forward(network, (float)i, (float)j);
            printf("%zu XOR %zu = %.4f\n", i, j, result);
        }
    }
    
    // Show intermediate neuron outputs
    printf("\n=== Intermediate Neuron Outputs ===\n");
    
    printf("\nOR Neuron Outputs:\n");
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            float output = sigmoidf(network.or_w1 * i + network.or_w2 * j + network.or_b);
            printf("%zu OR %zu = %.4f\n", i, j, output);
        }
    }
    
    printf("\nNAND Neuron Outputs:\n");
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            float output = sigmoidf(network.nand_w1 * i + network.nand_w2 * j + network.nand_b);
            printf("%zu NAND %zu = %.4f\n", i, j, output);
        }
    }
    
    return 0;
}




