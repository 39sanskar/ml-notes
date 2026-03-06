#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// usually in C you have function like this cos(), sin(), sqrt() by defaukt in standard c library  they all work with doubles  so they accept doubles
// if you want to work with floats you have to use this variance of the functions that have a suffix f  like cosf(), sinf(), sqrtf() etc...


// Sigmoid activation function
float sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// Training sample format: [input1, input2, expected_output]
typedef float sample[3];

// Training datasets for different logic gates
sample or_train[] = {
    {0, 0, 0}, // 0 OR 0 = 0
    {1, 0, 1}, // 1 OR 0 = 1
    {0, 1, 1}, // 0 OR 1 = 1
    {1, 1, 1}, // 1 OR 1 = 1
};

sample and_train[] = {
    {0, 0, 0}, // 0 AND 0 = 0
    {1, 0, 0}, // 1 AND 0 = 0
    {0, 1, 0}, // 0 AND 1 = 0
    {1, 1, 1}, // 1 AND 1 = 1
};

sample nand_train[] = {
    {0, 0, 1}, // 0 NAND 0 = 1
    {1, 0, 1}, // 1 NAND 0 = 1
    {0, 1, 1}, // 0 NAND 1 = 1
    {1, 1, 0}, // 1 NAND 1 = 0
};

sample xor_train[] = {
    {0, 0, 0}, // 0 XOR 0 = 0
    {1, 0, 1}, // 1 XOR 0 = 1
    {0, 1, 1}, // 0 XOR 1 = 1
    {1, 1, 0}, // 1 XOR 1 = 0
};

// Current training dataset (change this to train different gates)
sample *current_train_data = and_train;
size_t train_count = 4;

// Cost function: Mean Squared Error
float calculate_cost(float w1, float w2, float bias)
{
    float total_error = 0.0f;

    for (size_t i = 0; i < train_count; ++i)
    {
        float input1 = current_train_data[i][0];
        float input2 = current_train_data[i][1];
        float expected = current_train_data[i][2];

        // Forward pass through single neuron
        float prediction = sigmoidf(input1 * w1 + input2 * w2 + bias);

        // Calculate squared error
        float error = prediction - expected;
        total_error += error * error;
    }

    return total_error / train_count;
}

// Compute gradients using finite difference method
void compute_gradients_finite_diff(float epsilon,
                                   float w1, float w2, float bias,
                                   float *dw1, float *dw2, float *dbias)
{
    float current_cost = calculate_cost(w1, w2, bias);

    // Partial derivative with respect to w1
    *dw1 = (calculate_cost(w1 + epsilon, w2, bias) - current_cost) / epsilon;

    // Partial derivative with respect to w2
    *dw2 = (calculate_cost(w1, w2 + epsilon, bias) - current_cost) / epsilon;

    // Partial derivative with respect to bias
    *dbias = (calculate_cost(w1, w2, bias + epsilon) - current_cost) / epsilon;
}

// Compute gradients using analytical gradient calculation
void compute_gradients_analytical(float w1, float w2, float bias,
                                  float *dw1, float *dw2, float *dbias)
{
    *dw1 = 0.0f;
    *dw2 = 0.0f;
    *dbias = 0.0f;

    for (size_t i = 0; i < train_count; ++i)
    {
        float input1 = current_train_data[i][0];
        float input2 = current_train_data[i][1];
        float expected = current_train_data[i][2];

        // Forward pass
        float z = input1 * w1 + input2 * w2 + bias;
        float prediction = sigmoidf(z);

        // Derivative of cost with respect to prediction
        float dcost_dpred = 2.0f * (prediction - expected);

        // Derivative of sigmoid: sigmoid(z) * (1 - sigmoid(z))
        float dsigmoid_dz = prediction * (1.0f - prediction);

        // Chain rule: dcost/dz = dcost/dpred * dpred/dz
        float dcost_dz = dcost_dpred * dsigmoid_dz;

        // Accumulate gradients
        *dw1 += dcost_dz * input1;
        *dw2 += dcost_dz * input2;
        *dbias += dcost_dz;
    }

    // Average over all training samples
    *dw1 /= train_count;
    *dw2 /= train_count;
    *dbias /= train_count;
}

// Generate random float between 0 and 1
float rand_float(void)
{
    return (float)rand() / (float)RAND_MAX;
}

// Demonstrate XOR using bitwise operations
int demonstrate_xor_bitwise(void)
{
    printf("=== XOR Truth Table (Bitwise) ===\n");
    printf("Formula: (x | y) & ~(x & y)\n");

    for (size_t x = 0; x < 2; ++x)
    {
        for (size_t y = 0; y < 2; ++y)
        {
            size_t result = (x | y) & (~(x & y));
            printf("%zu XOR %zu = %zu\n", x, y, result);
        }
    }
    printf("\n");

    return 0;
}

int main(void)
{
    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Initialize weights and bias randomly
    float w1 = rand_float();
    float w2 = rand_float();
    float bias = rand_float();

    // Hyperparameters
    float learning_rate = 1e-1;
    size_t training_iterations = 10000;

    printf("Training Single Neuron...\n");
    printf("Initial cost: %.6f\n", calculate_cost(w1, w2, bias));
    printf("Initial weights: w1=%.4f, w2=%.4f, bias=%.4f\n\n", w1, w2, bias);

    // Training loop
    for (size_t iteration = 0; iteration < training_iterations; ++iteration)
    {
        float current_cost = calculate_cost(w1, w2, bias);

        // Print progress periodically
        if (iteration % 2000 == 0)
        {
            printf("Iteration %zu, Cost: %.6f\n", iteration, current_cost);
        }

        float dw1, dw2, dbias;

#if 1
        // Use finite difference method for gradients
        float epsilon = 1e-1;
        compute_gradients_finite_diff(epsilon, w1, w2, bias, &dw1, &dw2, &dbias);
#else
        // Use analytical gradient calculation
        compute_gradients_analytical(w1, w2, bias, &dw1, &dw2, &dbias);
#endif

        // Update parameters using gradient descent
        w1 -= learning_rate * dw1;
        w2 -= learning_rate * dw2;
        bias -= learning_rate * dbias;
    }

    // Final results
    float final_cost = calculate_cost(w1, w2, bias);
    printf("\n=== Training Complete ===\n");
    printf("Final cost: %.6f\n", final_cost);
    printf("Final weights: w1=%.4f, w2=%.4f, bias=%.4f\n\n", w1, w2, bias);

    // Test the trained neuron
    printf("=== Trained Neuron Results ===\n");
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 2; ++j)
        {
            float result = sigmoidf(i * w1 + j * w2 + bias);
            printf("Input (%zu, %zu) -> Output: %.4f\n", i, j, result);
        }
    }

    printf("\n");

    // Demonstrate XOR using bitwise operations
    demonstrate_xor_bitwise();

    return 0;
}

