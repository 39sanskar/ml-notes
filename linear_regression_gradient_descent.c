#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Training dataset: input -> output pairs
float train[][2] = {
    {0, 0},
    {1, 2},
    {3, 6},
    {4, 8},
};

#define TRAIN_COUNT (sizeof(train) / sizeof(train[0]))

// Generate random float in [0, 1]
float rand_float(void)
{
    return (float)rand() / (float)RAND_MAX;
}

// Cost function: Mean Squared Error (MSE)
// Measures how far our predictions are from actual values
float cost(float w)
{
    float total_error = 0.0f;
    for (size_t i = 0; i < TRAIN_COUNT; ++i)
    {
        float x = train[i][0];
        float predicted_y = x * w;
        float error = predicted_y - train[i][1];
        total_error += error * error;
    }
    return total_error / TRAIN_COUNT;
}

// Derivative of the cost function with respect to weight 'w'
// Used by gradient descent to update weights
float dcost(float w)
{
    float gradient = 0.0f;
    for (size_t i = 0; i < TRAIN_COUNT; ++i)
    {
        float x = train[i][0];
        float y = train[i][1];
        gradient += 2 * (x * w - y) * x;
    }
    return gradient / TRAIN_COUNT;
}

int main()
{
    // Seed RNG for reproducibility
    srand(69); 

    // Initialize weight randomly between 0 and 10
    float w = rand_float() * 10.0f;

    // Learning rate controls step size during optimization
    float learning_rate = 1e-1;

    printf("Initial cost = %f, initial w = %f\n", cost(w), w);

    // Gradient Descent Loop
    for (size_t epoch = 0; epoch < 50; ++epoch)
    {
        // Compute derivative of cost function at current w
        float dw = dcost(w);

        // Update weight using gradient descent rule
        w -= learning_rate * dw;

        // Print progress every iteration
        printf("Cost = %f, Weight (w) = %f\n", cost(w), w);
    }

    printf("------------------------------\n");
    printf("Final learned weight: w = %f\n", w);

    return 0;
}
