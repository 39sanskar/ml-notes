### To make your build.sh script executable on Linux/macOS

#### Step 1: Open terminal and run
```bash
At the top of your script: #!/bin/bash
chmod +x build.sh
```

#### Step 2: Run the script
```bash
./build.sh
```

- What this does
- chmod = change mode (permissions)
- +x = adds execute permission

#### Verify permissions
```bash
ls -l build.sh

You should see something like:
-rwxr-xr-x  build.sh
```

[Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)

[ReLU](https://en.wikipedia.org/wiki/Rectified_linear_unit)

- Sigmoid activation function for non-linear decision boundaries
- Mean Squared Error as the cost function
- Gradient descent for optimization
- Either finite difference or analytical gradients for computing derivatives
