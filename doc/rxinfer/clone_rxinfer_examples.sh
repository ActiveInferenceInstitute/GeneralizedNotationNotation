#!/bin/bash

# Clone the RxInferExamples.jl repository
git clone https://github.com/docxology/RxInferExamples.jl.git

echo "RxInferExamples.jl repository cloned successfully."

# Change directory to the cloned repository
cd RxInferExamples.jl

# Run the setup.jl script
echo "Running setup script..."
julia support/setup.jl

echo "Setup completed." 