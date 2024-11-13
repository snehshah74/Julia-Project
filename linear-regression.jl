using Pkg
Pkg.add("Flux")
Pkg.add("DataFrames")
Pkg.add("CSV")

using Flux
using DataFrames
using CSV
Load your dataset:
# Load your dataset using CSV.jl
df = CSV.read("data.csv")
Prepare your data:
# Assuming your dataset has columns 'x' and 'y'
x_data = df.x
y_data = df.y

# Convert data to Float64 arrays
x = Float64.(x_data)
y = Float64.(y_data)
Define the model:
# Define a simple linear regression model
model = Chain(Dense(1, 1))

# Define the loss function (mean squared error)
loss(x, y) = Flux.mse(model(x), y)

# Define the optimizer (e.g., Gradient Descent)
opt = Descent(0.01)
Train the model:
# Train the model using Flux's `train!` function
dataset = [(x[i], y[i]) for i in 1:length(x)]
Flux.train!(loss, params(model), dataset, opt)
Make predictions:
# Make predictions on new data
new_x = [1.0, 2.0, 3.0]  # Example new data
predictions = model(new_x)
