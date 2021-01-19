using Flux
using Flux: Data.DataLoader
using Flux: onehotbatch, onecold, crossentropy, throttle
using Flux: @epochs
using Statistics
using MLDatasets
using Serialization: serialize, deserialize

# Load the Data
println("Loading data")
x_train, y_train = FashionMNIST.traindata()
x_valid, y_valid = FashionMNIST.testdata()

# Add the channel layer
x_train = Flux.unsqueeze(x_train, 3)
x_valid = Flux.unsqueeze(x_valid, 3)

# Encode labels
y_train = onehotbatch(y_train, 0:9)
y_valid = onehotbatch(y_valid, 0:9)

println("Creatinh dataset")
# Create full dataset
train_data = DataLoader(x_train, y_train, batchsize=128)

if !isfile("model.jls")
    println("Creating model")
    model = Chain(
        # 28x28 => 14x14
        Conv((5, 5), 1 => 8, pad=2, stride=2, relu),
        # 14x14 => 7x7
        Conv((3, 3), 8 => 16, pad=1, stride=2, relu),
        # 7x7 => 4x4
        Conv((3, 3), 16 => 32, pad=1, stride=2, relu),
        # 4x4 => 28x28
        Conv((3, 3), 32 => 32, pad=1, stride=2, relu),

        # Average pooling
        GlobalMeanPool(),
        flatten,

        Dense(32, 10),
        softmax,
    )
else
    println("Loading model")
    model = open(io -> deserialize(io), "model.jls")
end
names = FashionMNIST.classnames()
println("Getting first predictions")
# Getting predictions
oy = model(x_train)
# Decoding predictions
y = onecold(oy, names)
n = rand(1:size(x_train, 4))

println("Prediction of random image($n): $(y[n])")
println("Real value: $(onecold(y_train[:,n], names))")

accuracy(y,ye) = mean(onecold(y) .== onecold(ye))
loss(x,y) = Flux.crossentropy(model(x), y)

# learning rate
lr = 0.1
opt = Descent(lr)

ps = Flux.params(model)

evalcb() = loss(x_train, y_train)

number_epochs = 10
println("Loss before: $(evalcb())")
@epochs number_epochs Flux.train!(loss, ps, train_data, opt)
println("Loss after: $(evalcb())")
println("Saving model")
open(io -> serialize(io, model), "model.jls", "w")

ony = model(x_train)
ny = onecold(ony, names)
println("Prediction after train($n): $(ny[n])")
println("Accuracy: $(accuracy(model(x_valid), y_valid))")
