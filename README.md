# Julia CNN FashionMNIST

Welcome to a simple example of cnn training with Julia, using Flux.jl.

## Requirements

The following requirements need to be fulfilled before executing the main script:
- Flux
- MLDatasets

```Julia
using Pkg
Pkg.add("Flux")
Pkg.add("MLDatasets")
```

## Run
cd to the project folder:
```bash
julia mnist.jl
```

### Saving and loading
The script automatically saves and loads a model if available.
