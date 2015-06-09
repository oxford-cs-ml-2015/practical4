# Practical 4
Machine Learning, spring 2015

In this practical, we write and test our own layer.

## Setup
Setup will be the same as last time in practical 1. Please refer to the [practical 1 repository](https://github.com/oxford-cs-ml-2015/practical1), and run the script as instructed last time.

We will use the Torch package `gnuplot` for plotting; you may use (advanced: recommended to try at home instead or if you have time afterwards) iTorch instead as the script installs it.

## Practical
See the writeup PDF for instructions. Be sure the clone the repository instead of downloading file-by-file so you don't miss any:
```bash
git clone https://github.com/oxford-cs-ml-2015/practical4
cd practical4
```

As you read the practical and implement what you are asked to, **search the lua files** for TODOs. As you finish each TODO, I suggest you remove the TODO comment, so you know you're done when no TODOs remain. This is helpful as there are more source files now.

### training
To run the training procedure in first few parts, do
```bash
th -i main.lua
```
and it will output several plots. The code's comments explain what they are, but you only need to worry about the loss curve at first. The heatmaps show the decision boundaries between two variables of your choice.
There are TODOs in this file for uncommenting once you finish ReQU.

### gradient check
To run the gradient checker (*after* you fill in the TODOs inside it, and fill in the TODOs in `requ.lua`), do
```bash
th -i gradcheck.lua
```

### Jacobian check
To run the Jacobian checker, do
```bash
th -i jacobiancheck.lua
```
It will run right off the bat, so **you may use this to test your ReQU derivative** before you try writing the gradient checker. There is a TODO in this file for the final part.

# See course page for practicals
<https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/>

