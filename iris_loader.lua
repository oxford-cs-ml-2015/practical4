-- Similar to mnist-loader from last practical, loads IRIS dataset train/test split
-- has load_train function that returns table like { inputs = ... , targets = ... }, a tensor for inputs and targets.
-- has classname_to_index function that returns a table mapping class number (1, 2, 3) to the 3 flower names

require 'torch'

local loader = {}

local lookup = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}
local classname_to_index = {}
for k,v in pairs(lookup) do
    classname_to_index[v] = k
end
function loader.classname_to_index(name)
  return classname_to_index[name]
end

function loader.load_data()
  -- load
  local data = {}
  data.inputs = {}
  data.targets = {}
  data.targets_by_name = {}

  local f = torch.DiskFile("iris.data.csv", "r")
  f:quiet()

  local line =  f:readString("*l")
  while line ~= '' do
      f1, f2, f3, f4, class_name = string.match(line, '([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)')
      data.inputs[#data.inputs + 1] = {tonumber(f1), tonumber(f2), tonumber(f3), tonumber(f4)}
      data.targets[#data.targets + 1] = loader.classname_to_index(class_name)
      data.targets_by_name[#data.targets_by_name + 1] = class_name
      line = f:readString("*l")
  end

  data.inputs = torch.Tensor(data.inputs)
  data.targets = torch.Tensor(data.targets)

  -- shuffle the dataset
  local shuffled_indices = torch.randperm(data.inputs:size(1)):long()
  -- creates a shuffled *copy*, with a new storage
  data.inputs = data.inputs:index(1, shuffled_indices):squeeze()
  data.targets = data.targets:index(1, shuffled_indices):squeeze()

  print('--------------------------------')
  print('Loaded. Sizes:')
  print('inputs', data.inputs:size())
  print('targets', data.targets:size())
  print('--------------------------------')

  return data
end

return loader

