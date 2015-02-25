require 'torch'
require 'math'
local loader = require 'iris_loader'
local train = require 'train'

torch.manualSeed(1)
local data = loader.load_data()

local opt = {
  nonlinearity_type = 'sigmoid',
  training_iterations = 150, -- note: the code uses *batches*, not *minibatches*, now.
  print_every = 25,          -- how many iterations to skip between printing the loss
}

-- train sigmoid and requ versions
model_sigmoid, losses_sigmoid = train(opt, data)
-- TODO: uncomment once you implement requ
--opt.nonlinearity_type = 'requ'
--model_requ, losses_requ = train(opt, data)


--------------------------------------------------------
-- EVALUATION STUFF: YOU CAN IGNORE ALL THIS CODE
-- NOTE: though we don't have a test set, but we'll plot the two training loss curves
-- We won't know if we overfit, but we can see how flexible our model is.

-- plot
gnuplot.figure()
gnuplot.plot({'sigmoid',
  torch.range(1, #losses_sigmoid), -- x-coordinates
  torch.Tensor(losses_sigmoid),    -- y-coordinates
  '-'}
  -- TODO: uncomment when you implement requ
  -- , {'requ',
  -- torch.range(1, #losses_requ),    -- x-coordinates
  -- torch.Tensor(losses_requ),       -- y-coordinates
  -- '-'}
  )

models = { 
    --requ = model_requ,  -- TODO: uncomment once you implement requ
    sigmoid = model_sigmoid 
}
for model_name, model in pairs(models) do
  -- classification error on train set
  local log_probs = model:forward(data.inputs)
  local _, predictions = torch.max(log_probs, 2)
  print(string.format('# correct for %s:', model_name))
  print(torch.mean(torch.eq(predictions:long(), data.targets:long()):double()))

  -- classification region in one slice (cf. Figure 1 scatterplots in writeup)
  -- not pretty, but the best we can do without hacking away at gnuplot or using another library
  local f1 = 4 -- feature on first axis
  local f2 = 3 -- feature on second axis
  local size = 60  -- resolution
  local f1grid = torch.linspace(data.inputs[{{},f1}]:min(), data.inputs[{{},f1}]:max(), size)
  local f2grid = torch.linspace(data.inputs[{{},f2}]:min(), data.inputs[{{},f2}]:max(), size)
  local result = torch.Tensor(size, size)
  local input = data.inputs[1]:clone()
  for i=1,size do
    input[f1] = f1grid[i]
    for j=1,size do
      input[f2] = f2grid[j]
      result[{i,j}] = math.exp(model:forward(input)[1])
    end
  end
  result[1][1] = 0 -- ugly hack to get the right scale
  result[1][2] = 1 -- ugly hack to get the right scale
  gnuplot.figure()
  gnuplot.imagesc(result, model_name)
end

