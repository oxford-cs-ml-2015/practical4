local create_model = require 'create_model'

--------------------------------------------------------------
-- SETTINGS
local opt = { nonlinearity_type = 'requ' }

-- function that numerically checks gradient of the loss:
-- f is the scalar-valued function
-- g returns the true gradient (assumes input to f is a 1d tensor)
-- returns difference, true gradient, and estimated gradient
local function checkgrad(f, g, x, eps)
  -- compute true gradient
  local grad = g(x)
  
  -- compute numeric approximations to gradient
  local eps = eps or 1e-7
  local grad_est = torch.DoubleTensor(grad:size())
  for i = 1, grad:size(1) do
    -- TODO: do something with x[i] and evaluate f twice, and put your estimate of df/dx_i into grad_est[i]
    x[i] = x[i] + eps
    ...something(s) here
    grad_est[i] = ...something here
  end

  -- computes (symmetric) relative error of gradient
  local diff = torch.norm(grad - grad_est) / torch.norm(grad + grad_est)
  return diff, grad, grad_est
end

function fakedata(n)
    local data = {}
    data.inputs = torch.randn(n, 4)                     -- random standard normal distribution for inputs
    data.targets = torch.rand(n):mul(3):add(1):floor()  -- random integers from {1,2,3}
    return data
end

---------------------------------------------------------
-- generate fake data, then do the gradient check
--
torch.manualSeed(1)
local data = fakedata(5)
local model, criterion = create_model(opt)
local parameters, gradParameters = model:getParameters()

-- returns loss(params)
local f = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  return criterion:forward(model:forward(data.inputs), data.targets)
end
-- returns dloss(params)/dparams
local g = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  gradParameters:zero()

  local outputs = model:forward(data.inputs)
  criterion:forward(outputs, data.targets)
  model:backward(data.inputs, criterion:backward(outputs, data.targets))

  return gradParameters
end

local diff = checkgrad(f, g, parameters)
print(diff)

