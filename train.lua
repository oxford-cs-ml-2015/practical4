require 'torch'
require 'math'
require 'nn'
require 'optim'
require 'gnuplot'
require 'requ'
create_model = require 'create_model'

local function train(opt, data)
    ------------------------------------------------------------------------
    -- create model and loss/grad evaluation function
    --
    local model, criterion = create_model(opt)
    local params, grads = model:getParameters()

    -- (re-)initialize weights
    params:uniform(-0.01, 0.01)
    if opt.nonlinearity_type == 'requ' then
        -- need to offset bias for requ/relu/etc s.t. we're at x > 0 (so dz/dx is nonzero)
        for _, lin in pairs(model:findModules('nn.Linear')) do
            lin.bias:add(0.5)
        end
    end

    -- return loss, grad
    local feval = function(x)
      if x ~= params then
        params:copy(x)
      end
      grads:zero()

      -- forward
      local outputs = model:forward(data.inputs)
      local loss = criterion:forward(outputs, data.targets)
      -- backward
      local dloss_doutput = criterion:backward(outputs, data.targets)
      model:backward(data.inputs, dloss_doutput)

      return loss, grads
    end

    ------------------------------------------------------------------------
    -- optimization loop
    --
    local losses = {}
    local optim_state = {learningRate = 1e-1}

    for i = 1, opt.training_iterations do
      local _, loss = optim.adagrad(feval, params, optim_state)
      losses[#losses + 1] = loss[1] -- append the new loss

      if i % opt.print_every == 0 then
          print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))
      end
    end

    return model, losses
end

return train

