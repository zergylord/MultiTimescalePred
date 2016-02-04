--ProFi = require 'ProFi'
--ProFi:start()
require 'Clockwork'
require 'nngraph'
require 'optim'
require 'gnuplot'
util = require 'util.model_utils'
--target = torch.linspace(-1,1,50):cat(torch.linspace(-1,1,50))
--target = torch.linspace(0,1,10)
target = torch.load('music.t7')
--target:add(-.5):mul(2)
in_pool = nn.Identity()()
rec_pool = nn.Identity()()
num_hid = 49
--
cw = nn.Clockwork(1,num_hid,7)
layer = cw{in_pool,rec_pool}:annotate{name='clock'}
--]]
--layer = nn.Tanh()(nn.Linear(num_hid+1,num_hid)(nn.JoinTable(1){in_pool,rec_pool}))
local temp= nn.Linear(num_hid,1)
--temp.weight:normal(0,.1)
out_pool = temp(layer)
n_step = nn.Linear(num_hid,1)
--TODO: tie weights
local mask = nn.Identity()()
masked_hid = nn.CMulTable(){mask,layer}
n_step_pool = n_step(masked_hid) 

network = nn.gModule({in_pool,rec_pool,mask},{out_pool,layer,n_step_pool})
parameters, gradients = network:getParameters()
network:zeroGradParameters()
timer = torch.Timer()
max_steps = target:size()[1]
local net_clones = util.clone_many_times(network,max_steps)
for i,node in ipairs(network.forwardnodes) do
    if node.data.annotations.name == 'clock' then
        clock_node_ind = i
        break
    end
end

if clock_node_ind then
    print('setting clocks')
    for i=1,max_steps do
        net_clones[i].forwardnodes[clock_node_ind].data.module:setTime(i-1)
    end
end

local mse_crit = nn.MSECriterion()
local opfunc = function(x)
    if x ~= parameters then
        parameters:copy(x)
    end
    network:zeroGradParameters()
    data = {}
    output = torch.zeros(max_steps)
    n_out = torch.zeros(max_steps)
    rec = torch.zeros(num_hid)
    for t = 1,max_steps do
        if t > 1 then
            data[t] = {target[{{t-1}}],rec:clone(),torch.ones(num_hid)}
        else
            data[t] = {torch.zeros(1),rec:clone(),torch.ones(num_hid)}
        end
        output[t],rec,n_out[t] = unpack(net_clones[t]:forward(data[t]))
    end
    local loss = 0
    local prev_grad = torch.zeros(num_hid)
    for t = max_steps,1,-1 do
        loss = loss + mse_crit:forward(output[{{t}}],target[{{t}}])
        local grad = mse_crit:backward(output[{{t}}],target[{{t}}]):clone()
        local n_grad
        if t <= max_steps-8 then
            loss = loss + mse_crit:forward(n_out[{{t}}],target[{{t+8}}])
            n_grad = mse_crit:backward(n_out[{{t}}],target[{{t+8}}]):clone()
        else
            n_grad = torch.zeros(1)
        end
        --TODO: add n step loss
        _,prev_grad = unpack(net_clones[t]:backward(data[t],{grad,prev_grad,n_grad}))
    end
    return loss,gradients
end
config = {
    learningRate = 3e-4, momentum = .95, nesterov = true, dampening = 0
}
local cumloss = 0
for i = 1,1e5 do
    x, batchloss = optim.sgd(opfunc, parameters, config)
    --[[
    if i == 10 then
        os.exit()
    end
    --]]
    cumloss = cumloss + batchloss[1]
    --print(gradients)
    --print(net_clones[1].forwardnodes[clock_node_ind].data.module.net:parameters()[3])
    if i % 1e3 == 0 then
        print(i,cumloss,parameters:norm(),gradients:norm(),timer:time().real)
        timer:reset()
        gnuplot.plot({target},{output},{n_out[1]},{n_out[2]},{n_out[3]})
        cumloss = 0
        collectgarbage()
    end
end
--ProFi:stop()
--ProFi:writeReport('train_report.txt')
