--ODD: backpropping through the sigmoid is bad. I guess since its only used to sample?
require 'nngraph'
require 'optim'
require 'pprint'
in_dim = 1
hid_dim = 50
act_dim = 10
input = nn.Identity()()
hid = nn.ReLU()(nn.Linear(in_dim,hid_dim)(input))
act = nn.Linear(hid_dim,act_dim+1)(hid)
network = nn.gModule({input},{act})
w,dw = network:getParameters()
local mse_crit = nn.MSECriterion()
goal = torch.rand(act_dim):gt(.5):double()
epsilon = .3
net_reward = 0
squash = nn.Sigmoid()
local opfunc = function(x)
    if x ~= w then
        w:copy(x)
    end
    network:zeroGradParameters()
    data = torch.zeros(1)
    output = network:forward(data)
    mask = output:gt(0)
    mask[-1] = 1 -- bias
    --[[ random binary string
    if torch.rand(1)[1] < epsilon then
        mask[{{1,-2}}] = torch.rand(act_dim):gt(.5)
    end
    --]]
    --[[bitwise
    swap = torch.rand(act_dim):lt(epsilon)
    mask[{{1,-2}}][swap]  = mask[{{1,-2}}][swap]:eq(0)
    --]]
    --softmax
    inv_temp = 20
    mask[{{1,-2}}] = torch.rand(act_dim):lt(squash:forward(torch.mul(output[{{1,-2}}],inv_temp)))
    --]]


    action = mask[{{1,-2}}]:double()
    r = torch.zeros(1)
    if action:eq(goal):all() then
        r[1] = 1
    end
    net_reward = net_reward + r[1]
    q = torch.zeros(1)
    q[1] = output[mask]:sum()
    target = r
    loss = mse_crit:forward(q,target)
    grad = mse_crit:backward(q,target)
    grad = grad:expand(act_dim+1):clone()
    --grad[{{1,-2}}] = squash:backward(torch.mul(output[{{1,-2}}],inv_temp),grad[{{1,-2}}])
    grad[mask:eq(0)] = 0 
    network:backward(data,grad)
    return loss,dw
end
config = {
    learningRate  = 1e-2
    }
local cumloss = 0
for i=1,1e6 do
    x,batchloss = optim.adam(opfunc,w,config)
    cumloss = cumloss + batchloss[1]
    if i % 1e4 == 0 then
        print(i,net_reward,cumloss,w:norm(),dw:norm())
        pprint(action:cat(goal,2))
        net_reward = 0
        cumloss = 0
    end
end
    

--]]



