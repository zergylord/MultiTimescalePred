require 'nngraph'
require 'optim'
require 'pprint'
in_dim = 1
hid_dim = 50
act_dim = 10
input = nn.Identity()()
hid = nn.ReLU()(nn.Linear(in_dim,hid_dim)(input))
value = nn.Linear(hid_dim,1)(hid)
--act1 = nn.Linear(hid_dim,act_dim/2)(hid)
--act2 = nn.Linear(hid_dim,act_dim/2)(hid)
act1 = nn.SoftMax()(nn.Linear(hid_dim,act_dim/2)(hid))
act2 = nn.SoftMax()(nn.Linear(hid_dim,act_dim/2)(hid))
network = nn.gModule({input},{value,act1,act2})
w,dw = network:getParameters()
local mse_crit = nn.MSECriterion()
goal = torch.zeros(act_dim)
goal[torch.multinomial(torch.rand(act_dim/2),1)[1]] = 1
goal[act_dim/2+torch.multinomial(torch.rand(act_dim/2),1)[1]] = 1
print(goal)
epsilon = .3
net_reward = 0
squash = nn.Sigmoid()
softmax = nn.SoftMax()
local opfunc = function(x)
    if x ~= w then
        w:copy(x)
    end
    network:zeroGradParameters()
    data = torch.zeros(1)
    v,out1,out2 = unpack(network:forward(data))

    mask1,mask2 = torch.zeros(act_dim/2):byte(),torch.zeros(act_dim/2):byte()
    inv_temp = 20
    --mask1[torch.multinomial(softmax:forward(out1:mul(inv_temp)),1)[1]] = 1
    --mask2[torch.multinomial(softmax:forward(out2:mul(inv_temp)),1)[1]] = 1
    mask1[torch.multinomial(out1,1)[1]] = 1
    mask2[torch.multinomial(out2,1)[1]] = 1


    action = mask1:cat(mask2):double()
    r = torch.zeros(1)
    if action:eq(goal):all() then
        r[1] = 1
    end
    net_reward = net_reward + r[1]
    q = torch.zeros(1)
    q[1] = v+out1[mask1]:sum()+out2[mask2]:sum()
    target = r
    loss = mse_crit:forward(q,target)
    grad = mse_crit:backward(q,target)
    grad1 = grad:expand(act_dim/2):clone()
    grad2 = grad1:clone()
    grad1[mask1:eq(0)] = 0 
    grad2[mask2:eq(0)] = 0 
    network:backward(data,{grad,grad1,grad2})
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



