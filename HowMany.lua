require 'distributions'
require 'nngraph'
require 'optim'
require 'pprint'
require 'gnuplot'
task_dim = 3
token_dim = 60
speak_dim = token_dim+1
motor_dim = 3
hid_dim = 50
act_dim = speak_dim+motor_dim
input1 = nn.Identity()()
input2 = nn.Identity()()
task = nn.Identity()()
hid = nn.ReLU()(nn.Linear(token_dim+speak_dim+task_dim,hid_dim)(nn.JoinTable(2){input1,input2,task}))
value = nn.Linear(hid_dim,1)(hid)
act1 = nn.SoftMax()(nn.Linear(hid_dim,motor_dim)(hid))
act2 = nn.SoftMax()(nn.Linear(hid_dim,speak_dim)(hid))
network = nn.gModule({input1,input2,task},{value,act1,act2})
w,dw = network:getParameters()
local mse_crit = nn.MSECriterion()
next_num = 1
tokens = 0
epsilon = .3
net_reward = 0
squash = nn.Sigmoid()
softmax = nn.SoftMax()
timer = torch.Timer()

mb_dim = (token_dim+1)*(speak_dim)
data = {torch.zeros(mb_dim,token_dim),torch.zeros(mb_dim,speak_dim),torch.zeros(mb_dim,task_dim)}
goal = torch.zeros(task_dim,mb_dim,motor_dim+speak_dim)
train_pos = torch.zeros(mb_dim,1):byte()
test_pos = torch.zeros(mb_dim,1):byte()
for i=1,speak_dim do --each last said config
    if i > 1 then
        data[2][{{(i-1)*speak_dim+1,i*speak_dim},{i-1}}] = 1
    end 
    goal[2][{{(i-1)*speak_dim+1,i*speak_dim},{motor_dim+i}}] = 1 --say next
    goal[3][{{(i-1)*speak_dim+1,i*speak_dim},{motor_dim+i}}] = 1 --say next
    
    goal[1][(i-1)*speak_dim+1][2] = 1 --no tokens, do nothing
    goal[3][(i-1)*speak_dim+1][2] = 1 --no tokens, do nothing
    goal[3][(i-1)*speak_dim+1][-1] = 1 --no tokens, say done

    if i <= 60 then
        if i <= 20 then
            train_pos[(i-1)*speak_dim+1] = 1
        end
        test_pos[(i-1)*speak_dim+1] = 1
    end
    for j=2,(token_dim+1) do --each token config
        data[1][(i-1)*speak_dim+j][{{1,j-1}}] = 1
        goal[1][(i-1)*speak_dim+j][1] = 1 --take away another
        goal[3][(i-1)*speak_dim+j][1] = 1 --take away another
        if i + (j-1) <= 60 then
            if i + (j-1) <= 20 then
                train_pos[(i-1)*speak_dim+j] = 1
            end
            --test_pos[(i-1)*speak_dim+j] = 1
        end
    end
end
print('total train set:',mb_dim,'how many train:', train_pos:sum(),'how many test:', test_pos:sum(), 'how many total overfit:',train_pos:sum()/test_pos:sum())
local how_many_scale = ((mb_dim-train_pos:sum())/2)/(train_pos:sum())
test_data = {data[1][test_pos:expandAs(data[1])]:reshape(test_pos:sum(),token_dim),
            data[2][test_pos:expandAs(data[2])]:reshape(test_pos:sum(),speak_dim),
            data[3][test_pos:expandAs(data[3])]:reshape(test_pos:sum(),task_dim)}

numbers = torch.range(1,speak_dim):repeatTensor(mb_dim,1)
not_train_pos = train_pos:eq(0)
local opfunc = function(x)
    if x ~= w then
        w:copy(x)
    end
    network:zeroGradParameters()
    
    task = torch.ones(mb_dim,1):mul(2)
    --TODO: sample in a non-shit way
    --[[if you can be task 3, you are
    task[{{}}] = distributions.cat.rnd(mb_dim,torch.ones(task_dim-1))
    task[train_pos] = 3
    --]]
    --random, then repick if a non-train got task 3
    task[{{}}] = distributions.cat.rnd(mb_dim,torch.ones(task_dim))
    local redo = task[not_train_pos]:eq(3)
    task[not_train_pos][redo] = distributions.cat.rnd(redo:sum(),torch.ones(task_dim-1)):double()
    --
    local range_mask = torch.range(1,task_dim):repeatTensor(mb_dim,1):double()
    data[3] = range_mask:eq(task:expandAs(range_mask)):double()
    
    
    v,out1,out2 = unpack(network:forward(data))

    mask1,mask2 = torch.zeros(mb_dim,motor_dim):byte(),torch.zeros(mb_dim,speak_dim):byte()
    
    mask1[numbers[{{},{1,motor_dim}}]:eq(torch.multinomial(out1,1):repeatTensor(1,motor_dim):double())] = 1
    mask2[numbers:eq(torch.multinomial(out2,1):repeatTensor(1,speak_dim):double())] = 1


    action = mask1:cat(mask2):double()

    r = torch.zeros(mb_dim,1)
    --task 1
    goal_mask = action[{{},{1,motor_dim}}]:eq(goal[{1,{},{1,motor_dim}}]):prod(2)
    r[task:eq(1):cmul(goal_mask)] = 1
    --task 2
    goal_mask = action[{{},{motor_dim+1,-1}}]:eq(goal[{2,{},{motor_dim+1,-1}}]):prod(2)
    r[task:eq(2):cmul(goal_mask)] = 1
    --task 3
    goal_mask = action:eq(goal[3]):prod(2)
    r[task:eq(3):cmul(goal_mask)] = 1
    --[[
    if task == 1 then
        r[action[{{},{1,motor_dim}}]:eq(goal[{task,{},{1,motor_dim}}]):prod(2)] = 1
    elseif task == 2 then
        r[action[{{},{motor_dim+1,-1}}]:eq(goal[{task,{},{motor_dim+1,-1}}]):prod(2)] = 1
    elseif task == 3 then
        r[action:eq(goal[task]):prod(2)] = 1
    else
        error(0)
    end
    --]]
    net_reward = net_reward + r:sum()
    q = v+torch.cmul(out1,mask1:double()):sum(2)+torch.cmul(out2,mask2:double()):sum(2)
    target = r
    loss = mse_crit:forward(q,target)
    grad = mse_crit:backward(q,target)
    --scale task 3 to be an important
    grad[train_pos]:mul(how_many_scale)
    grad1 = grad:expand(mb_dim,motor_dim):clone() --:zero() --screw motor learning!
    grad2 = grad:expand(mb_dim,speak_dim):clone()
    grad1[mask1:eq(0)] = 0 
    grad2[mask2:eq(0)] = 0 
    network:backward(data,{grad,grad1,grad2})
    return loss,dw
end
config = {
    learningRate  = 1e-2
    }
lr = torch.linspace(1e-4,1e-2,1e6)
local cumloss = 0
local refresh = 5e2
for i=1,1e6 do
    --config.learningRate = lr[1e6-i+1]
    x,batchloss = optim.adam(opfunc,w,config)
    cumloss = cumloss + batchloss[1]
    if i % refresh == 0 then
        --print(out1)
        gnuplot.imagesc(out2)
        gnuplot.plotflush()
        --testing------------------------------
        --set goal
        task[{{}}] = 1
        task[test_pos] = 3
        local range_mask = torch.range(1,task_dim):repeatTensor(mb_dim,1):double()
        data[3] = range_mask:eq(task:expandAs(range_mask)):double()
        --compute output
        v,out1,out2 = unpack(network:forward(data))
        --sample actions
        mask1,mask2 = torch.zeros(mb_dim,motor_dim):byte(),torch.zeros(mb_dim,speak_dim):byte()
        mask1[numbers[{{},{1,motor_dim}}]:eq(torch.multinomial(out1,1):repeatTensor(1,motor_dim):double())] = 1
        mask2[numbers:eq(torch.multinomial(out2,1):repeatTensor(1,speak_dim):double())] = 1
        action = mask1:cat(mask2):double()
        --get reward
        goal_mask = action:eq(goal[3]):prod(2)
        r = torch.zeros(mb_dim,1)
        r[task:eq(3):cmul(goal_mask)] = 1
        ---------------------------------------
        print(i,net_reward/mb_dim/refresh,r[train_pos]:sum()/train_pos:sum(),r[test_pos]:sum()/test_pos:sum(),cumloss,w:norm(),dw:norm(),timer:time().real)
        timer:reset()
        if net_reward/mb_dim/refresh > .99 then
            break
        end
        --pprint(mask2:double():cat(goal,2))
        net_reward = 0
        cumloss = 0
    end
end
    

--]]



