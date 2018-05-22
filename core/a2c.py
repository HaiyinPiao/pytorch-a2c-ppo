import torch
from torch.autograd import Variable

# from logger import Logger

# # Set the logger
# logger = Logger('./logs') # dive in later
# step=0



def to_np(x): # from tensor to numpy
    return x.data.cpu().numpy()

def to_var(x): # from tensor to Variable
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def a2c_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, returns, advantages, l2_reg):

    policy_net.train()
    value_net.train()
    
    """update critic"""
    values_target = Variable(returns)
    values_pred = value_net(Variable(states))
    value_loss = (values_pred - values_target).pow(2).mean()

    # weight decay
    for param in value_net.parameters():
        value_loss += param.pow(2).sum() * l2_reg
    optimizer_value.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm(value_net.parameters(), 40)

    optimizer_value.step()

    """update policy"""
    # TODO
    actions = actions.view(2100)
    log_probs = policy_net.get_log_prob(Variable(states), Variable(actions))
    policy_loss = -(log_probs * Variable(advantages)).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
    # torch.nn.utils.clip_grad_norm(policy_net.parameters(), 0.01)
    optimizer_policy.step()

    print("value loss:", value_loss.data[0])
    print("policy loss:", policy_loss.data[0])


    # # for tag, value in value_net.named_parameters():
    # #     tag = tag.replace('.', '/')
    # #     print(tag+'/grad', to_np(value.grad))# from Variable to np.array

    # global step

    # if step%20==0:
    #     #============ TensorBoard logging ============#
    #     # (1) Log the scalar values
    #     info = {
    #         'value_loss': value_loss.data[0], # scalar
    #         'policy_loss': policy_loss.data[0] # scalar
    #     }


    #     for tag, value in info.items():
    #         logger.scalar_summary(tag, value, step+1)


    #     # (2) Log values and gradients of the parameters (histogram)
    #     for tag, value in policy_net.named_parameters():
    #         tag = tag.replace('.', '/')
    #         logger.histo_summary(tag, to_np(value), step+1) # from Parameter to np.array
    #         logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)# from Variable to np.array

    #     for tag, value in value_net.named_parameters():
    #         tag = tag.replace('.', '/')
    #         logger.histo_summary(tag, to_np(value), step+1) # from Parameter to np.array
    #         logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)# from Variable to np.array

    # step+=1
