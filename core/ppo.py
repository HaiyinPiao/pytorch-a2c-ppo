import torch
from torch.autograd import Variable

from logger import Logger

# Set the logger
logger = Logger('./logs') # dive in later
step=0



def to_np(x): # from tensor to numpy
    return x.data.cpu().numpy()

def to_var(x): # from tensor to Variable
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages, fixed_log_probs, lr_mult, lr, clip_epsilon, l2_reg):

    optimizer_policy.lr = lr * lr_mult
    optimizer_value.lr = lr * lr_mult
    clip_epsilon = clip_epsilon * lr_mult

    """update critic"""
    values_target = Variable(returns)
    for _ in range(optim_value_iternum):
        values_pred = value_net(Variable(states))
        value_loss = (values_pred - values_target).pow(2).mean()
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

    """update policy"""
    advantages_var = Variable(advantages)
    # log_probs = policy_net.get_log_prob(Variable(states), Variable(actions))
    log_probs, entropy = policy_net.get_log_prob_entropy(Variable(states), Variable(actions))
    ratio = torch.exp(log_probs - Variable(fixed_log_probs))
    surr1 = ratio * advantages_var
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_var
    policy_surr = -torch.min(surr1, surr2).mean()
    entropy = torch.exp(entropy)
    policy_surr -= entropy
    optimizer_policy.zero_grad()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
    optimizer_policy.step()

    print("value loss:", value_loss.data[0])
    print("policy loss:", policy_surr.data[0])

    # for tag, value in value_net.named_parameters():
    #     tag = tag.replace('.', '/')
    #     print(tag+'/grad', to_np(value.grad))# from Variable to np.array

    global step

    if step%20==0:
        #============ TensorBoard logging ============#
        # (1) Log the scalar values
        info = {
            'value_loss': value_loss.data[0], # scalar
            'policy_surr': policy_surr.data[0] # scalar
        }


        for tag, value in info.items():
            logger.scalar_summary(tag, value, step+1)


        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in policy_net.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), step+1) # from Parameter to np.array
            logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)# from Variable to np.array

        for tag, value in value_net.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), step+1) # from Parameter to np.array
            logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)# from Variable to np.array

    step+=1
