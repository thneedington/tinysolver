from tinygrad.tensor import Tensor
from tinygrad import nn
from tinygrad.nn.optim import Adam
import numpy as np

class TinySolver:
    def __init__(self, input_dim, output_dim):            
        # init class vars
        self.num_actions = output_dim
        self.epsilon = 1.0
        self.e_decay = 0.0001
        self.min_epsilon = 0.1
        self.gamma = 0.9
        self.params = [self.l1.weight, self.l1.bias, self.l2.weight, self.l2.bias, self.l3.weight, self.l3.bias]
        self.optimizer = Adam(self.params, lr=0.001)

        # policy (learning) network
        self.l1 = nn.Linear(input_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, self.num_actions)
    
        # target (stable) network
        self.target_l1 = nn.Linear(input_dim, 512)
        self.target_l2 = nn.Linear(512, 256)
        self.target_l3 = nn.Linear(256, self.num_actions)

        # sync the networks
        self.sync_target()


    def sync_target(self):
        '''
        copy weights from policy network to target network
        '''
        self.target_l1.weight.assign(self.l1.weight)
        self.target_l1.bias.assign(self.l1.bias)
        self.target_l2.weight.assign(self.l2.weight)
        self.target_l2.bias.assign(self.l2.bias)
        self.target_l3.weight.assign(self.l3.weight)
        self.target_l3.bias.assign(self.l3.bias)


    def __call__(self, x, target=False):
        '''
        make a forward pass through the network

        params:
            x - state tensor
            target - should the pass use target or policy network
        '''
        if target:
            x = self.target_l1(x).relu()
            x = self.target_l2(x).relu()
            return self.target_l3(x)
        else:
            x = self.l1(x).relu()
            x = self.l2(x).relu()
            return self.l3(x)


    def train_step(self, states, actions, rewards, next_states, dones):
        '''
        train on a batch of samples

        params:
            states - all relevant hand info up to this point
            actions - action taken
            rewards - reward from action taken in state
            next_states - resulting state from combo of state + action
            dones - is this the final state of the hand
        '''
        # 1. prediction: Q(s, a) from policy net 
        q_action = self.get_q_action_taken(states, actions)

        # 2. max Q(s', a') from target net
        q_next_state = self(next_states, target=True).max(axis=1)

        # 3: taget = reward + gamma * q_next_state * (1 - done)
        target = rewards + self.gamma * q_next_state * (1 - dones)

        # calc loss
        loss = (target.detach() - q_action).square().mean()

        # back propogate loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon decay
        self.epsilon = max(self.min_epsilon, self.epsilon - self.e_decay)

        return loss.item()


    def get_q_action_taken(self, states, actions):
        '''
        get the predicted q value of the action taken from the policy network
        '''
        # get all q-values for state
        all_q = self(states)
        
        # mask all non-chosen actions
        action_mask = Tensor.eye(self.num_actions)[actions]
        action_q = (all_q * action_mask).sum(axis = 1)

        return action_q


    def select_action(self, state):
        '''
        select an action - random action taken with probability e
        '''
        # potentially choose a random action
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        
        # otherwise choose best option
        else:
            state_tensor = Tensor(state).reshape(1, -1)
            q_values = self(state_tensor)
            return q_values.argmax().item()