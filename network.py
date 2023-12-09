import torch.nn as nn

#Reinforce with Baseline
class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,complexity):
        super(Policy, self).__init__()
        self.complexity=complexity
        if complexity==1:
            self.fc=nn.Linear(input_size,output_size)
        elif complexity==2:
            self.fc1 = nn.Linear(input_size, hidden_size[0])
            # self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size[0], output_size)
            self.softmax = nn.Softmax(dim=-1)
        elif complexity==3:
            self.fc1 = nn.Linear(input_size, hidden_size[0])
            self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
            self.fc3 = nn.Linear(hidden_size[0], hidden_size[1])
            self.fc4 = nn.Linear(hidden_size[0], hidden_size[1])
            self.fc5 = nn.Linear(hidden_size[1], output_size)
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        if self.complexity==1:
            return nn.functional.softmax(self.fc(state),dim=-1)
        elif self.complexity==2:
            state = self.fc1(state)
            # state = self.relu(state)
            state = self.fc2(state)
            return self.softmax(state)
        elif self.complexity==3:
            state = self.relu(self.fc1(state))
            state = self.tanh(self.fc2(state))
            state = self.tanh(self.fc3(state))
            state = self.tanh(self.fc4(state))
            state = self.fc5(state)
            return self.softmax(state)

        

class ValueFunction(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,complexity):
        super(ValueFunction, self).__init__()
        self.complexity=complexity
        if complexity==1:
            self.fc=nn.Linear(input_size,output_size)
        elif complexity==2:
            self.fc1 = nn.Linear(input_size, hidden_size[0])
            # self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size[0], output_size)
        elif complexity==3:
            self.fc1 = nn.Linear(input_size, hidden_size[0])
            self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
            self.fc3 = nn.Linear(hidden_size[0], hidden_size[1])
            self.fc4 = nn.Linear(hidden_size[0], hidden_size[1])
            self.fc5 = nn.Linear(hidden_size[1], output_size)
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()

    def forward(self, state):
        if self.complexity==1:
            return nn.functional.softmax(self.fc(state),dim=-1)
        elif self.complexity==2:
            state = self.fc1(state)
            # state = self.relu(state)
            return self.fc2(state)
        elif self.complexity==3:
            state = self.relu(self.fc1(state))
            state = self.tanh(self.fc2(state))
            state = self.tanh(self.fc3(state))
            state = self.tanh(self.fc4(state))
            return self.fc5(state)


#Actor Critic

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,complexity):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0], output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size,output_size,complexity):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0], output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
