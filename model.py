import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        self.learning_rate = 1
        self.numTrainingGames = 2500
        self.batch_size = 100
        self.parameters = []

        hidden_layer_1_size =  100
        hidden_layer_2_size = 50
        hidden_layer_3_size = self.num_actions
        
        hidden_layer_1 = nn.Parameter(self.state_size, hidden_layer_1_size)
        hidden_layer_2 = nn.Parameter(hidden_layer_1_size, hidden_layer_2_size)
        output_layer = nn.Parameter(hidden_layer_2_size, hidden_layer_3_size)
        bias_1 = nn.Parameter(1, hidden_layer_1_size)
        bias_2 = nn.Parameter(1, hidden_layer_2_size)
        bias_3 = nn.Parameter(1, hidden_layer_3_size)
        layers = [hidden_layer_1, bias_1, hidden_layer_2, bias_2, output_layer, bias_3]
        self.set_weights(layers)

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        predicted_q = self.run(states)
        return nn.SquareLoss(predicted_q, Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        params = iter(self.parameters)
        result = nn.Linear(states, next(params))
        result = nn.AddBias(result, next(params))
        result = nn.ReLU(result)
        result = nn.Linear(result, next(params))
        result = nn.AddBias(result, next(params))
        result = nn.ReLU(result)
        result = nn.Linear(result, next(params))
        result = nn.AddBias(result, next(params))
        return result

        

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """

        loss_node = self.get_loss(states, Q_target)
        grads = nn.gradients(loss_node, self.parameters)

        for i in range(len(self.parameters)):
            self.parameters[i].update(grads[i], -self.learning_rate)
        

