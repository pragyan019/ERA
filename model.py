import torch.nn as nn
import torch.nn.functional as F

#CODE BLOCK: 7
class Net(nn.Module):
    """
    This class represents a neural network model.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer with input channels=1 and output channels=32.
        conv2 (nn.Conv2d): Second convolutional layer with input channels=32 and output channels=64.
        conv3 (nn.Conv2d): Third convolutional layer with input channels=64 and output channels=128.
        conv4 (nn.Conv2d): Fourth convolutional layer with input channels=128 and output channels=256.
        fc1 (nn.Linear): First fully connected layer with input features=4096 and output features=50.
        fc2 (nn.Linear): Second fully connected layer with input features=50 and output features=10.

    Methods:
        __init__(): Initializes the Net class by defining and initializing the layers of the neural network.
        forward(x): Performs the forward pass of the neural network model on the input tensor x and returns the output tensor.
    """
    def __init__(self):
        """
        Initializes the Net class by defining and initializing the layers of the neural network.

        Parameters:
            None
        
        Returns:
            None
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Performs the forward pass of the neural network model on the input tensor x and returns the output tensor.

        Parameters:
            x (torch.Tensor): Input tensor to the neural network model.
        
        Returns:
            torch.Tensor: Output tensor from the neural network model.
        """
        x = F.relu(self.conv1(x)) 
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) 
        x = F.relu(self.conv3(x)) 
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 256*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
