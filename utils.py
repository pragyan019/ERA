from torchvision import transforms
import torch
from tqdm import tqdm

#CODE BLOCK: 3
# Train data transformations
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),  # Randomly applies CenterCrop transformation with a probability of 0.1
    transforms.Resize((28, 28)),  # Resizes the image to the size (28, 28)
    transforms.RandomRotation((-15., 15.), fill=0),  # Randomly rotates the image between -15 to 15 degrees, filling the empty space with 0
    transforms.ToTensor(),  # Converts the image to a tensor
    transforms.Normalize((0.1307,), (0.3081,)),  # Normalizes the image tensor with mean 0.1307 and standard deviation 0.3081
])

# Test data transformations
test_transforms = transforms.Compose([
    transforms.ToTensor(),  # Converts the image to a tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalizes the image tensor with mean 0.1307 and standard deviation 0.3081
])



#CODE BLOCK: 8


def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = F.nll_loss(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
     
