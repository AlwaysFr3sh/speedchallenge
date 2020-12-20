# repeated from my notebook, 
# repeating code is bad, I will fix this later
import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2

# downscales image
def downscale(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

class SpeedNet(nn.Module): # does the name make sense?(yes)
    def __init__(self):
        # init parent
        super().__init__()
        
        # conv layers (?)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv4 = nn.Conv2d(64, 64, 5)
        self.conv5 = nn.Conv2d(64,128, 5)
        
        x = torch.randn(240, 320).view(-1, 1, 240 ,320)
        self._to_linear = None
        self.convs(x)
        
        # Linear Layer # is less linear layer worse or same or better?
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 1)
        #self.forward(x)
    
    def convs(self, x):
        # I feel like this is the equivalent of a noob spamming buttons
        # in street fighter and blundering into a win(hopefully)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv5(x)), (2,2))
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        #print(self._to_linear)
        
        return x
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear) # reshape to pass through linear biz
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
