import torch.nn	as nn
import torch

def	weights_init(m):
	classname = m.__class__.__name__
	if	classname.find('Linear') !=	-1:
		m.weight.data.normal_(0.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)




class MLP_Dropout_G(nn.Module):
	def __init__(self,	opt):
		super(MLP_Dropout_G, self).__init__()
		self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
		self.fc2 = nn.Linear(opt.ngh,	opt.resSize)
		self.lrelu = nn.LeakyReLU(0.2, True)
		self.relu	= nn.ReLU(True)
		self.dropout = nn.Dropout(p=0.2)

		self.apply(weights_init)

	def forward(self, noise, att):
		h	= torch.cat((noise,	att), 1)
		h	= self.dropout(self.lrelu(self.fc1(h)))
		h	= self.relu(self.fc2(h))
		return h

class MLP_G(nn.Module):
	def __init__(self,	opt):
		super(MLP_G, self).__init__()
		self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
		self.fc2 = nn.Linear(opt.ngh,	opt.resSize)
		self.lrelu = nn.LeakyReLU(0.2, True)
		#self.prelu =	nn.PReLU()
		self.relu	= nn.ReLU(True)
		#self.sigmoid = nn.Sigmoid()

		self.apply(weights_init)

	def forward(self, noise, att):
		h	= torch.cat((noise,	att), 1)
		h	= self.lrelu(self.fc1(h))
		h = self.relu(self.fc2(h))
		#h	= self.sigmoid(self.fc2(h))
		return h
	
	
class MLP_Q(nn.Module):
	def __init__(self,	opt):
		super(MLP_Q, self).__init__()
		self.fc1 =	nn.Linear(opt.resSize , opt.nch)
		self.fc2 = nn.Linear(opt.nch,1)
		self.lrelu = nn.LeakyReLU(0.2, True)
		self.sigmoid = nn.Sigmoid()
			
		self.apply(weights_init)

	def forward(self, x): 
		#o	= self.logic(self.fc(x),dim=1)
		h	= self.lrelu(self.fc1(x))
		h	= self.sigmoid(self.fc2(h))
		return h

	
class MLP_C(nn.Module):
	def __init__(self,	opt):
		super(MLP_C, self).__init__()
		self.fc =	nn.Linear(opt.resSize , opt.nclass_train)
			
		self.apply(weights_init)

	def forward(self, x): 
		h	= self.fc(x)

		return h
	
	
