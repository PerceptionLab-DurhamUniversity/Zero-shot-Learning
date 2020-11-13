import torch
import torch.nn	as nn
from torch.autograd	import Variable
import torch.optim as optim
import numpy as	np
import util
from sklearn.preprocessing import MinMaxScaler 
import sys
import time
import torch.nn.functional as F


class CLASSIFIER:
	# train_Y is interger 
	def __init__(self,	_train_X, _train_Y,	_test_X,_test_Y,
				 _testclasses, data_loader, _nclass, _input_dim, _cuda, 
				 _lr=0.001,	_beta1=0.5,	_nepoch=20,	_batch_size=100, pretrain_classifier='',teacher_type='seen_classes'):
		self.train_X =  _train_X 
		self.train_Y = _train_Y 
		self.test_X = _test_X
		self.test_Y = _test_Y
		self.testclasses = _testclasses

		self.test_seen_feature = data_loader.test_seen_feature
		self.test_seen_label = data_loader.test_seen_label 
		self.seenclasses = data_loader.seenclasses
		self.batch_size =	_batch_size
		self.nepoch =	_nepoch
		self.nclass =	_nclass
		self.input_dim = _input_dim
		self.cuda	= _cuda
		self.model =	LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
		self.model.apply(util.weights_init)
		#self.criterion =	nn.NLLLoss()
		self.criterion =	nn.CrossEntropyLoss()
		self.pretrain_classifier=	pretrain_classifier
		self.ntrain =	self.train_X.size()[0]

		
		
		self.input = torch.FloatTensor(_batch_size, self.input_dim) 
		self.label = torch.LongTensor(_batch_size) 
		
		self.lr =	_lr
		self.beta1 = _beta1
		#	setup optimizer
		self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

		if self.cuda:
			self.model.cuda()
			self.criterion.cuda()
			self.input =	self.input.cuda()
			self.label =	self.label.cuda()

		self.index_in_epoch =	0
		self.epochs_completed	= 0
		self.ntrain =	self.train_X.size()[0]

		if pretrain_classifier	==	'':
			self.acc	= self.fit()	
			#print('acc=%.4f'	% (self.acc))		
		else:
			#print('pretrain_classifier',pretrain_classifier)
			#self.model.load_state_dict(torch.load(pretrain_classifier))
			torch.load(pretrain_classifier)
		

	def fit(self):
		best_acc = 0
		for epoch	in range(self.nepoch):
			for i in	range(0, self.ntrain, self.batch_size):		 
				self.model.zero_grad()
				batch_input, batch_label = self.next_batch(self.batch_size)	
				self.input.copy_(batch_input)
				self.label.copy_(batch_label)
				   
				inputv = Variable(self.input)
				labelv = Variable(self.label)
				output = self.model(inputv)
				loss = self.criterion(output, labelv)
				loss.backward()
				self.optimizer.step()
			
			#print('output',output)		
			#time.sleep(10)
			print('Training	classifier	loss=	',	loss.data)
			
			acc = self.val(self.test_X, self.test_Y,self.testclasses)
			
			#acc = self.val(self.test_seen_feature, self.test_seen_label,self.seenclasses)
			print('[%d/%d]    acc:  %.4f'	% (epoch, self.nepoch, acc))
			if acc >	best_acc:
				best_acc = acc
		return best_acc 

	
	def next_batch(self, batch_size):
		start	= self.index_in_epoch
		#	shuffle	the	data at	the	first epoch
		if self.epochs_completed == 0	and	start == 0:
			perm	= torch.randperm(self.ntrain)
			self.train_X	= self.train_X[perm]
			self.train_Y	= self.train_Y[perm]
		#	the	last batch
		if start + batch_size	> self.ntrain:
			self.epochs_completed +=	1
			rest_num_examples = self.ntrain - start
			if rest_num_examples	> 0:
				X_rest_part	= self.train_X[start:self.ntrain]
				Y_rest_part	= self.train_Y[start:self.ntrain]
			# shuffle the data
			perm	= torch.randperm(self.ntrain)
			self.train_X	= self.train_X[perm]
			self.train_Y	= self.train_Y[perm]
			# start next	epoch
			start = 0
			self.index_in_epoch = batch_size	- rest_num_examples
			end = self.index_in_epoch
			X_new_part =	self.train_X[start:end]
			Y_new_part =	self.train_Y[start:end]
			if rest_num_examples	> 0:
				return torch.cat((X_rest_part, X_new_part),	0) , torch.cat((Y_rest_part, Y_new_part), 0)
			else:
				return X_new_part, Y_new_part
		else:
			self.index_in_epoch += batch_size
			end = self.index_in_epoch
			# from index	start to index end-1
			return self.train_X[start:end], self.train_Y[start:end]

	# test_label is integer 
	def val(self, test_X, test_label, target_classes):	
		start	= 0
		ntest	= test_X.size()[0]
		predicted_label =	torch.LongTensor(test_label.size())
		for i	in range(0,	ntest, self.batch_size):
			end = min(ntest,	start+self.batch_size)
			if self.cuda:
				output = F.softmax(self.model(Variable(test_X[start:end].cuda(), requires_grad=False)) )
			else:
				output = F.softmax(self.model(Variable(test_X[start:end],	requires_grad=False))	)
			_, predicted_label[start:end] = torch.max(output.data, 1)
			start = end

		acc =	self.compute_acc(util.map_label(test_label,	target_classes), predicted_label)
		#print('predicted_label',	predicted_label)
		#print('gt label', util.map_label(test_label,	target_classes))
		#print('accuracy: ',acc)
		return acc

 	
  
	def compute_per_class_acc(self, test_label, predicted_label, nclass):
		acc_per_class	= torch.FloatTensor(nclass).fill_(0)
		for i	in range(nclass):
			idx = (test_label ==	i)
			acc_per_class[i]	= torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
		return acc_per_class.mean() 

	def compute_acc(self, test_label, predicted_label):
		test_label = test_label.cuda()
		predicted_label = predicted_label.cuda()
		idx =	(test_label==predicted_label).cuda()
		total_num	= torch.tensor(test_label.size()[0]).cuda()
		#print('idx:	',idx)
		#print('acc number:  ',torch.sum(idx))
		#print('total	number:	 ',total_num ) 
		#time.sleep(4)
		acc	=	torch.sum(idx).float()/	total_num.float()
		print('acc:  ',acc)
		return acc
	

class LINEAR_LOGSOFTMAX(nn.Module):
	def __init__(self,	input_dim, nclass):
		super(LINEAR_LOGSOFTMAX, self).__init__()
		self.fc =	nn.Linear(input_dim, nclass)
		#self.logic =	nn.LogSoftmax(dim=1)
	def forward(self, x): 
		#o	= self.logic(self.fc(x),dim=1)
		#output	= F.softmax(self.fc(x),dim=1)
		output = self.fc(x)
		return output
