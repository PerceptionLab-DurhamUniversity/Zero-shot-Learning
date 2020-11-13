import torch
import torch.nn	as nn
from torch.autograd	import Variable
import torch.optim as optim
import numpy as	np
import util
from sklearn.preprocessing import MinMaxScaler 
import sys

class CLASSIFIER:
	# train_Y is interger 
	def __init__(self,	_train_X, _train_Y, _test_X,_test_Y,_testseenclasses,_testunseenclasses,_test_seen_num,
				 data_loader, _nclass, _cuda,
                _lr=0.001,	_beta1=0.5,	_nepoch=20,	_batch_size=100, generalized=True):
		
		self.train_X =  _train_X 
		self.train_Y = _train_Y 
		self.test_X = _test_X
		self.test_Y = _test_Y
		self.testseenclasses = _testseenclasses
		self.testunseenclasses = _testunseenclasses
		self.test_seen_num = _test_seen_num
		
		self.test_seen_feature = data_loader.test_seen_feature
		self.test_seen_label = data_loader.test_seen_label 
		self.test_unseen_feature = data_loader.test_unseen_feature
		self.test_unseen_label = data_loader.test_unseen_label 
		self.seenclasses = data_loader.seenclasses
		self.unseenclasses = data_loader.unseenclasses
		self.batch_size =	_batch_size
		self.nepoch =	_nepoch
		self.nclass =	_nclass
		self.input_dim = _train_X.size(1)
		self.cuda	= _cuda
		self.model =	LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
		self.model.apply(util.weights_init)
		self.criterion = nn.NLLLoss()
		#self.test = test
		
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

		if generalized:
			self.acc_seen_test,self.acc_unseen_test, self.H_test,self.acc_seen, self.acc_unseen, self.H =	self.fit()
			#print('Final: acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' %	(self.acc_seen,	self.acc_unseen, self.H))
		else:
			self.acc_test,self.acc	= self.fit_zsl() 
			#print('acc=%.4f' % (self.acc))

	
	def fit_zsl(self):
		best_acc = 0
		best_acc_test=0
		mean_loss	= 0
		last_loss_epoch =	1e8	
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
				mean_loss += loss
				loss.backward()
				self.optimizer.step()
				#print('Training classifier	loss= ', loss.data[0])
			

			acc_test = self.val(self.test_X, self.test_Y,self.unseenclasses)		
			print('acc_test',acc_test)
			acc = self.val(self.test_unseen_feature, self.test_unseen_label,	self.unseenclasses)
			print('acc',acc)
			
			#print('acc %.4f' % (acc))
			if acc >	best_acc:
				best_acc = acc
			if acc_test >	best_acc_test:
				best_acc_test = acc_test				
		return best_acc_test,best_acc 

	def fit(self):
		best_H = 0
		best_seen	= 0
		best_unseen =	0
	
		best_H_test = 0
		best_seen_test	= 0
		best_unseen_test =	0	
		
		test_feat_seen = self.test_X[:self.test_seen_num]
		test_label_seen = self.test_Y[:self.test_seen_num]
		test_feat_unseen = self.test_X[self.test_seen_num:]
		test_label_unseen = self.test_Y[self.test_seen_num:]
		
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
				#print('Training classifier	loss= ', loss.data[0])\
						
			acc_seen	= 0
			acc_unseen =	0
			acc_seen_test	= 0
			acc_unseen_test =	0
			
			acc_seen_test	= self.val_gzsl(test_feat_seen,	test_label_seen, self.testseenclasses)
			acc_unseen_test =	self.val_gzsl(test_feat_unseen,	test_label_unseen,	self.testunseenclasses)		

			#print('acc_seen_test',acc_seen_test)
			#print('acc_unseen_test',acc_unseen_test)

			acc_seen	= self.val_gzsl(self.test_seen_feature,	self.test_seen_label, self.seenclasses)
			acc_unseen =	self.val_gzsl(self.test_unseen_feature,	self.test_unseen_label,	self.unseenclasses)
			#print('acc_seen',acc_seen)
			#print('acc_unseen',acc_unseen)

			if acc_seen_test ==0 and acc_unseen_test==0:
				H_test = 0
			else:
				H_test = 2*acc_seen_test*acc_unseen_test / (acc_seen_test + acc_unseen_test)
			if H_test	> best_H_test:
				best_seen_test =	acc_seen_test
				best_unseen_test	= acc_unseen_test
				best_H_test = H_test
				
			if acc_seen ==0 and acc_unseen==0:
				H = 0
			else:
				H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
			if H	> best_H:
				best_seen =	acc_seen
				best_unseen	= acc_unseen
				best_H = H	
				
			#print('acc_seen=%.4f,	acc_unseen=%.4f,   H=%.4f   ' % (acc_seen, acc_unseen, H))
		return best_seen_test,	best_unseen_test, best_H_test, best_seen,	best_unseen, best_H
					 
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
			#print(start, end)
			if rest_num_examples	> 0:
				return torch.cat((X_rest_part, X_new_part),	0) , torch.cat((Y_rest_part, Y_new_part), 0)
			else:
				return X_new_part, Y_new_part
		else:
			self.index_in_epoch += batch_size
			end = self.index_in_epoch
			#print(start, end)
			# from index	start to index end-1
			return self.train_X[start:end], self.train_Y[start:end]


	def val_gzsl(self,	test_X,	test_label,	target_classes): 
		start	= 0
		ntest	= test_X.size()[0]
		predicted_label =	torch.LongTensor(test_label.size())
		for i	in range(0,	ntest, self.batch_size):
			end = min(ntest,	start+self.batch_size)
			if self.cuda:
				output = self.model(Variable(test_X[start:end].cuda(), requires_grad=False)) 
			else:
				output = self.model(Variable(test_X[start:end],	requires_grad=False))	
			_, predicted_label[start:end] = torch.max(output.data, 1)
			start = end

		acc =	self.compute_per_class_acc_gzsl(util.map_label(test_label, target_classes), predicted_label,	target_classes.size(0))

		#acc =	self.compute_per_class_acc_gzsl(test_label,	predicted_label, target_classes)
		return acc
	
	'''
	def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
		acc_per_class	= 0
		for i	in target_classes:
			idx = (test_label ==	i)
			print(print('idx',idx))
			print('torch.sum(test_label[idx]==predicted_label[idx])',torch.sum(test_label[idx]==predicted_label[idx]))
			print('torch.sum(idx)',torch.sum(idx))
			acc_per_class +=	torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
			print('acc_per_class for class %d = %.4f '%(i,acc_per_class))
		acc_per_class	/= target_classes.size(0)
		return acc_per_class 
	'''	
	
	def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
		acc_per_class	= torch.FloatTensor(target_classes).fill_(0)
		acc_per_class= acc_per_class.cuda()	
		test_label=test_label.cuda()
		predicted_label=predicted_label.cuda()
		for i	in range(target_classes):
			idx = (test_label ==	i)
			#print(print('idx',idx))
			#print('torch.sum(test_label[idx]==predicted_label[idx])',torch.sum(test_label[idx]==predicted_label[idx]))
			#print('torch.sum(idx)',torch.sum(idx))
			acc_per_class [i]=	torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
			#print('acc_per_class for class %d = %.4f '%(i,acc_per_class [i]))
		#print('acc_per_class:',acc_per_class)

		acc_per_class_mean = acc_per_class.mean() 
		#acc_per_class	/= target_classes.size(0)
		return acc_per_class_mean
	
	
	# test_label is integer 
	def val(self, test_X, test_label, target_classes):	
		start	= 0
		ntest	= test_X.size()[0]
		predicted_label =	torch.LongTensor(test_label.size())
		for i	in range(0,	ntest, self.batch_size):
			end = min(ntest,	start+self.batch_size)
			if self.cuda:
				output = self.model(Variable(test_X[start:end].cuda(),requires_grad=False)) 
			else:
				output = self.model(Variable(test_X[start:end],requires_grad=False))	
			_, predicted_label[start:end] = torch.max(output.data, 1)
			start = end
		#print('test_label',test_label)
		#print('target_classes',target_classes)
		#print('util.map_label(test_label, target_classes',util.map_label(test_label, target_classes	))
		#print('predicted_label',predicted_label)
		acc =	self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label,	target_classes.size(0))
		return acc

	def compute_per_class_acc(self, test_label, predicted_label, nclass):
		acc_per_class	= torch.FloatTensor(nclass).fill_(0)
		acc_per_class= acc_per_class.cuda()
		test_label=test_label.cuda()
		predicted_label=predicted_label.cuda()
		
    
		for i	in range(nclass):
			idx = (test_label ==	i)
			#print('torch.sum(test_label[idx]==predicted_label[idx])',torch.sum(test_label[idx]==predicted_label[idx]))
			#print('torch.sum(idx)',torch.sum(idx))
			acc_per_class[i]	= torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
			print('acc_per_class for class %d = %.4f '%(i,acc_per_class[i]))
		print('acc_per_class:',acc_per_class)
		return acc_per_class.mean() 

class LINEAR_LOGSOFTMAX(nn.Module):
	def __init__(self,	input_dim, nclass):
		super(LINEAR_LOGSOFTMAX, self).__init__()
		self.fc =	nn.Linear(input_dim, nclass)
		self.logic = nn.LogSoftmax(dim=1)
	def forward(self, x): 
		o	= self.logic(self.fc(x))
		return o	
