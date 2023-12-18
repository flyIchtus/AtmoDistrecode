import torch
import torch.nn.optimize as optim
import torch.nn as nn
import TimeDeltaDataset as TDD
import metrics
import networks
import tqdm

class Trainer():
    def __init__(self, config):

        self.config = config
        
        self.Optimizer = optim.SGD(Model.parameters(), self.config.lr, beta = self.config.beta, weight_decay = self.config.wd)

    def log(self):
        pass
        
    def test(self, Model, testIter):
        xt1, xt2, delta_t = next(testIter)
        xt1 = xt1.cuda()
        xt2 = xt2.cuda()
        delta_t = delta_t.cuda()
            
        Model.eval()
        
        with torch.no_grad():
            interm1 = Model.Repre(x1)
            interm2 = Model.Repre(x2)       
        
            atmoDist = metrics.atmodist(interm1, interm2)
            
            pred_delta_t = Model.Compar(torch.cat((interm1, interm2), dim=1))
            
            test_loss = nn.CrossEntropy(pred_delta_t, delta_t)
            
        Model.train()
        return atmoDist, test_loss
    
    def pretrain(self, Model):
    	""" Pretraining with fewer data (specific sub-train dataset)"""
        PreTrainDataset = TDD.Tdataset(config.pre_train_file)

    def fit(self, Model, train_file, test_file):
    	""" Main training function"""
     	
    	TrainDataset = TDD.Tdataset(train_file)
    	TestDataset = TDD.Tdataset(test_file)
    	
    	trainDataloader = TDD.Tdataloader(dataset = TrainDataset, batch_size = self.config.batch_size)
    	testDataloader = TDD.Tdataloader(dataset = TestDataset, batch_size = self.config.batch_size)
    	
    	N_batch = len(TrainDataset)//self.config.batch_size
    	
    	testDataloader.sampler.set_epoch(0)
    	
    	testIter = iter(testDataloader)
    	
    	for e in self.config.epochs_num:
    	    
	    trainDataloader.sampler.set_epoch(e)

	    loop = enumerate(trainDataloader)
	    
	    for step, batch in loop:
	        
	        losses_epoch = {}
	    
	    	Step = e * N_batch + step
	    
	    	xt1, xt2, delta_t = batch
	    	xt1 = xt1.cuda()
	    	xt2 = xt2.cuda()
    	    
    	    	delta_t = delta_t.cuda()
		    	    	
    	    	pred_delta_t = Model(x1, x2)
    	    	
    	    	loss = nn.CrossEntropy(pred_delta_t, delta_t)
    	    	
    	    	losses_epoch[step] = loss.detach().cpu().item()
    	    	
		self.Optimizer.zero_grad()
		loss.backward()
		self.Optimizer.step()
		
            
            
            
            
            
            
		    	    
    	
    	
    
