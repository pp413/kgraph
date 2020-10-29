import kgraph
from kgraph.utils import Trainer, Tester
from kgraph.module.model import TransE
from kgraph.module.loss import MarginLoss
from kgraph.module.strategy import NegativeSampling
from kgraph.datasets import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	dataset_name= "FB15k-237", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("FB15k-237", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_epoch = 1, learning_rate = 1.0, use_gpu = True)
trainer.run()
transe.save_checkpoint('./checkpoint/transe.ckpt')

# test the model
transe.load_checkpoint('./checkpoint/transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)

