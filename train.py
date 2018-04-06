from __future__ import unicode_literals, print_function, division
import numpy as np
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn
import model
import utils
import sys
import time

torch.manual_seed(1)

# ==============================================
## DATA PATH
TRAIN_DATA_PATH = "../../../../dataset/instagram/rnn/prediction/train_total.txt"
VALIDATAION_DATA_PATH = "../../../../dataset/instagram/rnn/prediction/validation_total.txt"
TEST_DATA_PATH = "../../../../dataset/instagram/rnn/prediction/test_total.txt"

## HYPER-PARAMETERS
EMBEDDING_DIM = 50
RNN_STEP = 5
EPOCHS = 1000
BATCH_SIZE = 2000
LEARNING_RATE = 0.001
SAMPLE_NUM = 8

# =============================================
## DATA PREPARATION
print("========================================")
print("Data Loading..")
train, validation, test = utils.load_data(TRAIN_DATA_PATH, VALIDATAION_DATA_PATH, TEST_DATA_PATH)

print("Make Dictionary..")
train, validation, test, user2id, id2user, venue2id, id2venue, venue_frequency = utils.make_dict(train, validation, test)

print("Make Input..")
train, validation, test = utils.make_input(train, validation, test, RNN_STEP)

# =============================================
def get_eval_score(candidate, rank):
	_mrr = .0

	for i in xrange(len(candidate)):
		_rank = np.where(rank[i] == candidate[i])
		_mrr += (1.0/(_rank[0]+1))

	return _mrr

## Training
print("========================================")
print("Training..")

drcf = model.DRCF(EMBEDDING_DIM, RNN_STEP, len(user2id), len(venue2id), SAMPLE_NUM)
drcf = nn.DataParallel(drcf).cuda()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, drcf.parameters()), lr=LEARNING_RATE)
criterion = nn.LogSigmoid()

for i in xrange(EPOCHS):
	# Training

	drcf.train()
	step = 0
	loss = .0
	batch_num = int(len(train)/BATCH_SIZE) + 1

	batches = utils.batches(train, BATCH_SIZE, SAMPLE_NUM, venue_frequency)
	for batch in batches:
		user, candidate, checkins, samples = batch
		input_user = Variable(torch.cuda.LongTensor(user))
		input_candidate = Variable(torch.cuda.LongTensor(candidate))
		input_checkins = Variable(torch.cuda.LongTensor(checkins))
		input_samples = Variable(torch.cuda.LongTensor(samples))

		# Optimizing
		optimizer.zero_grad()
		_loss = -criterion(drcf(input_user, input_candidate, input_checkins, input_samples)).sum()
		_loss.backward()
		optimizer.step()
		loss+=_loss.cpu().data.numpy()[0]

		# Printing Progress
		step+=1
		sys.stdout.write("\033[F")
		sys.stdout.write("\033[K")
		print("Process Training Epoch: [{}/{}] Batch: [{}/{}] Loss: {}".format(i+1, EPOCHS, step, batch_num, _loss.cpu().data.numpy()[0]))


	if (i+1) % 10 == 0:
		# Validation
		drcf.eval()
		step = 0
		mrr = .0
		batch_num = int(len(validation)/100) + 1

		batches = utils.batches(validation, 100, SAMPLE_NUM, venue_frequency)
		for batch in batches:
			user, candidate, checkins, _ = batch
			input_user = Variable(torch.cuda.LongTensor(user))
			input_checkins = Variable(torch.cuda.LongTensor(checkins))
			
			# Optimizing
			rank = drcf.module.evaluation(input_user, input_checkins).cpu().data.numpy()
			mrr += get_eval_score(candidate, rank)

			# Printing Progress
			step+=1
			sys.stdout.write("\033[F")
			sys.stdout.write("\033[K")
			print("Process Evaluation Epoch: [{}/{}] Batch: [{}/{}]".format(i+1, EPOCHS, step, batch_num))

		sys.stdout.write("\033[F")
		sys.stdout.write("\033[K")
		print("Process Epoch: [{}/{}] loss : [{}] / Eval: [{}]\n".format(i+1, EPOCHS, loss, mrr/len(validation)))

	else:
		sys.stdout.write("\033[F")
		sys.stdout.write("\033[K")
		print("Process Epoch: [{}/{}] loss : [{}]\n".format(i+1, EPOCHS, loss, ))
