from __future__ import unicode_literals, print_function, division
import numpy as np
import random
import time

def load_data(TRAIN_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH):
	train = [line.strip() for line in open(TRAIN_DATA_PATH).readlines()]
	validation = [line.strip() for line in open(VALIDATION_DATA_PATH).readlines()]
	test = [line.strip() for line in open(TEST_DATA_PATH).readlines()]

	return train, validation, test

def make_dict(_train, _validation, _test):
	user2id = {}
	id2user = []
	venue2id = {"<PAD>": 0}
	id2venue = ["<PAD>"]

	train = {}
	validation = {}
	test = {}

	"""
		Dataset Format: user \t venue \t timestamp
	"""
	for line in _train+_validation+_test:
		user, venue, timestamp = line.split("\t")
		
		if user not in user2id:
			user2id[user] = len(user2id)
			id2user.append(user)
		if venue not in venue2id:
			venue2id[venue] = len(venue)
			id2venue.append(venue)

	venue_frequency = np.zeros(len(venue2id), dtype=np.float32)

	for line in _train:
		user, venue, timestamp = line.split("\t")
		venue_frequency[venue2id[venue]] += 1
		if user not in train:
			train[user] = {"id":user2id[user], "checkins":[venue2id[venue]]}
		else:
			train[user]["checkins"].append(venue2id[venue])

	for line in _validation:
		user, venue, timestamp = line.split("\t")
		venue_frequency[venue2id[venue]] += 1
		if user not in validation:
			validation[user] = {"id":user2id[user], "checkins":[venue2id[venue]]}
		else:
			validation[user]["checkins"].append(venue2id[venue])

	for line in _test:
		user, venue, timestamp = line.split("\t")
		venue_frequency[venue2id[venue]] += 1
		if user not in test:
			test[user] = {"id":user2id[user], "checkins":[venue2id[venue]]}
		else:
			test[user]["checkins"].append(venue2id[venue])

	return train, validation, test, user2id, id2user, venue2id, id2venue, venue_frequency

def make_input(_train, _validation, _test, RNN_STEPS):
	train, validation, test = [], [], []

	for _, value in _train.items():
		user, checkins = value["id"], value["checkins"]
		if len(checkins) <= RNN_STEPS:
			checkins = [0]*(RNN_STEPS - len(checkins) + 1) + checkins
		for i in xrange(RNN_STEPS, len(checkins)):
			train.append((user, checkins[i], checkins[i - RNN_STEPS:i]))

	for _, value in _validation.items():
		user, checkins = value["id"], value["checkins"]
		if len(checkins) <= RNN_STEPS:
			checkins = [0]*(RNN_STEPS - len(checkins) + 1) + checkins

		validation.append((user, checkins[-1], checkins[-(1+RNN_STEPS):-1]))

	for _, value in _test.items():
		user, checkins = value["id"], value["checkins"]
		if len(checkins) <= RNN_STEPS:
			checkins = [0]*(RNN_STEPS - len(checkins) + 1) + checkins

		test.append((user, checkins[-1], checkins[-(1+RNN_STEPS):-1]))

	return train, validation, test

def batches(_data, BATCH_SIZE, SAMPLES_NUM, venue_frequency):

	random.shuffle(_data)
	batch_num = int(len(_data)/BATCH_SIZE) + 1

	for i in xrange(batch_num):
		user = []
		candidate = []
		checkins = []
		# samples = []

		left = i*BATCH_SIZE
		right = min((i+1)*BATCH_SIZE, len(_data))

		
		start = time.time()
		for data in _data[left:right]:
			user.append(data[0])
			candidate.append(data[1])
			checkins.append(data[2])
		
			# # For Weighted Negative Sampling
			# total_checkins = sum(venue_frequency) - venue_frequency[data[1]]
			# sampled_probability = [j/total_checkins for j in venue_frequency]
			# sampled_probability[0] = 0
			# sampled_probability[data[1]] = 0
			# samples.append(np.random.choice(len(venue_frequency), SAMPLES_NUM, replace=False, p=sampled_probability))
		samples = np.random.randint(len(venue_frequency), size=(right-left, SAMPLES_NUM))
		yield user, candidate, checkins, samples