import time
import os
import collections
import json
import numpy as np
import data.data_io as data_io

class DumbModel:
    def train(self, batch):
        pass
        pass
    def evaluate(self, batch):
        pass
    def save_checkpoint(self):
        pass

def batch_iterator(iterator, batch_size):
    while True:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(iterator))
            except StopIteration:
                return
        yield batch

class TimeTrainer:
    def __init__(self, lpd5_root, model, checkpoint_dir, total_time=600,
                 epoch_interval=1, batch_size=64, track_name='Piano',
                 beat_resolution=4, split_len=256):
        self.test_set, self.train_set = data_io.test_train_paths_lpd5(lpd5_root)
        self.model = model
        self.total_time_remaining = total_time
        self.num_epochs = 0
        self.epoch_interval = epoch_interval
        self.batch_size = batch_size
        self.track_name = track_name
        self.beat_resolution = beat_resolution
        self.split_len = split_len
        self.checkpoint_dir = checkpoint_dir
        self.losses = list()
    
    def load_checkpoint(self, path):
        with open(path, 'r') as f:
            self.num_epochs, self.total_time_remaining, self.losses = json.load(f)
    
    def save_checkpoints(self):
        self.model.save_checkpoint()
        time_str = str(int(time.time()))+".json"
        with open(os.path.join(self.checkpoint_dir, time_str), 'w') as f:
            json.dump([self.num_epochs, self.total_time_remaining, self.losses], f)
    
    def initialize_epoch(self):
        self.epoch_start_time = time.time()
        self.train_iterator = batch_iterator(
            data_io.iter_lpd5_paths(self.train_set, 
                                    self.track_name,
                                    self.beat_resolution,
                                    self.split_len),
            self.batch_size)
    
    def run_loop(self):
        self.initialize_epoch()
        while self.total_time_remaining > 0:
            try:
                next_batch = next(self.train_iterator)
            except StopIteration:
                self.update_losses(time.time())
                self.save_checkpoints()
                self.conclude_epoch()
                self.initialize_epoch()
                print(f"Finish epoch {self.num_epochs} with {self.total_time_remaining:.2f} time remaining")
            else:
                self.model.train(next_batch)
    
    def conclude_epoch(self):
        self.num_epochs += 1
        self.total_time_remaining -= (time.time() - self.epoch_start_time)
    
    def update_losses(self, current_time):
        evaluation_iterator = batch_iterator(
            data_io.iter_lpd5_paths(self.test_set, self.track_name,
                                    self.beat_resolution, self.split_len),
            self.batch_size)
        temp_losses = [self.calculate_loss(batch) for batch in evaluation_iterator]
        if temp_losses:
            self.losses.append([current_time, np.mean(temp_losses)])
        else:
            print("ERROR: No losses calculated!")
    
    def calculate_loss(self, batch):
        return self.model.evaluate(batch)