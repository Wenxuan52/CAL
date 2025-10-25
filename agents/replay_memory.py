import random
import numpy as np
from operator import itemgetter

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = [state, action, reward, next_state, done]
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * int(append_len))

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[:len(self.buffer) - self.position]
            self.buffer[:len(batch) - len(self.buffer) + self.position] = batch[len(self.buffer) - self.position:]
            self.position = len(batch) - len(self.buffer) + self.position

    def _ordered_buffer(self):
        if len(self.buffer) < self.capacity:
            return list(self.buffer)
        return list(self.buffer[self.position:] + self.buffer[:self.position])

    def sample(self, batch_size, rollout_len=1):
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty buffer")

        rollout_len = max(1, int(rollout_len))

        if rollout_len == 1:
            if batch_size > len(self.buffer):
                batch_size = len(self.buffer)
            batch = random.sample(self.buffer, int(batch_size))
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done

        ordered_buffer = self._ordered_buffer()
        if len(ordered_buffer) < rollout_len:
            raise ValueError("Not enough transitions for the requested rollout length")

        max_start = len(ordered_buffer) - rollout_len + 1
        start_indices = np.random.randint(0, max_start, int(batch_size))
        sequences = [ordered_buffer[idx: idx + rollout_len] for idx in start_indices]

        state = np.stack([[step[0] for step in seq] for seq in sequences], axis=0)
        action = np.stack([[step[1] for step in seq] for seq in sequences], axis=0)
        reward = np.stack([[step[2] for step in seq] for seq in sequences], axis=0)
        next_state = np.stack([[step[3] for step in seq] for seq in sequences], axis=0)
        done = np.stack([[step[4] for step in seq] for seq in sequences], axis=0)

        return state, action, reward, next_state, done

    def sample_all_batch(self, batch_size):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def return_all(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)

