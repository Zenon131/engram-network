import torch
import unittest
from dragon.model_torch import BDH_GPU
from dragon.data import TextDataset, TextDataLoader
from dragon.train_loop import train_step
import torch.optim as optim


class TestTrainStep(unittest.TestCase):
    def setUp(self):
        # Create a small model for testing
        self.model = BDH_GPU(vocab_size=128, n_neurons=256, d_model=64, n_layers=2)
        
        # Create a small dummy dataset
        self.dataset = TextDataset('data/raw/sample.txt', 'byte', max_length=32)
        self.data_loader = TextDataLoader(self.dataset, batch_size=4, shuffle=True)
        
        self.device = torch.device('cpu')  # Use CPU for testing
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def test_train_step_overfit(self):
        """
        Test that the model can overfit a tiny batch.
        """
        # Get a single batch
        batch = next(iter(self.data_loader))
        inputs, targets = batch
        
        # Convert to tensors
        inputs = torch.tensor(inputs, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
        
        # Create a new data loader with just this batch repeated
        class SingleBatchDataLoader:
            def __init__(self, inputs, targets, num_batches=10):
                self.inputs = inputs
                self.targets = targets
                self.num_batches = num_batches
                self.current_batch = 0
                
            def __iter__(self):
                self.current_batch = 0
                return self
                
            def __next__(self):
                if self.current_batch < self.num_batches:
                    self.current_batch += 1
                    return self.inputs, self.targets
                else:
                    raise StopIteration
                    
            def __len__(self):
                return self.num_batches
        
        single_batch_loader = SingleBatchDataLoader(inputs, targets, num_batches=10)
        
        # Initial loss
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(inputs)
            initial_loss = torch.nn.functional.cross_entropy(logits, targets.view(-1)).item()
        
        # Train for several steps
        self.model.train()
        for _ in range(10):
            train_step(self.model, single_batch_loader, self.optimizer, self.device)
        
        # Final loss should be lower
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(inputs)
            final_loss = torch.nn.functional.cross_entropy(logits, targets.view(-1)).item()
        
        self.assertLess(final_loss, initial_loss)


if __name__ == '__main__':
    unittest.main()