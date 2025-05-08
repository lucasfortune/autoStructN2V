# trainers/callbacks.py

class EarlyStopping:
    """
    Early stopping callback to prevent overfitting during training.
    
    This callback monitors a validation metric and stops training when the metric
    has stopped improving for a specified number of epochs.
    
    Args:
        patience (int, optional): Number of epochs with no improvement after which
            training will be stopped. Defaults to 10.
        min_delta (float, optional): Minimum change in the monitored quantity to
            qualify as an improvement. Defaults to 0.001.
    """
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        """
        Check if training should be stopped based on validation loss.
        
        Args:
            val_loss (float): Current validation loss
            
        Returns:
            bool: True if training should be stopped, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience