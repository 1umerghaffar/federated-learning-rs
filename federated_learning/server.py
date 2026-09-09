"""
Federated Learning Server - Central aggregation and coordination
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedServer:
    """Central server for federated learning coordination"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.01,
        num_rounds: int = 100,
        fraction_fit: float = 1.0,
        device: str = "cpu"
    ):
        """
        Initialize federated server
        
        Args:
            model: PyTorch model to train
            learning_rate: Server-side learning rate
            num_rounds: Number of federated rounds
            fraction_fit: Fraction of clients to train in each round (0-1)
            device: Device to use (cpu/cuda)
        """
        self.model = model.to(device)
        self.learning_rate = learning_rate
        self.num_rounds = num_rounds
        self.fraction_fit = fraction_fit
        self.device = device
        self.client_updates = []
        self.round_loss = []
        
    def get_parameters(self) -> List[torch.Tensor]:
        """Extract model parameters"""
        return [param.clone().detach() for param in self.model.parameters()]
    
    def set_parameters(self, parameters: List[torch.Tensor]):
        """Set model parameters"""
        with torch.no_grad():
            for param, update in zip(self.model.parameters(), parameters):
                param.copy_(update)
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """
        Federated Averaging (FedAvg) aggregation
        
        Args:
            client_updates: List of client model updates with weights
            
        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            return self.get_parameters()
        
        # Calculate weighted average
        total_samples = sum(update["num_samples"] for update in client_updates)
        aggregated = None
        
        for update in client_updates:
            weight = update["num_samples"] / total_samples
            
            if aggregated is None:
                aggregated = [p.clone() * weight for p in update["parameters"]]
            else:
                for i, param in enumerate(update["parameters"]):
                    aggregated[i] += param * weight
        
        logger.info(f"Aggregated {len(client_updates)} client updates")
        return aggregated
    
    def federated_round(self, client_updates: List[Dict[str, Any]]) -> float:
        """
        Execute one round of federated learning
        
        Args:
            client_updates: Updates from clients
            
        Returns:
            Loss value
        """
        # Aggregate client updates
        aggregated_params = self.aggregate(client_updates)
        
        # Update server model
        self.set_parameters(aggregated_params)
        
        # Calculate average loss from clients
        avg_loss = sum(u["loss"] for u in client_updates) / len(client_updates)
        self.round_loss.append(avg_loss)
        
        logger.info(f"Round loss: {avg_loss:.4f}")
        return avg_loss
    
    def train(self, client_updates_generator):
        """
        Train for multiple rounds
        
        Args:
            client_updates_generator: Generator yielding client updates for each round
        """
        for round_num in range(self.num_rounds):
            logger.info(f"\n--- Federated Round {round_num + 1}/{self.num_rounds} ---")
            
            # Get updates from clients
            updates = next(client_updates_generator)
            
            # Perform aggregation and update
            loss = self.federated_round(updates)
            
            if (round_num + 1) % 10 == 0:
                logger.info(f"Checkpoint at round {round_num + 1}")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")
