from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor, nn


def compute_NLL_loss(targets: Tensor, predictions: Tensor):
    """Compute the negative log-likelihood (NLL) loss between `predictions` and `targets`.

    Args:
        targets A [num_actors x T x 2] tensor, containing the ground truth targets.
        predictions: A [num_actors x T x 6] tensor, containing the predictions.

    Returns:
        A scalar NLL loss between `predictions` and `targets`
    """
    num_actors, T, _ = targets.shape

    mask = torch.isnan(targets).any(-1).any(-1)
    targets = targets[~mask].view(-1, T, 2)
    num_actors, T, _ = predictions.shape
    predictions = predictions[~mask].view(-1, T, 6)

    num_actors, T, _ = targets.shape

    mu = predictions[:, :, :2]
    sigma = predictions[:, :, 2:]

    det_sigma = (sigma[:, :, 0]*sigma[:, :, 3] - sigma[:, :, 1]*sigma[:, :, 2]).unsqueeze(-1)

    det_sigma[det_sigma <= 0] = 1e-7

    sigma_inv = torch.clone(sigma)
    sigma_inv[:, :, 0] = torch.clone(sigma[:, :, 3])
    sigma_inv[:, :, 1] = torch.clone(-sigma[:, :, 1])
    sigma_inv[:, :, 2] = torch.clone(-sigma[:, :, 2])
    sigma_inv[:, :, 3] = torch.clone(sigma[:, :, 0])
    sigma_inv = sigma_inv/torch.cat((det_sigma, det_sigma, det_sigma, det_sigma), dim = 2)

    sigma_inv = sigma_inv.reshape(num_actors, T, 2, 2)
    ll = torch.matmul((targets - mu).reshape(num_actors, T, 1, 2), sigma_inv)
    ll = torch.matmul(ll, (targets - mu).unsqueeze(-1)).squeeze(-1)

    # print("Sigma Inv", torch.sum(sigma_inv.isnan()))
    # print("Targets", torch.sum(targets.isnan()))
    # print("Mu", torch.sum(mu.isnan()))
    # print("ll", torch.sum(ll.isnan()))

    nll = 0.5*(torch.log(det_sigma) + ll)

    # print("Det Sigma", torch.sum(det_sigma.isnan()))
    # print("Det Sigma Less than 0", torch.sum(det_sigma <= 0))
    # print("Log Det Sigma", torch.sum(torch.log(det_sigma).isnan()))
    # print("Log Det Sigma Negative Inf", torch.sum(torch.log(det_sigma + eps) == float("-inf")))
    # print("NLL loss", torch.sum(nll)/num_actors, "\n")

    return torch.sum(nll)/num_actors


def compute_l1_loss(targets: Tensor, predictions: Tensor) -> Tensor:
    """Compute the mean absolute error (MAE)/L1 loss between `predictions` and `targets`.

    Specifically, the l1-weighted MSE loss can be computed as follows:
    1. Compute a binary mask of the `targets` that are not NaN, and apply it to the `targets` and `predictions`
    2. Compute the MAE loss between `predictions` and `targets`.
        This should give us a [batch_size * num_actors x T x 2] tensor `l1_loss`.
    3. Compute the mean of `l1_loss`. This gives us our final scalar loss.

    Args:
        targets: A [batch_size * num_actors x T x 2] tensor, containing the ground truth targets.
        predictions: A [batch_size * num_actors x T x 2] tensor, containing the predictions.

    Returns:
        A scalar MAE loss between `predictions` and `targets`
    """
    # TODO: Implement.
    idx = torch.isnan(targets)
    targets = targets[~idx]
    predictions = predictions[~idx]
    loss = torch.nn.L1Loss(reduction = 'sum')
    l1_loss = loss(predictions, targets)
    return l1_loss


@dataclass
class PredictionLossConfig:
    """Prediction loss function configuration.

    Attributes:
        l1_loss_weight: The multiplicative weight of the L1 loss
    """

    l1_loss_weight: float


@dataclass
class PredictionLossMetadata:
    """Detailed breakdown of the Prediction loss."""

    total_loss: torch.Tensor
    l1_loss: torch.Tensor


class PredictionLossFunction(torch.nn.Module):
    """A loss function to train a Prediction model."""

    def __init__(self, config: PredictionLossConfig) -> None:
        super(PredictionLossFunction, self).__init__()
        self._l1_loss_weight = config.l1_loss_weight

    def forward(
        self, predictions: List[Tensor], targets: List[Tensor]
    ) -> Tuple[torch.Tensor, PredictionLossMetadata]:
        """Compute the loss between the predicted Predictions and target labels.

        Args:
            predictions: A list of batch_size x [num_actors x T x 2] tensor containing the outputs of
                `PredictionModel`.
            targets:  A list of batch_size x [num_actors x T x 2] tensor containing the ground truth output.

        Returns:
            The scalar tensor containing the weighted loss between `predictions` and `targets`.
        """
        predictions_tensor = torch.cat(predictions)  # [num_actors x T x 6]
        targets_tensor = torch.cat(targets)  # [num_actors x T x 2]
        # 1. Unpack the targets tensor.
        target_centroids = targets_tensor[..., :2]  # [batch_size * num_actors x T x 2]

        # # 2. Unpack the predictions tensor.
        # predicted_centroids = predictions_tensor[
        #     ..., :2
        # ]  # [batch_size * num_actors x T x 2]

        # # 3. Compute individual loss terms for l1
        # l1_loss = compute_l1_loss(target_centroids, predicted_centroids)
        nll_loss = compute_NLL_loss(target_centroids, predictions_tensor)

        # 4. Aggregate losses using the configured weights.
        total_loss = nll_loss * self._l1_loss_weight
        loss_metadata = PredictionLossMetadata(total_loss, nll_loss)
        return total_loss, loss_metadata
