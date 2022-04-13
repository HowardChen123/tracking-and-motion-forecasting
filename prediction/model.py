from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from torch import Tensor, nn

from prediction.modules.loss_function import PredictionLossConfig
from prediction.types import Trajectories
from prediction.utils.reshape import flatten, unflatten_batch
from prediction.utils.transform import transform_using_actor_frame


@dataclass
class PredictionModelConfig:
    """Prediction model configuration."""

    loss: PredictionLossConfig = field(
        default_factory=lambda: PredictionLossConfig(
            l1_loss_weight=1.0,
        )
    )
    num_history_timesteps: int = 10  # Number of timesteps in the history
    num_label_timesteps: int = 10  # Number of timesteps to predict


class PredictionModel(nn.Module):
    """A basic object Prediction model."""

    def __init__(self, config: PredictionModelConfig) -> None:
        super().__init__()

        # TODO: Implement
        self._encoder = nn.Linear(config.num_history_timesteps*3, 128)

        # TODO: Implement
        self._decoder = nn.Linear(128, config.num_label_timesteps*6)

    @staticmethod
    def _preprocess(x_batches: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Preprocess the inputs

        1. Flatten batch and actor dimensions
        2. Transform each actor's history so that its position at the latest timestep is (0, 0) with 0 rad of yaw
            (i.e. it is in actor frame)
        3. Pad nans with zero
        4. Remove the bounding box size from the inputs
        5. Flatten the time and feature dimensions

        Args:
            x_batches (List[Tensor]): List of length batch_size of [N x T x 5] trajectories

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - preprocessed input trajectories [batch_size * N x T * 3]
                - id of each actor's batch in the flattened list [batch_size * N]
                - original position and yaw of each actor at the latest timestep in SDV frame [batch_size * N, 3]
        """
        x, batch_ids = flatten(x_batches)  # [batch_size * N x T x 5]
        original_x_pose = torch.clone(x[:, -1, :3])

        # Move positions to actor frame
        transformed_positions = transform_using_actor_frame(
            x[..., :2], x[:, -1, :3], translate_to=True
        )
        x[..., :2] = transformed_positions
        # Move yaw to actor frame
        x[..., 2] = x[..., 2] - x[:, -1:, 2]

        # Pad nans
        x[x.isnan()] = 0

        # Remove box size
        x = x[..., :3]

        x = x.flatten(1, 2)  # [batch_size * N x T * 3]

        return x, batch_ids, original_x_pose

    @staticmethod
    def _postprocess(
        out: Tensor, batch_ids: Tensor, original_x_pose: Tensor
    ) -> List[Tensor]:
        """Postprocess predictions

        1. Unflatten time and position dimensions
        2. Transform predictions back into SDV frame
        3. Unflatten batch and actor dimension

        Args:
            out (Tensor): predicted input trajectories [batch_size * N x T * 6]
            batch_ids (Tensor): id of each actor's batch in the flattened list [batch_size * N]
            original_x_pose (Tensor): original position and yaw of each actor at the latest timestep in SDV frame
                [batch_size * N, 3]

        Returns:
            List[Tensor]: List of length batch_size of output predicted trajectories in SDV frame [N x T x 6]
        """
        num_actors = len(batch_ids)
        out = out.reshape(num_actors, -1, 6)  # [batch_size * N x T x 6]
        
        # Transform from actor frame, to make the prediction problem easier
        transformed_out = transform_using_actor_frame(
            out[:, :, :2], original_x_pose, translate_to=False      # torch.Size([85, 10, 2])
        )
        # Transform the Covariance matrix to positive semidefinite
        # transposed = torch.cat((out[: , :, 2].unsqueeze(-1), out[: , :, 4].unsqueeze(-1), 
        #                         out[: , :, 3].unsqueeze(-1), out[: , :, 5].unsqueeze(-1)), 
        #                         dim = 2)     # torch.Size([85, 10, 4])
        a = torch.clone(out[:, :, 2])
        b = torch.clone(out[:, :, 3])
        c = torch.clone(out[:, :, 4])
        d = torch.clone(out[:, :, 5])
        covariance_matrix = out[:, :, 2:]
        covariance_matrix[:, :, 0] = torch.square(a) + torch.square(b)
        covariance_matrix[:, :, 1] = a*c+b*d
        covariance_matrix[:, :, 2] = a*c+b*d
        covariance_matrix[:, :, 3] = torch.square(c) + torch.square(d)

        # Concat back with the covariance matrix
        transformed_out = torch.cat((transformed_out, covariance_matrix), dim = 2)    # torch.Size([85, 10, 6])

        # Translate so that latest timestep for each actor is the origin
        out_batches = unflatten_batch(transformed_out, batch_ids)

        return out_batches

    def forward(self, x_batches: List[Tensor]) -> List[Tensor]:
        """Perform a forward pass of the model's neural network.

        Args:
            x_batches: A [batch_size x N x T_in x 5] tensor, representing the input history
                centroid, yaw and size in a bird's eye view voxel representation.

        Returns:
            A [batch_size x N x T_out x 6] tensor, representing the future trajectory
                centroid outputs.
        """
        x, batch_ids, original_x_pose = self._preprocess(x_batches)
        out = self._decoder(self._encoder(x))   # torch.Size([85, 60])
        out_batches = self._postprocess(out, batch_ids, original_x_pose)    # (out_batches[0]).shape == torch.Size([85, 10, 6])
        return out_batches

    @torch.no_grad()
    def inference(self, history: Tensor) -> Trajectories:
        """Predict a set of 2d future trajectory predictions from the detection history

        Args:
            history: A [batch_size x N x T x 5] tensor, representing the input history
                centroid, yaw and size in a bird's eye view voxel representation.

        Returns:
            A set of 2D future trajectory centroid predictions.
        """
        self.eval()
        pred = self.forward([history])[0]  # shape: B * N x T x 2
        num_timesteps, num_coords = pred.shape[-2:]

        # Add dummy values for yaws and boxes here because we will fill them in from the ground truth
        return Trajectories(
            pred,
            torch.zeros(pred.shape[0], num_timesteps),
            torch.ones(pred.shape[0], num_coords),
        )
