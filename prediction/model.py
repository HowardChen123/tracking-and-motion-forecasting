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
        modules = []
        hidden_dim = [32, 64, 128]
        input_dim = config.num_history_timesteps*3
        for h_dim in hidden_dim:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, h_dim),
                    nn.ReLU()
                )
            )
            input_dim = h_dim
        # self._encoder = nn.Sequential(*modules)
        # self.fc_mu = nn.Linear(128, 128)
        # self.fc_var = nn.Linear(128, 128)
        self._encoder = nn.Linear(config.num_history_timesteps*3, 128)

        # TODO: Implement
        self._decoder = nn.Linear(128, config.num_label_timesteps*5)
        

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
            out (Tensor): predicted input trajectories [batch_size * N x T * 4]
            batch_ids (Tensor): id of each actor's batch in the flattened list [batch_size * N]
            original_x_pose (Tensor): original position and yaw of each actor at the latest timestep in SDV frame
                [batch_size * N, 3]

        Returns:
            List[Tensor]: List of length batch_size of output predicted trajectories in SDV frame [N x T x 4]
        """
        num_actors = len(batch_ids)
        out = out.reshape(num_actors, -1, 5)  # [batch_size * N x T x 4]
        
        # Transform from actor frame, to make the prediction problem easier
        transformed_out = transform_using_actor_frame(
            out[:, :, :2], original_x_pose, translate_to=False      # torch.Size([85, 10, 2])
        )
        # Calculate the Covariance matrix
        sigma_x = torch.clone(out[:, :, 2])
        sigma_y = torch.clone(out[:, :, 3])
        rho = nn.functional.sigmoid(torch.clone(out[:, :, 4]))

        covariance_matrix = out[:, :, 1:]
        covariance_matrix[:, :, 0] = torch.square(sigma_x)
        covariance_matrix[:, :, 1] = rho*sigma_x*sigma_y
        covariance_matrix[:, :, 2] = rho*sigma_x*sigma_y
        covariance_matrix[:, :, 3] = torch.square(sigma_y)

        # Concat back with the covariance matrix
        transformed_out = torch.cat((transformed_out, covariance_matrix), dim = 2)    # torch.Size([85, 10, 4])

        # Translate so that latest timestep for each actor is the origin
        out_batches = unflatten_batch(transformed_out, batch_ids)

        return out_batches

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x_batches: List[Tensor]) -> List[Tensor]:
        """Perform a forward pass of the model's neural network.

        Args:
            x_batches: A [batch_size x N x T_in x 5] tensor, representing the input history
                centroid, yaw and size in a bird's eye view voxel representation.

        Returns:
            A [batch_size x N x T_out x 4] tensor, representing the future trajectory
                centroid outputs.
        """
        x, batch_ids, original_x_pose = self._preprocess(x_batches)
        encode = self._encoder(x)
        # mu = self.fc_mu(encode)
        # log_var = self.fc_var(encode)
        # z = self.reparameterize(mu, log_var)

        out = self._decoder(encode)
        out_batches = self._postprocess(out, batch_ids, original_x_pose)    # (out_batches[0]).shape == torch.Size([85, 10, 4])
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
            pred[..., :2],
            torch.zeros(pred.shape[0], num_timesteps),
            torch.ones(pred.shape[0], num_coords),
        )
