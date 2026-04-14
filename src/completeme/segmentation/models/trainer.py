# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Tuple

import torch
from ignite.engine import Engine
from monai.engines import SupervisedTrainer
from monai.engines.utils import CommonKeys as Keys
from monai.engines.utils import IterationEvents
from torch.nn.parallel import DistributedDataParallel
from monai.inferers import SlidingWindowInferer
from collections.abc import Callable, Mapping, Sequence
from monai.utils import ensure_tuple
from ignite.engine import Engine

class DynUNetTrainerAccumulateGradient(SupervisedTrainer):
    def __init__(self, gradient_accumulation_steps: int = 1, **kwargs):
        super().__init__(**kwargs)
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._iteration_count = 0

    def _iteration(self, engine: Engine, batchdata: Dict[str, torch.Tensor]):
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")

        # Increment iteration count for gradient accumulation
        self._iteration_count += 1
        is_accumulation_step = (self._iteration_count % self.gradient_accumulation_steps == 0)

        batch = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)
        if len(batch) == 2:
            inputs, targets = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, targets, args, kwargs = batch

        # Initialize engine.state.output as a dictionary
        # This will hold the results of this specific iteration
        iteration_output = {Keys.IMAGE: inputs, Keys.LABEL: targets}

        def _compute_pred_loss_and_backward():
            self.network.train() # Ensure network is in train mode for this step

            # Forward pass
            preds = self.inferer(inputs, self.network, *args, **kwargs)
            
            # Handle deep supervision outputs if necessary
            if preds.size(-1) != targets.size(-1):
                preds_list = [preds[..., i::self.network.deep_supr_num + 1] for i in range(self.network.deep_supr_num + 1)]
            else:
                preds_list = [preds]
            
            iteration_output[Keys.PRED] = preds_list # Store the list of predictions

            engine.fire_event(IterationEvents.FORWARD_COMPLETED)

            # Calculate total loss for deep supervision
            current_loss = sum(
                0.5**i * self.loss_function.forward(p, targets) for i, p in enumerate(iteration_output[Keys.PRED])
            )
            
            # Scale loss by gradient_accumulation_steps for correct averaging
            current_loss = current_loss / self.gradient_accumulation_steps

            # Store the normalized loss in iteration_output
            iteration_output[Keys.LOSS] = current_loss.item() 

            # Backward pass
            if self.amp and self.scaler is not None:
                self.scaler.scale(current_loss).backward()
            else:
                current_loss.backward()

            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)

        # Execute forward and backward pass, handling AMP
        if self.amp and self.scaler is not None:
            with torch.amp.autocast('cuda'):
                _compute_pred_loss_and_backward()
        else:
            _compute_pred_loss_and_backward()

        # Optimizer step and zero_grad only on accumulation steps
        if is_accumulation_step:
            if self.amp and self.scaler is not None:
                self.scaler.unscale_(self.optimizer) # Unscale gradients before clipping
                if isinstance(self.network, torch.nn.parallel.DistributedDataParallel):
                    torch.nn.utils.clip_grad_norm_(self.network.module.parameters(), 12)
                else:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.scaler.step(self.optimizer)
                self.scaler.update() # Update scaler for next iteration
            else:
                if isinstance(self.network, torch.nn.parallel.DistributedDataParallel):
                    torch.nn.utils.clip_grad_norm_(self.network.module.parameters(), 12)
                else:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
            
            self.optimizer.zero_grad() # Zero gradients AFTER the step
            engine.fire_event(IterationEvents.MODEL_COMPLETED) # Fire this only once per true step
            self._iteration_count = 0 # Reset iteration counter for the next accumulation cycle
        else:
            # If not an accumulation step, we still want to fire MODEL_COMPLETED for handlers
            # (like progress bar) that might update every iteration.
            engine.fire_event(IterationEvents.MODEL_COMPLETED)
        
        # Scale the stored loss back for logging/metrics to represent the true loss value
        iteration_output[Keys.LOSS] = iteration_output[Keys.LOSS] * self.gradient_accumulation_steps
        
        # Explicitly set engine.state.output to the dictionary for this iteration
        # This is the key change to prevent `engine.state.output` from becoming a list
        engine.state.output = iteration_output

        return iteration_output # Explicitly return the dictionary
    
class DynUNetTrainer(SupervisedTrainer):
    """
    This class inherits from SupervisedTrainer in MONAI, and is used with DynUNet
    on Decathlon datasets.

    """

    def _iteration(self, engine: Engine, batchdata: Dict[str, torch.Tensor]):
        """
        Callback function for the Supervised Training processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.
            - LOSS: loss value computed by loss function.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        
        batch = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)
        if len(batch) == 2:
            inputs, targets = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, targets, args, kwargs = batch

        # put iteration outputs into engine.state
        engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: targets}

        def _compute_pred_loss():
            preds = self.inferer(inputs, self.network, *args, **kwargs)
            if preds.size(-1) != targets.size(-1):
                # deep supervision mode, need to unbind feature maps first.
                preds = [preds[..., i::self.network.deep_supr_num+1] for i in range(self.network.deep_supr_num+1)]
            else:
                preds = [preds]
            engine.state.output[Keys.PRED] = preds
            del preds
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)
            engine.state.output[Keys.LOSS] = sum(
                0.5**i * self.loss_function.forward(p, targets) for i, p in enumerate(engine.state.output[Keys.PRED])
            )
            engine.fire_event(IterationEvents.LOSS_COMPLETED)

        self.network.train()
        self.optimizer.zero_grad()
        if self.amp and self.scaler is not None:
            with torch.amp.autocast('cuda'):
                _compute_pred_loss()
            self.scaler.scale(engine.state.output[Keys.LOSS]).backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            self.scaler.unscale_(self.optimizer)
            if isinstance(self.network, DistributedDataParallel):
                torch.nn.utils.clip_grad_norm_(self.network.module.parameters(), 12)
            else:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            engine.fire_event(IterationEvents.MODEL_COMPLETED)

        else:
            _compute_pred_loss()
            engine.state.output[Keys.LOSS].backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            if isinstance(self.network, DistributedDataParallel):
                torch.nn.utils.clip_grad_norm_(self.network.module.parameters(), 12)
            else:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output

class SliceInferer(SlidingWindowInferer):
    """
    SliceInferer extends SlidingWindowInferer to provide slice-by-slice (2D) inference when provided a 3D volume.
    A typical use case could be a 2D model (like 2D segmentation UNet) operates on the slices from a 3D volume,
    and the output is a 3D volume with 2D slices aggregated. Example::

        # sliding over the `spatial_dim`
        inferer = SliceInferer(roi_size=(64, 256), sw_batch_size=1, spatial_dim=1)
        output = inferer(input_volume, net)

    Args:
        spatial_dim: Spatial dimension over which the slice-by-slice inference runs on the 3D volume.
            For example ``0`` could slide over axial slices. ``1`` over coronal slices and ``2`` over sagittal slices.
        args: other optional args to be passed to the `__init__` of base class SlidingWindowInferer.
        kwargs: other optional keyword args to be passed to `__init__` of base class SlidingWindowInferer.

    Note:
        ``roi_size`` in SliceInferer is expected to be a 2D tuple when a 3D volume is provided. This allows
        sliding across slices along the 3D volume using a selected ``spatial_dim``.

    """

    def __init__(self, spatial_dim: int = 0, *args: Any, **kwargs: Any) -> None:
        self.spatial_dim = spatial_dim
        super().__init__(*args, **kwargs)
        self.orig_roi_size = ensure_tuple(self.roi_size)

    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
        """
        Args:
            inputs: 3D input for inference
            network: 2D model to execute inference on slices in the 3D input
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.
        """
        if self.spatial_dim > 2:
            raise ValueError("`spatial_dim` can only be `0, 1, 2` with `[H, W, D]` respectively.")

        # Check if ``roi_size`` tuple is 2D and ``inputs`` tensor is 3D
        self.roi_size = ensure_tuple(self.roi_size)
        if len(self.orig_roi_size) == 2 and len(inputs.shape[2:]) == 3:
            self.roi_size = list(self.orig_roi_size)
            self.roi_size.insert(self.spatial_dim, 1)
        else:
            raise RuntimeError(
                f"Currently, only 2D `roi_size` ({self.orig_roi_size}) with 3D `inputs` tensor (shape={inputs.shape}) is supported."
            )

        return super().__call__(inputs=inputs, network=lambda x: self.network_wrapper(network, x, *args, **kwargs))

    def network_wrapper(
        self,
        network: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
        x: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
        """
        Wrapper handles inference for 2D models over 3D volume inputs.
        """
        #  Pass 4D input [N, C, H, W]/[N, C, D, W]/[N, C, D, H] to the model as it is 2D.
        x = x.squeeze(dim=self.spatial_dim + 2)
        out = network(x, *args, **kwargs)

        #  Unsqueeze the network output so it is [N, C, D, H, W] as expected by the default SlidingWindowInferer class
        if isinstance(out, torch.Tensor):
            if out.dim() > 4:
                # deep supervision stacked the output feature maps along dim 1, 
                # move the deep supervision dim to the end for ease of processing
                return out.moveaxis(1, self.spatial_dim + 2)
            return out.unsqueeze(dim=self.spatial_dim + 2)

        if isinstance(out, Mapping):
            for k in out.keys():
                out[k] = out[k].unsqueeze(dim=self.spatial_dim + 2)
            return out

        return tuple(out_i.unsqueeze(dim=self.spatial_dim + 2) for out_i in out)
