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

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Iterable, Union

import torch
import torch.nn as nn
from ignite.engine import Engine
from ignite.metrics import Metric
from monai.engines import SupervisedEvaluator, EnsembleEvaluator
from monai.engines.utils import CommonKeys as Keys
from monai.engines.utils import IterationEvents, default_prepare_batch, ensure_tuple
from monai.inferers import Inferer
from monai.networks.utils import eval_mode
from monai.transforms import AsDiscrete, Transform
from torch.utils.data import DataLoader
from monai.config import KeysCollection

class DynUNetEnsembleEvaluator(EnsembleEvaluator):
    """
    Ensemble evaluation for multiple models, inherits from evaluator and Workflow.
    It accepts a list of models for inference and outputs a list of predictions for further operations.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be Iterable, typically be torch.DataLoader.
        epoch_length: number of iterations for one epoch, default to `len(val_data_loader)`.
        networks: networks to evaluate in order in the evaluator, should be regular PyTorch `torch.nn.Module`.
        pred_keys: the keys to store every prediction data.
            the length must exactly match the number of networks.
            if None, use "pred_{index}" as key corresponding to N networks, index from `0` to `N-1`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse expected data (usually `image`, `label` and other network args)
            from `engine.state.batch` for every iteration, for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.create_supervised_trainer.html.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `engine.state.batch` as inputs, return data will be stored in `engine.state.output`.
            if not provided, use `self._iteration()` instead. for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        postprocessing: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        mode: model forward mode during evaluation, should be 'eval' or 'train',
            which maps to `model.eval()` or `model.train()`, default to 'eval'.
        event_names: additional custom ignite events that will register to the engine.
            new events can be a list of str or `ignite.engine.events.EventEnum`.
        event_to_attr: a dictionary to map an event to a state attribute, then add to `engine.state`.
            for more details, check: https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html
            #ignite.engine.engine.Engine.register_events.
        decollate: whether to decollate the batch-first data to a list of data after model computation,
            recommend `decollate=True` when `postprocessing` uses components from `monai.transforms`.
            default to `True`.
        to_kwargs: dict of other args for `prepare_batch` API when converting the input data, except for
            `device`, `non_blocking`.
        amp_kwargs: dict of the args for `torch.cuda.amp.autocast()` API, for more details:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: Iterable | DataLoader,
        networks: Sequence[torch.nn.Module],
        num_classes: Union[str, int],
        pred_keys: KeysCollection | None = None,
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        inferer: Optional[Inferer] = None,
        postprocessing: Optional[Transform] = None,
        key_val_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        val_handlers: Sequence | None = None,
        amp: bool = False,
        to_cpu: bool = False,
        tta_val: bool = False,
    ) -> None:

        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            networks = networks,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=self._iteration,
            inferer=inferer,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            val_handlers=val_handlers,
            amp=amp,
        )

        if not isinstance(num_classes, int):
            num_classes = int(num_classes)
        self.num_classes = num_classes
        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.post_label = AsDiscrete(to_onehot=num_classes)
        self.tta_val = tta_val
        self.to_cpu = to_cpu
        self.networks = ensure_tuple(networks)
        self.pred_keys = (
            [f"{Keys.PRED}_{i}" for i in range(len(self.networks))] if pred_keys is None else ensure_tuple(pred_keys)
        )

        if len(self.pred_keys) != len(self.networks):
            raise ValueError("length of `pred_keys` must be same as the length of `networks`.")

    def _iteration(self, engine: Engine, batchdata: dict[str, Any]) -> Dict[str, torch.Tensor]: #-> dict:
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - pred_keys[0]: prediction result of network 0.
            - pred_keys[1]: prediction result of network 1.
            - ... ...
            - pred_keys[N]: prediction result of network N.

        Args:
            engine: `Engine` to execute operation for an iteration.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        
        batch_meta = batchdata.get("meta", None)
        if batch_meta is not None:
            del batchdata["meta"]
            engine.state.meta = batch_meta

        # Extract per-sample metadata
        batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
        if len(batch) == 2:
            inputs, targets = batch
            args: tuple = ()
            kwargs: dict = {}
        else:
            inputs, targets, args, kwargs = batch

        if self.to_cpu:
            targets = targets.cpu()

        def _compute_pred(net):
            ct = 1.0
            pred = self.inferer(inputs, net, *args, **kwargs).cpu()
            pred = nn.functional.softmax(pred, dim=1)
            if not self.tta_val:
                return pred
            else:
                for dims in [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                    flip_inputs = torch.flip(inputs, dims=dims)
                    flip_pred = torch.flip(self.inferer(flip_inputs, net).cpu(), dims=dims)
                    flip_pred = nn.functional.softmax(flip_pred, dim=1)
                    del flip_inputs
                    pred += flip_pred
                    del flip_pred
                    ct += 1
                return pred / ct
 
        engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: targets}

        for idx, network in enumerate(engine.networks):
            with eval_mode(network):
                if self.amp:
                    with torch.amp.autocast('cuda'):
                        if isinstance(engine.state.output, dict):
                            net = network
                            predictions = _compute_pred(net) 
                            predictions = self.post_pred(predictions.squeeze(0)).unsqueeze(0)
                            engine.state.output.update(
                                {engine.pred_keys[idx]: predictions}
                            )
                else:
                    if isinstance(engine.state.output, dict):
                        net = network
                        predictions = _compute_pred(net) 
                        predictions = self.post_pred(predictions.squeeze(0)).unsqueeze(0)

                        engine.state.output.update(
                            {engine.pred_keys[idx]: predictions}
                        )

            del predictions    

        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output

    def run(self, global_epoch=1):
        return super().run(global_epoch)


class DynUNetEvaluator(SupervisedEvaluator):
    """
    This class inherits from SupervisedEvaluator in MONAI, and is used with DynUNet
    on Decathlon datasets.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be
            torch.DataLoader.
        network: use the network to run model forward.
        num_classes: the number of classes (output channels) for the task.
        epoch_length: number of iterations for one epoch, default to
            `len(val_data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse image and label for current iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        postprocessing: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        tta_val: whether to do the 8 flips (8 = 2 ** 3, where 3 represents the three dimensions)
            test time augmentation, default is False.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: DataLoader,
        network: torch.nn.Module,
        num_classes: Union[str, int],
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        inferer: Optional[Inferer] = None,
        postprocessing: Optional[Transform] = None,
        key_val_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        val_handlers: Optional[Sequence] = None,
        amp: bool = False,
        tta_val: bool = False,
        output_image: bool = False
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            network=network,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            inferer=inferer,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            val_handlers=val_handlers,
            amp=amp,
        )

        if not isinstance(num_classes, int):
            num_classes = int(num_classes)
        self.num_classes = num_classes
        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.post_label = AsDiscrete(to_onehot=num_classes)
        self.tta_val = tta_val
        self.output_image = output_image

    def _iteration(self, engine: Engine, batchdata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.

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

        targets = targets.cpu()

        def _compute_pred():
            ct = 1.0
            pred = self.inferer(inputs, self.network, *args, **kwargs).cpu()
            pred = nn.functional.softmax(pred, dim=1)
            if not self.tta_val:
                return pred
            else:
                for dims in [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                    flip_inputs = torch.flip(inputs, dims=dims)
                    flip_pred = torch.flip(self.inferer(flip_inputs, self.network).cpu(), dims=dims)
                    flip_pred = nn.functional.softmax(flip_pred, dim=1)
                    del flip_inputs
                    pred += flip_pred
                    del flip_pred
                    ct += 1
                return pred / ct

        # execute forward computation
        with eval_mode(self.network):
            if self.amp:
                with torch.amp.autocast('cuda'):
                    predictions = _compute_pred()
            else:
                predictions = _compute_pred()


        inputs = inputs.cpu()

        predictions = self.post_pred(predictions.squeeze(0)).unsqueeze(0)

        engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: targets, Keys.PRED: predictions}      

        del predictions

        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output
    