import torch
from typing import List, Optional, Union
from leela_interp import Lc0sight, LeelaBoard
from .leela_types import AllowedOutputs, ALLOWED_OUTPUTS

# TODO: Maybe implement batching directly into the forward functions?


class LeelaLogitLens(torch.nn.Module):
    def __init__(self, model: Lc0sight):
        """
        Initialize the logit lens with the given Lc0sight model.

        This implementation supports the zero ablation method described in
        "Evidence of Learned Look-Ahead in a Chess-Playing Neural Network" by Jenner et al.
        """
        super().__init__()
        self.model = model
        self.num_layers = model.N_LAYERS
        self.hidden_dim = model.D_MODEL
        self.num_tokens = 64

    @torch.no_grad()
    def forward(
            self,
            boards: List[LeelaBoard],
            layer_idx: Optional[int] = None,
            output: Optional[AllowedOutputs] = None,
            return_probs: bool = True,
            return_policy_as_dict: bool = False,
    ) -> List[dict]:
        """
        Simple zero ablation of all layers from layer_idx onwards (main paper method).

        This method implements the standard approach used in the paper: zeroing out
        all transformer layers from the specified index onwards, then reading the
        policy/value heads.

        Args:
            boards: List of LeelaBoard objects to analyze
            layer_idx: Index from which to start ablation (0-based). If None, uses full model.
            output: Which outputs to return ("policy", "win_draw_loose", "moves_left")
            return_probs: Convert policy logits to probabilities
            return_policy_as_dict: Return policy as move->probability dict

        Returns:
            List of dictionaries, one per board, containing requested outputs
        """
        return self._forward_with_ablation(
            boards=boards,
            layer_idx=layer_idx,
            output=output,
            return_probs=return_probs,
            return_policy_as_dict=return_policy_as_dict,
            # Use simple zero ablation (paper's main method)
            set_ln_bias_zero=True,
            set_ffn_bias_zero=True,
            set_attn_output_bias_zero=True,
            set_attn_value_bias_zero=True,
            set_unbiased_ffn_zero=True,
            set_unbiased_attn_zero=True,
            keep_alpha_scaling=True,
            keep_ln_scaling=True,
        )

    @torch.no_grad()
    def forward_advanced(
            self,
            boards: List[LeelaBoard],
            layer_idx: Optional[int] = None,
            output: Optional[AllowedOutputs] = None,
            return_probs: bool = True,
            return_policy_as_dict: bool = False,
            project_after_attention: bool = False,
            set_ln_bias_zero: bool = True,
            set_ffn_bias_zero: bool = True,
            set_attn_output_bias_zero: bool = True,
            set_attn_value_bias_zero: bool = True,
            set_unbiased_ffn_zero: bool = True,
            set_unbiased_attn_zero: bool = True,
            keep_alpha_scaling: bool = True,
            keep_ln_scaling: bool = True,
    ) -> List[dict]:
        """
        Advanced ablation with fine-grained control over bias handling.

        This method provides all the ablation options explored during research,
        allowing precise control over which components are zeroed out.

        Most users should use the simple forward() method instead.
        """
        return self._forward_with_ablation(
            boards=boards,
            layer_idx=layer_idx,
            output=output,
            return_probs=return_probs,
            return_policy_as_dict=return_policy_as_dict,
            project_after_attention=project_after_attention,
            set_ln_bias_zero=set_ln_bias_zero,
            set_ffn_bias_zero=set_ffn_bias_zero,
            set_attn_output_bias_zero=set_attn_output_bias_zero,
            set_attn_value_bias_zero=set_attn_value_bias_zero,
            set_unbiased_ffn_zero=set_unbiased_ffn_zero,
            set_unbiased_attn_zero=set_unbiased_attn_zero,
            keep_alpha_scaling=keep_alpha_scaling,
            keep_ln_scaling=keep_ln_scaling,
        )

    def multi_layer_lens(
            self,
            boards: List[LeelaBoard],
            layer_indices: Optional[List[int]] = None,
            output: Optional[AllowedOutputs] = None,
            return_probs: bool = True,
            return_policy_as_dict: bool = False,
            use_advanced: bool = False,
            **kwargs
    ) -> List[dict]:
        """
        Analyze multiple layers at once.

        Args:
            boards: List of LeelaBoard objects to analyze
            layer_indices: List of layer indices to analyze. If None, analyzes all layers (0 to num_layers)
            output: Which outputs to return ("policy", "win_draw_loose", "moves_left")
            return_probs: Convert policy logits to probabilities
            return_policy_as_dict: Return policy as move->probability dict
            use_advanced: If True, use forward_advanced() with **kwargs.
                         If False, use simple forward() method.
            **kwargs: Additional arguments passed to forward_advanced() when use_advanced=True

        Returns:
            List of dictionaries, one per board, each containing:
            - "board": the original LeelaBoard
            - "layers": dict mapping layer_idx -> results for that layer
        """
        # Convert a single board to a list
        if isinstance(boards, LeelaBoard):
            boards = [boards]
        elif not isinstance(boards, List):
            raise TypeError("boards must be a LeelaBoard or a list of LeelaBoard objects.")
        else:
            if not isinstance(boards[0], LeelaBoard):
                raise TypeError("boards must be a LeelaBoard or a list of LeelaBoard objects.")

        if layer_indices is None:
            layer_indices = list(range(self.num_layers + 1))

        # Initialize results structure
        final_results = [{"board": b, "layers": {}} for b in boards]

        # Analyze each layer
        for layer_idx in layer_indices:
            if use_advanced:
                layer_output = self.forward_advanced(
                    boards=boards,
                    layer_idx=layer_idx,
                    output=output,
                    return_probs=return_probs,
                    return_policy_as_dict=return_policy_as_dict,
                    **kwargs
                )
            else:
                layer_output = self.forward(
                    boards=boards,
                    layer_idx=layer_idx,
                    output=output,
                    return_probs=return_probs,
                    return_policy_as_dict=return_policy_as_dict,
                )

            # Store results for this layer
            for i, subres in enumerate(layer_output):
                final_results[i]["layers"][layer_idx] = subres

        return final_results

    def _forward_with_ablation(
            self,
            boards: List[LeelaBoard],
            layer_idx: Optional[int] = None,
            output: Optional[AllowedOutputs] = None,
            return_probs: bool = True,
            return_policy_as_dict: bool = False,
            project_after_attention: bool = False,
            set_ln_bias_zero: bool = True,
            set_ffn_bias_zero: bool = True,
            set_attn_output_bias_zero: bool = True,
            set_attn_value_bias_zero: bool = True,
            set_unbiased_ffn_zero: bool = True,
            set_unbiased_attn_zero: bool = True,
            keep_alpha_scaling: bool = True,
            keep_ln_scaling: bool = True,
    ) -> List[dict]:
        """
        From layer_idx onwards, zero out the specified components based on the parameters.
        Then compute and return the requested heads (policy, win_draw_loose, moves_left)
        as a list of dicts, one for each board.
        """
        # --- 1. Handle arguments and pre-checks ---
        if output is None:
            output = ["policy", "win_draw_loose", "moves_left"]
        elif isinstance(output, str):
            output = [output]
        else:
            for o in output:
                if o not in ALLOWED_OUTPUTS:
                    raise ValueError(f"Invalid output type: {o}. Allowed outputs are: {ALLOWED_OUTPUTS}")

        # Convert a single board to a list
        if isinstance(boards, LeelaBoard):
            boards = [boards]
        elif not isinstance(boards, List):
            raise TypeError("boards must be a LeelaBoard or a list of LeelaBoard objects.")
        else:
            if not isinstance(boards[0], LeelaBoard):
                raise TypeError("boards must be a LeelaBoard or a list of LeelaBoard objects.")

        batch_size = len(boards)

        if layer_idx is None:
            layer_idx = self.num_layers
        if layer_idx < 0 or layer_idx > self.num_layers:
            raise ValueError(
                f"Invalid layer index: {layer_idx}. Model has {self.num_layers} layers. Use zero based indexing.")

        # Not really needed since you can simply either use the residual before or after the alpha scaling
        with (self.model.trace(boards)):
            alpha = getattr(self.model._lc0_model, 'encoder0/alpha*input').input[0][1].save()
        alpha = alpha.item()

        # --- 2. Do ablation via nnsight tracing context ---
        with (self.model.trace(boards)):

            for j in range(layer_idx, self.num_layers):
                # Attention modules

                # Residual state before attention sublayer
                if j == 0:
                    residual_pre_mha = getattr(self.model._lc0_model, "attn_body/ma_gating/rehape2")
                else:
                    residual_pre_mha = getattr(self.model._lc0_model, f"encoder{j-1}/ln2")

                mha_scaled_residual = getattr(self.model._lc0_model, f"encoder{j}/alpha*input")

                attention_output_pre_output_bias = getattr(self.model._lc0_model, f"encoder{j}/mha/out/dense/w")
                mha_output_post_bias = getattr(self.model._lc0_model, f"encoder{j}/mha/out/dense/b")
                mha_value_projection_bias = getattr(self.model._lc0_model, f'encoder{j}/mha/V/b').input[0][1]
                mha_out_projection_weights = getattr(self.model._lc0_model, f'encoder{j}/mha/out/dense/w').input[0][1]
                mha_out_projection_bias = getattr(self.model._lc0_model, f'encoder{j}/mha/out/dense/b').input[0][1]

                # Residual state after attention sublayer (attention output added)
                residual_post_mha = getattr(self.model._lc0_model, f'encoder{j}/mha/out/skip')

                # If you want to project from the attention sublayer don't ablate the specified layer
                if j != layer_idx or not project_after_attention:
                    if keep_alpha_scaling:
                        # If you want to set the unbiased attention output to zero (attention output - attention biases)
                        if set_unbiased_attn_zero:
                            # And the attention output bias to zero
                            if set_attn_output_bias_zero:
                                # And the attention value bias to zero
                                if set_attn_value_bias_zero:
                                    # Set the whole attention module output to zero
                                    residual_post_mha.output = mha_scaled_residual.output
                                # Don't set the value bias to zero -> Set the attention output to the projected value bias
                                else:
                                    residual_post_mha.output = mha_scaled_residual.output + (
                                            mha_value_projection_bias @ mha_out_projection_weights).view(1, self.hidden_dim).repeat(batch_size * self.num_tokens, 1)
                            # Don't set the attention output bias to zero
                            else:
                                # And the attention value bias to zero -> Set output to output bias
                                if set_attn_value_bias_zero:
                                    residual_post_mha.output = mha_scaled_residual.output + (mha_out_projection_bias).view(1, self.hidden_dim).repeat(batch_size * self.num_tokens, 1)
                                # Don't set the attention value bias to zero -> Set output to output bias + projected value bias
                                else:
                                    residual_post_mha.output = mha_scaled_residual.output + (mha_out_projection_bias + (
                                            mha_value_projection_bias @ mha_out_projection_weights)).view(1, self.hidden_dim).repeat(batch_size * self.num_tokens, 1)
                        # Don't set the unbiased attention output to zero
                        else:
                            # And the attention output bias to zero
                            if set_attn_output_bias_zero:
                                # And the attention value bias to zero
                                if set_attn_value_bias_zero:
                                    # Set the attention module output to the unbiased output
                                    residual_post_mha.output = mha_scaled_residual.output + mha_output_post_bias.output - (
                                            mha_out_projection_bias + (
                                            mha_value_projection_bias @ mha_out_projection_weights)).view(1, self.hidden_dim).repeat(batch_size * self.num_tokens, 1)
                                # Don't set the value bias to zero -> Set the attention output to the unbiased output plus the projected value bias
                                else:
                                    residual_post_mha.output = mha_scaled_residual.output + mha_output_post_bias.output - (
                                        mha_out_projection_bias).view(1, self.hidden_dim).repeat(
                                        batch_size * self.num_tokens, 1)
                            # Don't set the attention output bias to zero
                            else:
                                # And the attention value bias to zero -> Set output to the unbiased output plus the output bias
                                if set_attn_value_bias_zero:
                                    residual_post_mha.output = mha_scaled_residual.output + mha_output_post_bias.output - (
                                            mha_value_projection_bias @ mha_out_projection_weights).view(1, self.hidden_dim).repeat(batch_size * self.num_tokens, 1)
                                # Don't set the attention value bias to zero -> Set output to original output
                                else:
                                    residual_post_mha.output = mha_scaled_residual.output + mha_output_post_bias.output
                    else:
                        # If you want to set the unbiased attention output to zero (attention output - attention biases)
                        if set_unbiased_attn_zero:
                            # And the attention output bias to zero
                            if set_attn_output_bias_zero:
                                # And the attention value bias to zero
                                if set_attn_value_bias_zero:
                                    # Set the whole attention module output to zero
                                    residual_post_mha.output = residual_pre_mha.output
                                # Don't set the value bias to zero -> Set the attention output to the projected value bias
                                else:
                                    residual_post_mha.output = residual_pre_mha.output + (
                                            mha_value_projection_bias @ mha_out_projection_weights).view(1,
                                                                                                         self.hidden_dim).repeat(
                                        batch_size * self.num_tokens, 1)
                            # Don't set the attention output bias to zero
                            else:
                                # And the attention value bias to zero -> Set output to output bias
                                if set_attn_value_bias_zero:
                                    residual_post_mha.output = residual_pre_mha.output + (mha_out_projection_bias).view(1,
                                                                                                             self.hidden_dim).repeat(
                                        batch_size * self.num_tokens, 1)
                                # Don't set the attention value bias to zero -> Set output to output bias + projected value bias
                                else:
                                    residual_post_mha.output = residual_pre_mha.output + (mha_out_projection_bias + (
                                            mha_value_projection_bias @ mha_out_projection_weights)).view(1,
                                                                                                          self.hidden_dim).repeat(
                                        batch_size * self.num_tokens, 1)
                        # Don't set the unbiased attention output to zero
                        else:
                            # And the attention output bias to zero
                            if set_attn_output_bias_zero:
                                # And the attention value bias to zero
                                if set_attn_value_bias_zero:
                                    # Set the attention module output to the unbiased output
                                    residual_post_mha.output = residual_pre_mha.output + mha_output_post_bias.output - (
                                            mha_out_projection_bias + (
                                            mha_value_projection_bias @ mha_out_projection_weights)).view(1,
                                                                                                          self.hidden_dim).repeat(
                                        batch_size * self.num_tokens, 1)
                                # Don't set the value bias to zero -> Set the attention output to the unbiased output plus the projected value bias
                                else:
                                    residual_post_mha.output = residual_pre_mha.output + mha_output_post_bias.output - (
                                        mha_out_projection_bias).view(1, self.hidden_dim).repeat(
                                        batch_size * self.num_tokens, 1)
                            # Don't set the attention output bias to zero
                            else:
                                # And the attention value bias to zero -> Set output to the unbiased output plus the output bias
                                if set_attn_value_bias_zero:
                                    residual_post_mha.output = residual_pre_mha.output + mha_output_post_bias.output - (
                                            mha_value_projection_bias @ mha_out_projection_weights).view(1,
                                                                                                         self.hidden_dim).repeat(
                                        batch_size * self.num_tokens, 1)
                                # Don't set the attention value bias to zero -> Set output to original output
                                else:
                                    residual_post_mha.output = residual_pre_mha.output + mha_output_post_bias.output

                # LN 1 modules
                first_ln = getattr(self.model._lc0_model, f'encoder{j}/ln1')
                first_ln_bias = first_ln.bias
                first_ln_weight = first_ln.weight

                if not keep_ln_scaling:
                    if j != layer_idx or not project_after_attention:
                        first_ln.output = (first_ln.output - first_ln_bias.view(1, self.hidden_dim).repeat(
                            batch_size * self.num_tokens, 1)) / first_ln_weight.view(1, self.hidden_dim).repeat(
                            batch_size * self.num_tokens, 1)

                if not set_ln_bias_zero:
                    # Subtract LN biases after adding  -> a bit hacky
                    if j != layer_idx or not project_after_attention:
                        first_ln.output = first_ln.output + first_ln_bias.view(1, self.hidden_dim).repeat(
                            batch_size * self.num_tokens, 1)

                # FFN modules

                # Residual state before FFN sublayer
                residual_pre_ffn = getattr(self.model._lc0_model, f'encoder{j}/ln1')

                ffn_scaled_residual = getattr(self.model._lc0_model, f"encoder{j}/alpha*out1")


                ffn_output_pre_bias = getattr(self.model._lc0_model, f"encoder{j}/ffn/dense2/w")
                ffn_output_post_bias = getattr(self.model._lc0_model, f"encoder{j}/ffn/dense2/b")
                second_ffn_bias = getattr(self.model._lc0_model, f'encoder{j}/ffn/dense2/b').input[0][1]

                # Residual state after FFN sublayer
                residual_post_ffn = getattr(self.model._lc0_model, f'encoder{j}/ffn/skip')

                # Whether to keep the DeepNorm alpha scaling on the residual
                if keep_alpha_scaling:
                    # If you want to set the unbiased ffn output to zero (ffn output - ffn biases)
                    if set_unbiased_ffn_zero:
                        # And the ffn output bias to zero
                        if set_ffn_bias_zero:
                            # Ablate the whole FFN sublayer (output + bias) and only keep the scaled residual
                            residual_post_ffn.output = ffn_scaled_residual.output
                        # Don't set the ffn output bias to zero -> Set the ffn output to the ffn output bias
                        else:
                            residual_post_ffn.output = ffn_scaled_residual.output + second_ffn_bias.view(1,
                                                                                                self.hidden_dim).repeat(
                                batch_size * self.num_tokens, 1)
                    # Don't set the unbiased ffn output to zero
                    else:
                        # And the ffn output bias to zero
                        if set_ffn_bias_zero:
                            # Set the whole ffn module output to the unbiased ffn output
                            residual_post_ffn.output = ffn_scaled_residual.output + (
                                    ffn_output_post_bias.output - second_ffn_bias.view(1, self.hidden_dim).repeat(
                                batch_size * self.num_tokens, 1))
                        # Don't set the ffn output bias to zero -> Set the ffn output to the original output
                        else:
                            residual_post_ffn.output = ffn_scaled_residual.output + ffn_output_post_bias.output
                else:
                    # If you want to set the unbiased ffn output to zero (ffn output - ffn biases)
                    if set_unbiased_ffn_zero:
                        # And the ffn output bias to zero
                        if set_ffn_bias_zero:
                            # Ablate the whole FFN sublayer (output + bias) and only keep the unscaled residual
                            residual_post_ffn.output = residual_pre_ffn.output
                        # Don't set the ffn output bias to zero -> Set the ffn output to the ffn output bias
                        else:
                            residual_post_ffn.output = residual_pre_ffn.output + second_ffn_bias.view(1, self.hidden_dim).repeat(
                                batch_size * self.num_tokens, 1)
                    # Don't set the unbiased ffn output to zero
                    else:
                        # And the ffn output bias to zero
                        if set_ffn_bias_zero:
                            # Set the whole ffn module output to the unbiased ffn output
                            residual_post_ffn.output = residual_pre_ffn.output + (
                                    ffn_output_post_bias.output - second_ffn_bias.view(1, self.hidden_dim).repeat(
                                batch_size * self.num_tokens, 1))
                        # Don't set the ffn output bias to zero -> Set the ffn output to the original output
                        else:
                            residual_post_ffn.output = residual_pre_ffn.output + ffn_output_post_bias.output

                # Second LN modules
                second_ln = getattr(self.model._lc0_model, f'encoder{j}/ln2')
                second_ln_bias = second_ln.bias
                second_ln_weight = second_ln.weight

                if not keep_ln_scaling:
                        second_ln.output = (second_ln.output - second_ln_bias.view(1, self.hidden_dim).repeat(
                            batch_size * self.num_tokens, 1)) / second_ln_weight.view(1, self.hidden_dim).repeat(
                            batch_size * self.num_tokens, 1)

                if not set_ln_bias_zero:
                    # Subtract LN biases after adding  -> a bit hacky
                    second_ln.output = second_ln.output + second_ln_bias.view(1, self.hidden_dim).repeat(
                        batch_size * self.num_tokens, 1)

            # --- 3. Gather outputs from the model ---
            results = {}
            if "policy" in output:
                policy_layer = getattr(self.model._lc0_model, "output/policy")
                policy_logits = policy_layer.output.save()  # shape: (B, 1858)
                results["policy"] = policy_logits
            if "win_draw_loose" in output:
                wdl_layer = getattr(self.model._lc0_model, "output/wdl")
                wdl_logits = wdl_layer.output.save()  # shape: (B, 3)
                results["win_draw_loose"] = wdl_logits
            if "moves_left" in output:
                mlh_layer = getattr(self.model._lc0_model, "output/mlh")
                mlh_logits = mlh_layer.output.save()  # shape: (B, N) or (B, 1), etc.
                results["moves_left"] = mlh_logits

        # --- 4. Convert policy logits to probabilities if needed ---
        if return_probs and "policy" in results:
            results["policy"] = self.model.logits_to_probs(boards, results["policy"])

        # --- 5. Build final list of per-board dicts ---
        final_results = []
        for i, board in enumerate(boards):
            entry = {"board": board}
            if "policy" in results:
                # row is shape (1858,)
                row = results["policy"][i].squeeze(0)  # in case it had shape (1,1858)
                entry["policy"] = row
                if return_policy_as_dict:
                    entry["policy_as_dict"] = self.model.policy_as_dict(board, row)

            if "win_draw_loose" in results:
                entry["win_draw_loose"] = results["win_draw_loose"][i]
            if "moves_left" in results:
                entry["moves_left"] = results["moves_left"][i]

            final_results.append(entry)

        return final_results
