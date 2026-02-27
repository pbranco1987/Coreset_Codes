"""
Variational Autoencoder for tabular data.

Contains:
- TabularVAE: Legacy MSE-only VAE architecture (imported from _vae_networks)
- MixedTypeVAE: Product-of-likelihoods VAE (imported from _vae_networks)
- VAETrainer: Training and embedding utilities with automatic dispatch
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ._vae_networks import (
    ColumnSpec,
    DecoderOutput,
    MixedTypeVAE,
    TabularVAE,
)


class VAETrainer:
    """
    Trainer for TabularVAE and MixedTypeVAE.

    Handles training loop, loss computation, and embedding extraction.
    Automatically dispatches between MSE-only (legacy) and mixed-type
    (product-of-likelihoods) training depending on whether a
    :class:`ColumnSpec` is provided and ``cfg.use_mixed_likelihood`` is True.
    """

    def __init__(
        self,
        cfg,  # VAEConfig
        seed: int,
        device: torch.device,
        column_spec: Optional[ColumnSpec] = None,
    ):
        self.cfg = cfg
        self.seed = seed
        self.device = device
        self.column_spec = column_spec
        self.model: Optional[Union[TabularVAE, MixedTypeVAE]] = None

        # Decide whether to use mixed-type likelihood path
        self.use_mixed: bool = (
            column_spec is not None
            and bool(getattr(cfg, "use_mixed_likelihood", False))
        )

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # ------------------------------------------------------------------
    # KL weight with optional linear warmup
    # ------------------------------------------------------------------

    def _kl_weight(self, epoch: int) -> float:
        """Return the KL weight for a given epoch (linear warmup)."""
        base = float(self.cfg.kl_weight)
        warmup = int(getattr(self.cfg, "kl_warmup_epochs", 0))
        if warmup <= 0 or epoch >= warmup:
            return base
        return base * (epoch / warmup)

    # ------------------------------------------------------------------
    # Mixed-type reconstruction loss
    # ------------------------------------------------------------------

    def _mixed_loss(
        self,
        x: torch.Tensor,
        dec_out: DecoderOutput,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """Product-of-likelihoods reconstruction + KL with warmup.

        Loss = (1/B) * [ mean-per-block-losses ] + beta(epoch) * KL

        Per-type block losses (averaged over columns within block, then
        averaged across the active blocks so that each *type family*
        contributes equally regardless of column count):

        * **Gaussian** (continuous): MSE  (≡ NLL with sigma²=1 up to const)
        * **Bernoulli** (binary):    BCE with logits
        * **Categorical** (K>2):     CE per head, averaged
        """
        cs = self.column_spec
        assert cs is not None

        blocks: List[torch.Tensor] = []

        # --- Continuous (Gaussian NLL → MSE, σ²=1) ---
        if cs.n_continuous > 0:
            target_cont = x[:, cs.continuous_idx]
            # F.mse_loss with reduction="mean" averages over all elements
            loss_cont = F.mse_loss(dec_out.continuous, target_cont, reduction="mean")
            blocks.append(loss_cont)

        # --- Binary (Bernoulli → BCE with logits) ---
        if cs.n_binary > 0:
            target_bin = x[:, cs.binary_idx]
            loss_bin = F.binary_cross_entropy_with_logits(
                dec_out.binary, target_bin, reduction="mean",
            )
            blocks.append(loss_bin)

        # --- Multi-class categorical (CE per head) ---
        if cs.n_categorical > 0:
            cat_losses: List[torch.Tensor] = []
            for i, (col_idx, _K_j) in enumerate(cs.categorical_specs):
                target_ids = x[:, col_idx].long()
                logits = dec_out.categorical[i]  # (B, K_j)
                cat_losses.append(F.cross_entropy(logits, target_ids, reduction="mean"))
            # Average across categorical heads
            loss_cat = torch.stack(cat_losses).mean()
            blocks.append(loss_cat)

        # Block averaging: equal weight per type family
        recon_loss = torch.stack(blocks).mean()

        # KL divergence with warmup
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        beta = self._kl_weight(epoch)

        return recon_loss + beta * kl_loss

    # ------------------------------------------------------------------
    # Legacy MSE loss (inline for speed, but also available as method)
    # ------------------------------------------------------------------

    def _mse_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """Legacy MSE reconstruction + KL with warmup."""
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        beta = self._kl_weight(epoch)
        return recon_loss + beta * kl_loss

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        X_val: Optional[np.ndarray] = None,
    ) -> Union[TabularVAE, MixedTypeVAE]:
        """
        Train the VAE.

        Performance notes
        -----------------
        * For small datasets (N <= 50k), uses **full-batch** training which
          eliminates the inner Python loop entirely (1 forward + 1 backward
          per epoch instead of ceil(N/batch_size)).
        * ``torch.compile`` is used when available (PyTorch >= 2.0) for
          fused-kernel acceleration.
        * All debug instrumentation has been removed from the hot loop;
          logging happens only every ``log_every`` epochs.
        """
        import os

        # Enforce thread limit
        vae_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
        torch.set_num_threads(vae_threads)
        try:
            torch.set_num_interop_threads(vae_threads)
        except RuntimeError:
            pass  # Can only be set once per process

        # Ensure float32 + contiguous for zero-copy torch.from_numpy
        X = np.asarray(X, dtype=np.float32)
        if not X.flags["C_CONTIGUOUS"]:
            X = np.ascontiguousarray(X)

        input_dim = int(X.shape[1])
        n_train = int(X.shape[0])

        # ---- Create model (dispatch) ----
        if self.use_mixed and self.column_spec is not None:
            self.model = MixedTypeVAE(
                column_spec=self.column_spec,
                latent_dim=int(self.cfg.latent_dim),
                hidden_dim=int(self.cfg.hidden_dim),
                cat_embedding_dim=int(getattr(self.cfg, "cat_embedding_dim", 16)),
            ).to(self.device)
            model_tag = "MixedTypeVAE"
        else:
            self.model = TabularVAE(
                input_dim=input_dim,
                latent_dim=int(self.cfg.latent_dim),
                hidden_dim=int(self.cfg.hidden_dim),
            ).to(self.device)
            model_tag = "TabularVAE"

        # torch.compile (PyTorch 2.x) — significant CPU speedup
        compiled_model = self.model
        use_compiled = False
        try:
            if hasattr(torch, "compile"):
                compiled_model = torch.compile(self.model, mode="reduce-overhead")
                use_compiled = True
        except Exception:
            compiled_model = self.model

        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.cfg.lr))

        # Config knobs (backwards-compatible)
        epochs = int(getattr(self.cfg, "epochs", 0))
        batch_size = int(getattr(self.cfg, "batch_size", 256))
        if batch_size <= 0:
            batch_size = n_train

        # Full-batch when dataset is small (eliminates inner Python loop).
        use_full_batch = n_train <= max(batch_size, 50_000)

        log_every = int(getattr(self.cfg, "log_every", 10))
        if log_every <= 0:
            log_every = 10

        val_every = int(getattr(self.cfg, "val_every", log_every))
        if val_every <= 0:
            val_every = log_every

        early_stop_patience = int(getattr(self.cfg, "early_stopping_patience", 0))

        # CUDA speed knobs
        use_amp = False
        if self.device.type == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            try:
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            except Exception:
                pass
            # Automatic mixed precision — ~1.5-2× speedup on Volta+
            use_amp = True

        # Move training data to target device ONCE
        X_train_t = torch.from_numpy(X)
        if self.device.type == "cuda":
            X_train_t = X_train_t.pin_memory().to(self.device, non_blocking=True)
        else:
            X_train_t = X_train_t.to(self.device)

        X_val_t: Optional[torch.Tensor] = None
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            if not X_val.flags["C_CONTIGUOUS"]:
                X_val = np.ascontiguousarray(X_val)
            X_val_t = torch.from_numpy(X_val)
            if self.device.type == "cuda":
                X_val_t = X_val_t.pin_memory().to(self.device, non_blocking=True)
            else:
                X_val_t = X_val_t.to(self.device)

        # AMP scaler for mixed-precision training
        scaler = torch.amp.GradScaler(enabled=use_amp)

        n_params = sum(p.numel() for p in self.model.parameters())
        warmup_epochs = int(getattr(self.cfg, "kl_warmup_epochs", 0))
        print(
            f"[VAE] Training {model_tag}: {epochs} epochs, N={n_train}, D={input_dim}, "
            f"params={n_params:,}, device={self.device}, "
            f"{'full-batch' if use_full_batch else 'batch_size=' + str(batch_size)}"
            f"{', compiled' if use_compiled else ''}"
            f"{', amp' if use_amp else ''}"
            f"{', KL-warmup=' + str(warmup_epochs) if warmup_epochs > 0 else ''}",
            flush=True,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        last_val_loss: Optional[float] = None
        t0 = time.time()

        # ---- Dispatch: mixed-type vs. legacy MSE ----
        use_mixed = self.use_mixed

        if use_full_batch:
            # ===========================================================
            # FAST PATH: full-batch (no inner loop, no shuffling needed)
            # ===========================================================
            for epoch in range(epochs):
                compiled_model.train()
                with torch.amp.autocast(self.device.type, enabled=use_amp):
                    if use_mixed:
                        dec_out, mu, logvar = compiled_model(X_train_t)
                        loss = self._mixed_loss(X_train_t, dec_out, mu, logvar, epoch)
                    else:
                        recon, mu, logvar = compiled_model(X_train_t)
                        loss = self._mse_loss(X_train_t, recon, mu, logvar, epoch)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # --- logging (infrequent) ---
                if epoch % log_every == 0 or epoch == epochs - 1:
                    train_loss = float(loss.detach().item())
                    elapsed = time.time() - t0
                    msg = f"Epoch {epoch}: train_loss={train_loss:.4f}"
                    if last_val_loss is not None:
                        msg += f", val_loss={last_val_loss:.4f}"
                    msg += f" ({elapsed:.1f}s)"
                    print(msg, flush=True)

                # --- validation ---
                if X_val_t is not None and (
                    early_stop_patience > 0
                    or epoch % val_every == 0
                    or epoch == epochs - 1
                ):
                    last_val_loss = self._evaluate_tensor(
                        X_val_t, batch_size=n_train, epoch=epoch,
                    )
                    if early_stop_patience > 0:
                        if last_val_loss < best_val_loss:
                            best_val_loss = last_val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= early_stop_patience:
                                print(f"Early stopping at epoch {epoch}", flush=True)
                                break
        else:
            # ===========================================================
            # MINI-BATCH PATH (large datasets)
            # ===========================================================
            for epoch in range(epochs):
                compiled_model.train()
                perm = torch.randperm(n_train, device=X_train_t.device)
                epoch_loss = torch.zeros((), device=X_train_t.device)

                for start in range(0, n_train, batch_size):
                    batch = X_train_t[perm[start : start + batch_size]]

                    with torch.amp.autocast(self.device.type, enabled=use_amp):
                        if use_mixed:
                            dec_out, mu, logvar = compiled_model(batch)
                            loss = self._mixed_loss(batch, dec_out, mu, logvar, epoch)
                        else:
                            recon, mu, logvar = compiled_model(batch)
                            loss = self._mse_loss(batch, recon, mu, logvar, epoch)

                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss = epoch_loss + loss.detach() * batch.size(0)

                # --- logging ---
                if epoch % log_every == 0 or epoch == epochs - 1:
                    train_loss = float((epoch_loss / n_train).item())
                    elapsed = time.time() - t0
                    msg = f"Epoch {epoch}: train_loss={train_loss:.4f}"
                    if last_val_loss is not None:
                        msg += f", val_loss={last_val_loss:.4f}"
                    msg += f" ({elapsed:.1f}s)"
                    print(msg, flush=True)

                # --- validation ---
                if X_val_t is not None and (
                    early_stop_patience > 0
                    or epoch % val_every == 0
                    or epoch == epochs - 1
                ):
                    last_val_loss = self._evaluate_tensor(
                        X_val_t, batch_size=batch_size, epoch=epoch,
                    )
                    if early_stop_patience > 0:
                        if last_val_loss < best_val_loss:
                            best_val_loss = last_val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= early_stop_patience:
                                print(f"Early stopping at epoch {epoch}", flush=True)
                                break

        elapsed = time.time() - t0
        print(f"[VAE] Training complete ({elapsed:.1f}s, {epochs} epochs)", flush=True)
        return self.model

    # ------------------------------------------------------------------
    # Legacy loss (kept for backward compatibility with external callers)
    # ------------------------------------------------------------------

    def _vae_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Compute legacy VAE loss (MSE + KL, no warmup)."""
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.cfg.kl_weight * kl_loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_tensor(
        self,
        X_tensor: torch.Tensor,
        *,
        batch_size: int,
        epoch: int = -1,
    ) -> float:
        """Evaluate the model on an in-memory tensor already on the correct device."""
        self.model.eval()
        n = int(X_tensor.shape[0])
        if n == 0:
            return float("nan")
        if batch_size <= 0:
            batch_size = n

        total_loss = torch.zeros((), device=X_tensor.device)
        use_mixed = self.use_mixed

        with torch.inference_mode():
            for start in range(0, n, batch_size):
                batch = X_tensor[start : start + batch_size]
                if use_mixed:
                    dec_out, mu, logvar = self.model(batch)
                    loss = self._mixed_loss(batch, dec_out, mu, logvar, epoch)
                else:
                    recon, mu, logvar = self.model(batch)
                    loss = self._mse_loss(batch, recon, mu, logvar, epoch)
                total_loss = total_loss + loss.detach() * batch.size(0)

        return float((total_loss / n).item())

    def _evaluate(self, loader: DataLoader) -> float:
        """Evaluate model on a data loader (legacy MSE path only)."""
        self.model.eval()
        total_loss = torch.zeros((), device=self.device)
        n_samples = 0

        with torch.inference_mode():
            for (batch,) in loader:
                batch = batch.to(self.device, non_blocking=(self.device.type == "cuda"))
                recon, mu, logvar = self.model(batch)
                loss = self._vae_loss(batch, recon, mu, logvar)
                total_loss = total_loss + loss.detach() * batch.size(0)
                n_samples += int(batch.size(0))

        return float((total_loss / max(1, n_samples)).item())

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def embed(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get latent embeddings for data.

        Works identically for both TabularVAE and MixedTypeVAE, since
        both expose an ``.encode(x) -> (mu, logvar)`` interface.

        Parameters
        ----------
        X : np.ndarray
            Input data, shape (N, d)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (mu, logvar) latent parameters, each shape (N, latent_dim)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = np.asarray(X, dtype=np.float32)
        if not X.flags["C_CONTIGUOUS"]:
            X = np.ascontiguousarray(X)

        X_tensor = torch.from_numpy(X)
        if self.device.type == "cuda":
            X_tensor = X_tensor.pin_memory().to(self.device, non_blocking=True)
        else:
            X_tensor = X_tensor.to(self.device)

        n = int(X_tensor.shape[0])
        batch_size = int(getattr(self.cfg, "embed_batch_size", getattr(self.cfg, "batch_size", 1024)))
        if batch_size <= 0:
            batch_size = n

        self.model.eval()
        mus: list[torch.Tensor] = []
        logvars: list[torch.Tensor] = []

        with torch.inference_mode():
            for start in range(0, n, batch_size):
                batch = X_tensor[start : start + batch_size]
                mu, logvar = self.model.encode(batch)
                mus.append(mu.detach().cpu())
                logvars.append(logvar.detach().cpu())

        mu_all = torch.cat(mus, dim=0).numpy()
        logvar_all = torch.cat(logvars, dim=0).numpy()

        return mu_all, logvar_all
