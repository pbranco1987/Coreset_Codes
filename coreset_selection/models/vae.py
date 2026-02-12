"""
Variational Autoencoder for tabular data.

Contains:
- TabularVAE: VAE neural network architecture (imported from _vae_networks)
- VAETrainer: Training and embedding utilities
"""

from __future__ import annotations

from typing import Optional, Tuple

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ._vae_networks import TabularVAE


class VAETrainer:
    """
    Trainer for TabularVAE.

    Handles training loop, loss computation, and embedding extraction.
    """

    def __init__(
        self,
        cfg,  # VAEConfig
        seed: int,
        device: torch.device,
    ):
        self.cfg = cfg
        self.seed = seed
        self.device = device
        self.model: Optional[TabularVAE] = None

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def train(
        self,
        X: np.ndarray,
        X_val: Optional[np.ndarray] = None,
    ) -> TabularVAE:
        """
        Train the VAE.

        Performance notes
        -----------------
        * For small datasets (N <= 50k), uses **full-batch** training which
          eliminates the inner Python loop entirely (1 forward + 1 backward
          per epoch instead of ceil(N/batch_size)).
        * Loss is computed inline (no method-call overhead in the hot path).
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

        # Create model
        self.model = TabularVAE(
            input_dim=input_dim,
            latent_dim=int(self.cfg.latent_dim),
            hidden_dim=int(self.cfg.hidden_dim),
        ).to(self.device)

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

        # Steps per epoch (needed for scheduler)
        use_full_batch = n_train <= max(batch_size, 50_000)
        steps_per_epoch = 1 if use_full_batch else max(1, (n_train + batch_size - 1) // batch_size)

        # OneCycleLR: ramps LR up then cosine-anneals down — much faster convergence
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(self.cfg.lr) * 10,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )

        log_every = int(getattr(self.cfg, "log_every", 10))
        if log_every <= 0:
            log_every = 10

        val_every = int(getattr(self.cfg, "val_every", log_every))
        if val_every <= 0:
            val_every = log_every

        early_stop_patience = int(getattr(self.cfg, "early_stopping_patience", 0))

        # CUDA speed knobs
        if self.device.type == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            try:
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            except Exception:
                pass

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

        n_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"[VAE] Training: {epochs} epochs, N={n_train}, D={input_dim}, "
            f"params={n_params:,}, device={self.device}, "
            f"{'full-batch' if use_full_batch else 'batch_size=' + str(batch_size)}"
            f"{', compiled' if use_compiled else ''}",
            flush=True,
        )

        # Cache attribute lookup
        kl_weight = float(self.cfg.kl_weight)

        best_val_loss = float("inf")
        patience_counter = 0
        last_val_loss: Optional[float] = None
        t0 = time.time()

        if use_full_batch:
            # ===========================================================
            # FAST PATH: full-batch (no inner loop, no shuffling needed)
            # ===========================================================
            for epoch in range(epochs):
                compiled_model.train()
                recon, mu, logvar = compiled_model(X_train_t)
                recon_loss = F.mse_loss(recon, X_train_t, reduction="mean")
                kl_loss = -0.5 * torch.mean(
                    1 + logvar - mu.pow(2) - logvar.exp()
                )
                loss = recon_loss + kl_weight * kl_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

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
                        X_val_t, batch_size=n_train
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

                    recon, mu, logvar = compiled_model(batch)
                    recon_loss = F.mse_loss(recon, batch, reduction="mean")
                    kl_loss = -0.5 * torch.mean(
                        1 + logvar - mu.pow(2) - logvar.exp()
                    )
                    loss = recon_loss + kl_weight * kl_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

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
                        X_val_t, batch_size=batch_size
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

    def _vae_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VAE loss (reconstruction + KL divergence)."""
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.cfg.kl_weight * kl_loss

    def _evaluate_tensor(self, X_tensor: torch.Tensor, *, batch_size: int) -> float:
        """Evaluate the model on an in-memory tensor already on the correct device."""
        self.model.eval()
        n = int(X_tensor.shape[0])
        if n == 0:
            return float("nan")
        if batch_size <= 0:
            batch_size = n

        total_loss = torch.zeros((), device=X_tensor.device)

        with torch.inference_mode():
            for start in range(0, n, batch_size):
                batch = X_tensor[start : start + batch_size]
                recon, mu, logvar = self.model(batch)
                loss = self._vae_loss(batch, recon, mu, logvar)
                total_loss = total_loss + loss.detach() * batch.size(0)

        return float((total_loss / n).item())

    def _evaluate(self, loader: DataLoader) -> float:
        """Evaluate model on a data loader."""
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

    def embed(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get latent embeddings for data.

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
