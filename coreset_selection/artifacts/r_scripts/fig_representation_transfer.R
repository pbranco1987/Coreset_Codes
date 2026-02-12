# fig_representation_transfer.R — Fig 7: Representation transfer bars (Sec. VIII.G)
#
# CSV columns: representation, metric, best, mean, std
#   - representation: "Raw (p=D)", "PCA (p=20)", "VAE (p=32)"
#   - metric:         "e_Nys", "e_kPCA", "RMSE_4G"
#   - best, mean, std: numeric values
#
# Output: multi-panel bar chart, IEEE double-column width (~7 in).

script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("--file=", args, value = TRUE)
    if (length(file_arg) > 0) dirname(sub("--file=", "", file_arg[1])) else "."
  }
)
source(file.path(script_dir, "_theme_ieee.R"))

cli <- read_cli_data()
df  <- cli$df
out <- cli$out_path

# Ensure factor ordering
repr_order <- unique(df$representation)
df$representation <- factor(df$representation, levels = repr_order)

# Assign colours to representations
n_repr <- length(repr_order)
repr_colours <- setNames(COLORS[seq_len(n_repr)], repr_order)

# Panel labels
metric_order <- unique(df$metric)
panel_labels <- setNames(
  paste0("(", letters[seq_along(metric_order)], ")"),
  metric_order
)

# ggplot2 with faceting
p <- ggplot(df, aes(x = representation, y = best, fill = representation)) +
  geom_col(alpha = 0.75, colour = "black", linewidth = 0.3, width = 0.6) +

  # Error bars (mean ± std)
  geom_errorbar(
    aes(ymin = mean - std, ymax = mean + std),
    width = 0.2, linewidth = 0.6, colour = "black"
  ) +

  # Value labels on top of bars
  geom_text(
    aes(label = sprintf("%.4f", best), y = best),
    vjust = -0.8, size = 2.8, colour = "grey30"
  ) +

  # Facet by metric
  facet_wrap(~ metric, scales = "free_y", nrow = 1,
             labeller = labeller(metric = function(x) {
               paste0(panel_labels[x], " ", x)
             })) +

  scale_fill_manual(values = repr_colours, guide = "none") +

  labs(
    x     = NULL,
    y     = "Metric value (lower is better)",
    title = expression("Representation transfer: raw-space evaluation at " * k * "=300")
  ) +
  theme_ieee() +
  theme(
    axis.text.x  = element_text(angle = 15, hjust = 1, size = 8),
    strip.text   = element_text(size = 10, face = "bold"),
    panel.spacing = unit(0.8, "cm")
  )

# Dynamic width based on number of metrics
fig_w <- max(7, 3.5 * length(metric_order))
ggsave(out, plot = p, width = fig_w, height = 4.2, dpi = DPI, device = "pdf")
cat("Saved:", out, "\n")
