# fig_distortion_cardinality_R1.R — Fig 3: Distortion vs cardinality 2x2 (Sec. VIII.C)
#
# CSV columns (long/tidy format):
#   k, metric, value, stat
#   - k:      coreset size (integer)
#   - metric: one of "e_Nys", "e_kPCA", "RMSE_4G", "RMSE_5G"
#   - value:  numeric
#   - stat:   "envelope_best", "mean", "mean_plus_std", "mean_minus_std"
#
# Output: 2x2 faceted panel, IEEE double-column width (~7 in).

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

# Pivot data: separate envelope, mean, std bounds
env_df  <- df[df$stat == "envelope_best", ]
mean_df <- df[df$stat == "mean", ]
hi_df   <- df[df$stat == "mean_plus_std", ]
lo_df   <- df[df$stat == "mean_minus_std", ]

# Panel labels
metric_order <- unique(df$metric)
panel_labels <- setNames(
  paste0("(", letters[seq_along(metric_order)], ") ", metric_order),
  metric_order
)
env_df$metric  <- factor(env_df$metric, levels = metric_order)
mean_df$metric <- factor(mean_df$metric, levels = metric_order)

# Build ribbon data if std bounds exist
has_ribbon <- nrow(hi_df) > 0 && nrow(lo_df) > 0
if (has_ribbon) {
  ribbon_df <- merge(
    hi_df[, c("k", "metric", "value")],
    lo_df[, c("k", "metric", "value")],
    by = c("k", "metric"), suffixes = c("_hi", "_lo")
  )
  ribbon_df$metric <- factor(ribbon_df$metric, levels = metric_order)
}

p <- ggplot() +
  # Error band (mean ± std) if available
  {if (has_ribbon)
    geom_ribbon(
      data = ribbon_df,
      aes(x = k, ymin = value_lo, ymax = value_hi),
      fill = COLORS[1], alpha = 0.15
    )
  } +

  # Envelope (best) line
  geom_line(
    data = env_df, aes(x = k, y = value),
    colour = COLORS[1], linewidth = 1.0
  ) +
  geom_point(
    data = env_df, aes(x = k, y = value),
    colour = COLORS[1], size = 2.5, shape = 16
  ) +

  # 2x2 facet
  facet_wrap(~ metric, scales = "free_y", ncol = 2,
             labeller = labeller(metric = panel_labels)) +

  labs(
    x     = expression(k),
    y     = "Metric value",
    title = "R1: Raw-space metrics vs. coreset size k (metric-wise envelope)"
  ) +
  theme_ieee() +
  theme(
    strip.text    = element_text(size = 10, face = "bold"),
    panel.spacing = unit(0.6, "cm")
  )

ggsave(out, plot = p, width = 7.0, height = 5.25, dpi = DPI, device = "pdf")
cat("Saved:", out, "\n")
