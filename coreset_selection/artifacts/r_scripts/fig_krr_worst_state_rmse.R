# fig_krr_worst_state_rmse.R — Fig 4: Worst-state KRR RMSE vs k (Sec. VIII.D)
#
# CSV columns:
#   k, target, mean_rmse, worst_rmse, best_rmse
#   - k:          coreset size (integer)
#   - target:     "4G" or "5G"
#   - mean_rmse:  average RMSE across states
#   - worst_rmse: worst-state RMSE (may be NA)
#   - best_rmse:  best-state RMSE (may be NA)
#
# Output: multi-panel line plot, IEEE double-column width.

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

# Ensure factor ordering for targets
target_order <- unique(df$target)
panel_labels <- setNames(
  paste0("(", letters[seq_along(target_order)], ") RMSE y^(", target_order, ")"),
  target_order
)
df$target <- factor(df$target, levels = target_order)

# Determine if we have worst/best data
has_worst <- "worst_rmse" %in% names(df) && any(!is.na(df$worst_rmse))
has_best  <- "best_rmse" %in% names(df) && any(!is.na(df$best_rmse))

# Build plot
p <- ggplot(df, aes(x = k)) +

  # Shaded band: best to worst state
  {if (has_worst && has_best)
    geom_ribbon(
      aes(ymin = best_rmse, ymax = worst_rmse),
      fill = COLORS[4], alpha = 0.1
    )
  } +

  # Average RMSE (main line)
  geom_line(aes(y = mean_rmse), colour = COLORS[1], linewidth = 1.2) +
  geom_point(aes(y = mean_rmse), colour = COLORS[1], size = 3, shape = 16) +

  # Worst-state RMSE
  {if (has_worst)
    geom_line(aes(y = worst_rmse), colour = COLORS[4], linewidth = 1.0,
              linetype = "dashed")
  } +
  {if (has_worst)
    geom_point(aes(y = worst_rmse), colour = COLORS[4], size = 2.5, shape = 15)
  } +

  # Best-state RMSE
  {if (has_best)
    geom_line(aes(y = best_rmse), colour = COLORS[3], linewidth = 0.8,
              linetype = "dotted")
  } +
  {if (has_best)
    geom_point(aes(y = best_rmse), colour = COLORS[3], size = 2, shape = 17)
  } +

  facet_wrap(~ target, scales = "free_y", nrow = 1,
             labeller = labeller(target = panel_labels)) +

  labs(
    x     = expression("Coreset size " * k),
    y     = "RMSE",
    title = "Worst-state vs. average RMSE across coreset budget (R1, equity analysis)"
  ) +
  theme_ieee() +
  theme(
    strip.text    = element_text(size = 10, face = "bold"),
    panel.spacing = unit(0.6, "cm")
  )

# Dynamic width
n_panels <- length(target_order)
fig_w <- max(5.5, 5 * n_panels)
ggsave(out, plot = p, width = fig_w, height = 4.2, dpi = DPI, device = "pdf")
cat("Saved:", out, "\n")
