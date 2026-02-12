# fig_geo_ablation_scatter.R — Fig 2: Geo ablation scatter (Sec. VIII.B)
#
# CSV columns: nystrom_error, geo_l1, constraint_regime, is_r1_knee
# Optional:    feasible_boundary (single value as extra arg)
# Output: scatter plot, IEEE single-column width (~3.5 in).

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
extra <- cli$extra

# Separate R1 knee points from main scatter data
knee_df  <- df[!is.na(df$is_r1_knee) & df$is_r1_knee == TRUE, ]
main_df  <- df[is.na(df$is_r1_knee) | df$is_r1_knee == FALSE, ]

# Constraint regime shapes & colours
regime_map <- data.frame(
  regime = c("exactk_only", "exactk", "quota", "quota+exactk", "unconstrained", "none"),
  shape_val  = c(16, 16, 15, 17, 18, 18),
  colour_val = c(COLORS[1], COLORS[1], COLORS[2], COLORS[3], COLORS[5], COLORS[5]),
  label  = c("Exact-k only", "Exact-k only", "Quota-matched",
             "Quota + exact-k", "Unconstrained", "Unconstrained"),
  stringsAsFactors = FALSE
)

# Map regimes to main_df
main_df$regime_label <- regime_map$label[match(main_df$constraint_regime, regime_map$regime)]
main_df$regime_label[is.na(main_df$regime_label)] <- as.character(
  main_df$constraint_regime[is.na(main_df$regime_label)]
)

# De-duplicate legend labels
unique_labels <- unique(main_df$regime_label)
label_colours <- setNames(
  sapply(unique_labels, function(lbl) {
    idx <- which(regime_map$label == lbl)[1]
    if (!is.na(idx)) regime_map$colour_val[idx] else COLORS[8]
  }),
  unique_labels
)
label_shapes <- setNames(
  sapply(unique_labels, function(lbl) {
    idx <- which(regime_map$label == lbl)[1]
    if (!is.na(idx)) regime_map$shape_val[idx] else 16
  }),
  unique_labels
)

p <- ggplot() +
  # Main scatter
  geom_point(
    data = main_df,
    aes(x = nystrom_error, y = geo_l1, colour = regime_label, shape = regime_label),
    size = 2, alpha = 0.65, stroke = 0.3
  ) +
  scale_colour_manual(values = label_colours, name = NULL) +
  scale_shape_manual(values = label_shapes, name = NULL) +

  # R1 knee point overlay
  {if (nrow(knee_df) > 0)
    geom_point(
      data = knee_df,
      aes(x = nystrom_error, y = geo_l1),
      shape = 8, size = 4, colour = COLORS[4], stroke = 0.8
    )
  } +
  {if (nrow(knee_df) > 0)
    annotate("text",
      x = knee_df$nystrom_error[1], y = knee_df$geo_l1[1],
      label = "R1 knee\n(constrained)", hjust = -0.15, vjust = 0.5,
      size = 2.8, colour = COLORS[4]
    )
  } +

  # Feasible boundary (if provided)
  {if (!is.null(extra[["feasible_boundary"]]))
    geom_hline(yintercept = extra[["feasible_boundary"]],
               linetype = "dashed", colour = "grey50", linewidth = 0.6)
  } +

  # Log scale on x-axis
  scale_x_log10(labels = scales::label_number(accuracy = 0.001)) +

  labs(
    x     = expression(paste("Nyström error ", e[Nys])),
    y     = expression(paste("Geographic ", ell[1], " drift")),
    title = expression("Composition drift vs. Nyström error (R6, " * k * "=300)")
  ) +
  theme_ieee() +
  theme(
    legend.position = c(0.02, 0.98),
    legend.justification = c(0, 1)
  )

ggsave(out, plot = p, width = 3.5, height = 2.8, dpi = DPI, device = "pdf")
cat("Saved:", out, "\n")
