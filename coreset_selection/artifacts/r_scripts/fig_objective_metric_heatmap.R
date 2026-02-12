# fig_objective_metric_heatmap.R — Fig 8: Objective-metric Spearman rho heatmap
#
# CSV columns: objective, metric, spearman_rho
# Output: annotated heatmap, IEEE single-column width.

# Robust script directory detection
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

# Ensure correct factor order (objectives as rows, metrics as columns)
if (!"objective" %in% names(df) || !"metric" %in% names(df)) {
  stop("CSV must contain 'objective', 'metric', 'spearman_rho' columns")
}

# Determine figure size from data dimensions
n_obj <- length(unique(df$objective))
n_met <- length(unique(df$metric))
fig_w <- max(3.5, n_met * 0.9 + 1.2)
fig_h <- max(2.0, n_obj * 0.65 + 1.0)

# Text colour: white on dark cells, black on light
df$text_colour <- ifelse(abs(df$spearman_rho) > 0.55, "white", "black")

# Preserve order from Python (first occurrence in data)
df$objective <- factor(df$objective, levels = unique(df$objective))
df$metric    <- factor(df$metric,    levels = unique(df$metric))

# Determine symmetric colour limits
abs_lim <- max(abs(df$spearman_rho), na.rm = TRUE)
abs_lim <- max(abs_lim, 0.05)  # minimum range

p <- ggplot(df, aes(x = metric, y = objective, fill = spearman_rho)) +
  geom_tile(colour = "white", linewidth = 0.5) +

  # Cell annotations
  geom_text(
    aes(label = sprintf("%.2f", spearman_rho), colour = text_colour),
    size = 3.2, fontface = "plain", show.legend = FALSE
  ) +
  scale_colour_identity() +

  # Diverging colour scale centred at 0
  scale_fill_gradient2(
    low = "#2166AC", mid = "white", high = "#B2182B",
    midpoint = 0, limits = c(-abs_lim, abs_lim),
    name = expression(Spearman ~ rho)
  ) +

  labs(
    x     = NULL,
    y     = NULL,
    title = expression(Spearman ~ rho * ": optimisation objectives vs. raw-space metrics")
  ) +
  theme_ieee() +
  theme(
    axis.text.x = element_text(angle = 40, hjust = 1, size = 9),
    axis.text.y = element_text(size = 10),
    panel.grid   = element_blank(),
    legend.position = "right"
  )

ggsave(out, plot = p, width = fig_w, height = fig_h, dpi = DPI, device = "pdf")
cat("Saved:", out, "\n")
