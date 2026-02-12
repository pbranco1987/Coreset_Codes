# fig_baseline_comparison.R — Fig 6: Baseline comparison grouped bars (Sec. VIII.E)
#
# CSV columns: method, variant, metric, value, is_r1_knee
#   - method:     "Uniform", "k-Means", "Herding", "Farthest-First", ...
#   - variant:    "unconstrained", "quota_matched"
#   - metric:     "e_Nys", "RMSE_4G", "KL_geo"
#   - value:      numeric
#   - is_r1_knee: TRUE/FALSE (TRUE for R1-knee reference)
#
# Output: multi-panel grouped bar chart, IEEE double-column width.

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

# Preserve method order from data
method_order <- unique(df$method)
df$method <- factor(df$method, levels = method_order)

# Determine if we have variant data
has_variant <- "variant" %in% names(df) && length(unique(df$variant)) > 1

# Compute R1 knee reference values per metric
r1_df <- df[!is.na(df$is_r1_knee) & df$is_r1_knee == TRUE, ]
r1_ref <- aggregate(value ~ metric, data = r1_df, FUN = min)
names(r1_ref)[2] <- "r1_value"

# Metrics as facet panels
metric_order <- unique(df$metric)
df$metric <- factor(df$metric, levels = metric_order)

if (has_variant) {
  # Side-by-side bars for unconstrained / quota-matched
  variant_colours <- c("unconstrained" = COLORS[1], "quota_matched" = COLORS[2])

  p <- ggplot(df, aes(x = method, y = value, fill = variant)) +
    geom_col(
      position = position_dodge(width = 0.75), width = 0.65,
      colour = "white", linewidth = 0.3, alpha = 0.85
    ) +
    scale_fill_manual(
      values = variant_colours,
      labels = c("unconstrained" = "Unconstrained", "quota_matched" = "Quota-matched"),
      name = NULL
    )
} else {
  # Single bars — highlight R1-knee differently
  df$bar_fill <- ifelse(
    !is.na(df$is_r1_knee) & df$is_r1_knee == TRUE,
    "R1-knee", "Baseline"
  )
  bar_colours <- c("Baseline" = COLORS[1], "R1-knee" = COLORS[4])

  p <- ggplot(df, aes(x = method, y = value, fill = bar_fill)) +
    geom_col(width = 0.6, colour = "white", linewidth = 0.3, alpha = 0.85) +
    scale_fill_manual(values = bar_colours, name = NULL)
}

p <- p +
  # R1 knee reference line per facet
  geom_hline(
    data = r1_ref,
    aes(yintercept = r1_value),
    linetype = "dashed", colour = COLORS[4], linewidth = 0.6, alpha = 0.7
  ) +

  facet_wrap(~ metric, scales = "free_y", nrow = 1) +

  labs(
    x     = NULL,
    y     = "Value",
    title = expression("Baseline comparison at " * k * "=300 (R10)")
  ) +
  theme_ieee() +
  theme(
    axis.text.x   = element_text(angle = 35, hjust = 1, size = 7.5),
    strip.text     = element_text(size = 9, face = "bold"),
    legend.position = "bottom",
    panel.spacing  = unit(0.6, "cm")
  )

fig_w <- max(7, length(metric_order) * 4)
ggsave(out, plot = p, width = fig_w, height = 4.5, dpi = DPI, device = "pdf")
cat("Saved:", out, "\n")
