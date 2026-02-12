# fig_regional_validity_k300.R — Fig 5: Regional KPI validity at k=300 (Sec. VIII.D)
#
# CSV columns: run, target, metric_name, value
#   - run:         "R1", "R5"
#   - target:      "4G", "5G"
#   - metric_name: "Max drift", "Avg drift", "Kendall's tau"
#   - value:       numeric
#
# Output: 3-panel grouped bar chart, IEEE double-column width (~7 in).

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

# Create interaction for bar positioning
df$group <- paste0(df$run, " (", df$target, ")")

# Colour mapping: R1-4G, R1-5G, R5-4G, R5-5G
group_colours <- c(
  "R1 (4G)" = COLORS[1],
  "R1 (5G)" = "#aec7e8",
  "R5 (4G)" = COLORS[2],
  "R5 (5G)" = "#ffbb78"
)

# Panel labels
metric_order <- unique(df$metric_name)
panel_labels <- setNames(
  paste0("(", letters[seq_along(metric_order)], ")"),
  metric_order
)

df$metric_name <- factor(df$metric_name, levels = metric_order)
df$group       <- factor(df$group, levels = names(group_colours))

p <- ggplot(df, aes(x = target, y = value, fill = group)) +
  geom_col(
    position = position_dodge(width = 0.7), width = 0.6,
    colour = "black", linewidth = 0.3, alpha = 0.85
  ) +

  facet_wrap(~ metric_name, scales = "free_y", nrow = 1,
             labeller = labeller(metric_name = function(x) {
               paste0(panel_labels[x], " ", x)
             })) +

  scale_fill_manual(values = group_colours, name = NULL) +

  labs(
    x     = "Technology target",
    y     = NULL,
    title = expression("State-conditioned KPI stability at " * k * "=300: R1 vs R5")
  ) +
  theme_ieee() +
  theme(
    legend.position  = "bottom",
    legend.direction = "horizontal",
    strip.text       = element_text(size = 10, face = "bold"),
    panel.spacing    = unit(0.6, "cm")
  ) +
  guides(fill = guide_legend(nrow = 1))

ggsave(out, plot = p, width = 7.0, height = 3.8, dpi = DPI, device = "pdf")
cat("Saved:", out, "\n")
