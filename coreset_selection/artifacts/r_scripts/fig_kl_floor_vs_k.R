# fig_kl_floor_vs_k.R — Fig 1: KL feasibility floor vs coreset size k
#
# CSV columns: k, kl_min
# Extra args:  --tau1, --tau2, --tau3 (threshold values, default 0.01, 0.02, 0.05)
#              --grid_k (comma-separated grid, default "50,100,200,300,400,500")
#
# Output: single-panel line plot, IEEE single-column width (~5.5 in).

# Source shared theme
source(file.path(dirname(sys.frame(1)$ofile %||% "."),"_theme_ieee.R"))

# Fallback for `%||%` if running via Rscript --vanilla
if (!exists("%||%", mode = "function")) {
  `%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a
}

# Source theme (robust path resolution)
script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) {
    # When run via Rscript, use commandArgs to find script location
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("--file=", args, value = TRUE)
    if (length(file_arg) > 0) {
      dirname(sub("--file=", "", file_arg[1]))
    } else {
      "."
    }
  }
)
source(file.path(script_dir, "_theme_ieee.R"))

# Read data
cli <- read_cli_data()
df  <- cli$df
out <- cli$out_path
extra <- cli$extra

# Parse thresholds
tau1 <- extra[["tau1"]] %||% 0.01
tau2 <- extra[["tau2"]] %||% 0.02
tau3 <- extra[["tau3"]] %||% 0.05
grid_k_str <- extra[["grid_k"]] %||% "50,100,200,300,400,500"
grid_k <- as.numeric(strsplit(grid_k_str, ",")[[1]])

# Sort data
df <- df[order(df$k), ]

# Build tau reference data
tau_df <- data.frame(
  tau   = c(tau1, tau2, tau3),
  label = c(
    paste0("tau == ", tau1),
    paste0("tau == ", tau2),
    paste0("tau == ", tau3)
  ),
  color = c(COLORS[4], COLORS[2], COLORS[3]),  # red, orange, green
  stringsAsFactors = FALSE
)

# Find minimum feasible k for each tau
tau_df$k_feasible <- NA_real_
tau_df$kl_at_k    <- NA_real_
for (i in seq_len(nrow(tau_df))) {
  idx <- which(df$kl_min <= tau_df$tau[i])
  if (length(idx) > 0) {
    first <- idx[1]
    tau_df$k_feasible[i] <- df$k[first]
    tau_df$kl_at_k[i]    <- df$kl_min[first]
  }
}

# Grid point annotations
grid_df <- data.frame(k = grid_k)
grid_df$kl_min <- approx(df$k, df$kl_min, xout = grid_k, rule = 2)$y
# Only label selected grid points
grid_df$show_label <- grid_df$k %in% c(50, 100, 300, 500)

# ---- Build plot ----
p <- ggplot(df, aes(x = k, y = kl_min)) +
  # Main KL floor curve
  geom_line(linewidth = 1.2, colour = COLORS[1]) +

  # Tau threshold lines
  geom_hline(
    data = tau_df,
    aes(yintercept = tau),
    linetype = "dashed", linewidth = 0.6, alpha = 0.75,
    colour = tau_df$color
  ) +

  # Tau labels on right margin
  geom_text(
    data = tau_df,
    aes(x = max(df$k) * 0.98, y = tau, label = paste0("tau==", tau)),
    parse = TRUE, hjust = 1, vjust = -0.5, size = 3, colour = tau_df$color
  ) +

  # Feasibility markers (inverted triangle)
  geom_point(
    data = tau_df[!is.na(tau_df$k_feasible), ],
    aes(x = k_feasible, y = tau),
    shape = 25, size = 3, fill = tau_df$color[!is.na(tau_df$k_feasible)],
    colour = "black", stroke = 0.3
  ) +

  # Feasibility annotations (k >= ...)
  geom_text(
    data = tau_df[!is.na(tau_df$k_feasible), ],
    aes(x = k_feasible + 25, y = tau + 0.008,
        label = paste0("k >= ", k_feasible)),
    size = 2.8, colour = tau_df$color[!is.na(tau_df$k_feasible)],
    hjust = 0
  ) +

  # Grid point markers
  geom_point(
    data = grid_df,
    aes(x = k, y = kl_min),
    shape = 21, size = 3, fill = COLORS[4],
    colour = "black", stroke = 0.4
  ) +

  # Grid point labels (selected)
  geom_text(
    data = grid_df[grid_df$show_label, ],
    aes(x = k, y = kl_min + 0.004, label = sprintf("%.4f", kl_min)),
    size = 2.5, colour = "grey40", vjust = 0
  ) +

  # Labels and theme
  labs(
    x     = "Coreset size k",
    y     = expression(KL[min](k)),
    title = expression("Feasibility floor " * KL[min](k) * " vs. coreset budget")
  ) +
  scale_x_continuous(limits = c(25, NA), expand = expansion(mult = c(0, 0.02))) +
  theme_ieee()

# Save
ggsave(out, plot = p, width = 5.5, height = 3.8, dpi = DPI, device = "pdf")
cat("Saved:", out, "\n")
