# _theme_ieee.R — Shared ggplot2 theme for IEEE-compliant manuscript figures.
#
# Sourced by all 8 figure R scripts.  Provides:
#   theme_ieee()      — ggplot2 theme matching _ma_helpers._set_style()
#   COLORS            — standard colour palette (matches matplotlib tab10)
#   IEEE_SINGLE_W     — single-column width (inches)
#   IEEE_DOUBLE_W     — double-column width (inches)
#   DPI               — save resolution
#   parse_extra_args() — utility for --key=value CLI args
#
# Mirrors the Python _set_style() settings:
#   font.size: 10, axes.titlesize: 11, axes.labelsize: 10,
#   xtick.labelsize: 9, ytick.labelsize: 9, legend.fontsize: 8,
#   grid.alpha: 0.25, grid.linewidth: 0.5,
#   axes.spines.top: FALSE, axes.spines.right: FALSE
#   savefig.dpi: 300

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

# ---------------------------------------------------------------------------
# Dimensions & quality
# ---------------------------------------------------------------------------
IEEE_SINGLE_W <- 3.5   # inches
IEEE_DOUBLE_W <- 7.0   # inches
DPI           <- 300

# ---------------------------------------------------------------------------
# Colour palette (matplotlib tab10 first 8)
# ---------------------------------------------------------------------------
COLORS <- c(
  "#1f77b4",  # blue
  "#ff7f0e",  # orange
  "#2ca02c",  # green
  "#d62728",  # red
  "#9467bd",  # purple
  "#8c564b",  # brown
  "#e377c2",  # pink
  "#7f7f7f"   # gray
)

# ---------------------------------------------------------------------------
# IEEE theme
# ---------------------------------------------------------------------------
theme_ieee <- function(base_size = 10) {
  theme_minimal(base_size = base_size) %+replace%
    theme(
      # Text sizes
      plot.title       = element_text(size = base_size + 1, face = "bold",
                                       hjust = 0.5, margin = margin(b = 6)),
      axis.title       = element_text(size = base_size),
      axis.text        = element_text(size = base_size - 1),
      legend.text      = element_text(size = base_size - 2),
      legend.title     = element_text(size = base_size - 1),
      strip.text       = element_text(size = base_size, face = "bold"),

      # Remove top and right "spines" (axis lines)
      axis.line.x      = element_line(colour = "black", linewidth = 0.4),
      axis.line.y      = element_line(colour = "black", linewidth = 0.4),
      panel.border     = element_blank(),

      # Grid: light grey, thin
      panel.grid.major = element_line(colour = "grey85", linewidth = 0.5),
      panel.grid.minor = element_blank(),

      # Legend background with slight transparency
      legend.background = element_rect(fill = alpha("white", 0.9),
                                        colour = "grey80", linewidth = 0.3),
      legend.key        = element_rect(fill = "transparent"),

      # Margin
      plot.margin = margin(t = 4, r = 6, b = 4, l = 4, unit = "pt")
    )
}

# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------
parse_extra_args <- function(args) {
  # Parse --key=value arguments from commandArgs into a named list
  result <- list()
  for (arg in args) {
    if (startsWith(arg, "--")) {
      parts <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
      if (length(parts) == 2) {
        key <- parts[1]
        val <- parts[2]
        # Try numeric conversion
        num_val <- suppressWarnings(as.numeric(val))
        if (!is.na(num_val)) {
          result[[key]] <- num_val
        } else {
          result[[key]] <- val
        }
      }
    }
  }
  result
}

# ---------------------------------------------------------------------------
# Convenience: read CSV + output path from first two CLI args
# ---------------------------------------------------------------------------
read_cli_data <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 2) {
    stop("Usage: Rscript <script.R> <input.csv> <output.pdf> [--key=value ...]")
  }
  csv_path <- args[1]
  out_path <- args[2]
  extra    <- if (length(args) > 2) parse_extra_args(args[3:length(args)]) else list()

  df <- read.csv(csv_path, stringsAsFactors = FALSE)
  list(df = df, out_path = out_path, extra = extra)
}
