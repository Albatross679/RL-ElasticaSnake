#!/usr/bin/env python3
"""Produce an interactive Vega-Lite report for training_data.json."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Training" / "Logs" / "training_data.json"
# DATA_PATH = ROOT / "Training" / "Logs" / "test_training_data.json"
OUTPUT_PATH = ROOT / "Training" / "Results" / "training_data_overview.html"
# OUTPUT_PATH = ROOT / "Training" / "Results" / "fixed_action_testing_data_overview.html"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
CONFIG_PATHS = [
    ROOT / "Training" / "Logs" / "PPO_Snake_Model_config.json",
    ROOT / "Training" / "Logs" / "PPO_Snake_Checkpoint_config.json",
]

REWARD_TERMS = [
    "forward_progress",
    "speed_perpendicular_to_heading_penalty",
    "curvature_range_penalty",
    "curvature_oscillation_reward",
    "energy_penalty",
    "smoothness_penalty",
    "alignment_bonus",
    "projected_speed",
    "speed_perpendicular_to_target_penalty",
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Training Data Overview</title>
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <style>
    :root {{
      color-scheme: light;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      background: #f8fbff;
      color: #102a43;
    }}
    h1, h2, h3 {{
      color: #1d2d44;
      margin-bottom: 12px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 24px;
    }}
    .chart-full {{
      margin-top: 32px;
    }}
    .metadata {{
      display: flex;
      gap: 32px;
      flex-wrap: wrap;
      margin-bottom: 24px;
    }}
    .metadata div {{
      background: #fff;
      padding: 12px 18px;
      border-radius: 10px;
      border: 1px solid #e0e7ff;
      box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
    }}
    a {{
      color: #1d4ed8;
    }}
    .term-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 24px;
      margin-top: 16px;
    }}
    .term-panel {{
      background: #fff;
      border: 1px solid #e0e7ff;
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
    }}
    .config-section {{
      background: #fff;
      border: 1px solid #e0e7ff;
      border-radius: 12px;
      padding: 24px;
      margin-top: 24px;
      box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
    }}
    .config-category {{
      margin-bottom: 32px;
    }}
    .config-category h3 {{
      color: #1d4ed8;
      border-bottom: 2px solid #e0e7ff;
      padding-bottom: 8px;
      margin-bottom: 16px;
    }}
    .config-item {{
      display: flex;
      justify-content: space-between;
      padding: 8px 0;
      border-bottom: 1px solid #f1f5f9;
    }}
    .config-item:last-child {{
      border-bottom: none;
    }}
    .config-label {{
      font-weight: 500;
      color: #334155;
    }}
    .config-value {{
      color: #64748b;
      text-align: right;
      max-width: 60%;
      word-break: break-word;
    }}
  </style>
</head>
<body>
  <h1>Training Data Overview</h1>
  <section class="metadata">
    <div><strong>Saved at</strong><br>{saved_at}</div>
    <div><strong>Total timesteps</strong><br>{total_timesteps:,}</div>
    <div><strong>Episode count</strong><br>{episode_count}</div>
  </section>

  <h2>Training Configuration</h2>
  <div class="config-section">
    {configuration_html}
  </div>

  <h2>Episode metrics</h2>
  <div class="chart-grid">
    <div id="episodes-reward"></div>
    <div id="episodes-length"></div>
    <div id="episodes-timesteps"></div>
  </div>
  <div class="chart-full" id="reward-frequency"></div>

  <h2>Learning Progress & Trend Analysis</h2>
  <div class="chart-grid">
    <div id="episodes-reward-trend"></div>
    <div id="reward-variance"></div>
  </div>
  <div class="chart-full" id="early-late-comparison"></div>

  <h2>Efficiency Metrics</h2>
  <div class="chart-grid">
    <div id="reward-per-timestep"></div>
    <div id="reward-per-simtime"></div>
  </div>

  <h2>Step-level signals</h2>
  <div class="chart-full" id="step-reward"></div>
  <div class="chart-full" id="step-speed"></div>
  <div class="chart-full" id="step-simtime"></div>

  <h2>Gradient norms</h2>
  <div class="chart-grid">
    <div id="gradient-norm-policy"></div>
    <div id="gradient-norm-value"></div>
  </div>
  <div class="chart-full" id="gradient-norms-combined"></div>

  <h2>Action signals</h2>
  <div class="term-grid" id="action-grid"></div>

  <h2>Curvature norms</h2>
  <div class="chart-grid">
    <div id="curvature-norm-avg"></div>
  </div>
  <h3>Sections</h3>
  <div class="term-grid" id="curvature-section-grid"></div>

  <h2>Reward term contributions</h2>
  <div class="chart-full" id="reward-terms"></div>
  <h3>Individual terms</h3>
  <div class="term-grid" id="reward-term-grid"></div>

  <h2>Curvature statistics</h2>
  <div class="chart-grid">
    <div id="curvature-mean"></div>
    <div id="curvature-std"></div>
  </div>

  <script>
    const episodesData = {episodes_data};
    const stepsData = {steps_data};
    const curvatureMeanData = {curvature_mean_data};
    const curvatureStdData = {curvature_std_data};
    const rewardTerms = {reward_terms};
    const curvatureSectionsData = {curvature_sections_data};
    const actionFields = {action_fields};
    const earlyLateComparisonData = {early_late_comparison_data};
    const rewardVarianceData = {reward_variance_data};
    const rewardTermLabels = {{
      forward_progress: "Forward progress",
      speed_perpendicular_to_heading_penalty: "Lateral penalty (perpendicular to heading)",
      curvature_range_penalty: "Curvature range penalty",
      curvature_oscillation_reward: "Curvature oscillation reward",
      energy_penalty: "Energy penalty",
      smoothness_penalty: "Smoothness penalty",
      alignment_bonus: "Alignment bonus",
      projected_speed: "Projected speed",
      speed_perpendicular_to_target_penalty: "Lateral penalty (perpendicular to target direction)"
    }};

    const baseConfig = {{
      background: "#ffffff",
      font: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      axis: {{
        gridColor: "#e3e7ef",
        gridOpacity: 0.7,
        tickColor: "#cbd5f5",
        labelFontSize: 12,
        titleFontSize: 13,
        labelColor: "#1f2933",
        titleColor: "#1f2933"
      }},
      header: {{
        labelFontSize: 14,
        titleFontSize: 16,
        labelFontWeight: 600,
        labelColor: "#1f2933",
        titleColor: "#1f2933"
      }},
      legend: {{
        labelFontSize: 12,
        titleFontSize: 13,
        direction: "horizontal",
        orient: "bottom"
      }},
      view: {{
        stroke: "#e3e7ef"
      }}
    }};

    const embedChart = (selector, spec) =>
      vegaEmbed(selector, {{ ...spec, config: baseConfig }}, {{ actions: false }});

    embedChart("#episodes-reward", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 360,
      height: 200,
      data: {{ values: episodesData }},
      layer: [
        {{
          mark: {{ type: "line", color: "#42a5f5", opacity: 0.5 }},
          encoding: {{
            x: {{ field: "episode", type: "quantitative" }},
            y: {{ field: "reward", type: "quantitative" }},
            tooltip: [
              {{ field: "episode", type: "quantitative" }},
              {{ field: "reward", type: "quantitative", format: ".2f" }}
            ]
          }}
        }},
        {{
          transform: [
            {{
              window: [{{ op: "mean", field: "reward", as: "reward_ma" }}],
              frame: [-5, 5],
              sort: [{{ field: "episode", order: "ascending" }}]
            }}
          ],
          mark: {{ type: "line", color: "#1d4ed8", strokeWidth: 2.5 }},
          encoding: {{
            x: {{ field: "episode", type: "quantitative" }},
            y: {{ field: "reward_ma", type: "quantitative", title: "Reward" }},
            tooltip: [
              {{ field: "episode", type: "quantitative" }},
              {{ field: "reward_ma", type: "quantitative", format: ".2f", title: "Moving Avg (10-ep)" }}
            ]
          }}
        }}
      ],
      title: "Episode reward (with 10-episode moving average)"
    }});

    embedChart("#episodes-length", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 360,
      height: 200,
      data: {{ values: episodesData }},
      mark: {{ type: "bar", color: "#ffb74d" }},
      encoding: {{
        x: {{ field: "episode", type: "ordinal" }},
        y: {{ field: "length", type: "quantitative" }},
        tooltip: [
          {{ field: "episode", type: "quantitative" }},
          {{ field: "length", type: "quantitative" }}
        ]
      }},
      title: "Episode lengths"
    }});

    embedChart("#episodes-timesteps", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 360,
      height: 200,
      data: {{ values: episodesData }},
      mark: {{ type: "line", color: "#66bb6a" }},
      encoding: {{
        x: {{ field: "episode", type: "quantitative" }},
        y: {{ field: "timesteps", type: "quantitative" }},
        tooltip: [
          {{ field: "episode", type: "quantitative" }},
          {{ field: "timesteps", type: "quantitative" }}
        ]
      }},
      title: "Cumulative training timesteps"
    }});

    embedChart("#episodes-reward-trend", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 480,
      height: 260,
      data: {{ values: episodesData }},
      layer: [
        {{
          mark: {{ type: "line", color: "#42a5f5", opacity: 0.4 }},
          encoding: {{
            x: {{ field: "episode", type: "quantitative" }},
            y: {{ field: "reward", type: "quantitative" }},
            tooltip: [
              {{ field: "episode", type: "quantitative" }},
              {{ field: "reward", type: "quantitative", format: ".2f" }}
            ]
          }}
        }},
        {{
          transform: [
            {{
              window: [{{ op: "mean", field: "reward", as: "reward_ma" }}],
              frame: [-5, 5],
              sort: [{{ field: "episode", order: "ascending" }}]
            }}
          ],
          mark: {{ type: "line", color: "#1d4ed8", strokeWidth: 3 }},
          encoding: {{
            x: {{ field: "episode", type: "quantitative" }},
            y: {{ field: "reward_ma", type: "quantitative", title: "Reward" }},
            tooltip: [
              {{ field: "episode", type: "quantitative" }},
              {{ field: "reward_ma", type: "quantitative", format: ".2f", title: "Moving Avg (10-ep)" }}
            ]
          }}
        }}
      ],
      title: "Episode reward trend (10-episode moving average)"
    }});

    embedChart("#reward-variance", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 480,
      height: 260,
      data: {{ values: rewardVarianceData }},
      layer: [
        {{
          mark: {{ type: "line", color: "#ef5350", strokeWidth: 2 }},
          encoding: {{
            x: {{ field: "episode", type: "quantitative", title: "Episode" }},
            y: {{ field: "variance", type: "quantitative", title: "Variance" }},
            tooltip: [
              {{ field: "episode", type: "quantitative" }},
              {{ field: "variance", type: "quantitative", format: ".2f" }},
              {{ field: "std_dev", type: "quantitative", format: ".2f", title: "Std Dev" }}
            ]
          }}
        }},
        {{
          mark: {{ type: "line", color: "#ab47bc", strokeDash: [4,4], strokeWidth: 2 }},
          encoding: {{
            x: {{ field: "episode", type: "quantitative" }},
            y: {{ field: "std_dev", type: "quantitative", title: "Std Dev" }},
            tooltip: [
              {{ field: "episode", type: "quantitative" }},
              {{ field: "std_dev", type: "quantitative", format: ".2f" }}
            ]
          }}
        }}
      ],
      title: "Reward variance over time (rolling window = 10 episodes)"
    }});

    embedChart("#early-late-comparison", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 960,
      height: 260,
      data: {{ values: earlyLateComparisonData }},
      mark: {{ type: "bar" }},
      encoding: {{
        x: {{ field: "metric", type: "nominal", title: "Metric" }},
        y: {{ field: "value", type: "quantitative", title: "Value" }},
        color: {{ field: "period", type: "nominal", title: "Period" }},
        tooltip: [
          {{ field: "period", type: "nominal" }},
          {{ field: "metric", type: "nominal" }},
          {{ field: "value", type: "quantitative", format: ".2f" }}
        ]
      }},
      title: "Early vs Late Episodes Comparison (First 25% vs Last 25%)"
    }});

    embedChart("#reward-per-timestep", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 480,
      height: 260,
      data: {{ values: episodesData }},
      layer: [
        {{
          mark: {{ type: "line", color: "#ffb74d", opacity: 0.5 }},
          encoding: {{
            x: {{ field: "episode", type: "quantitative" }},
            y: {{ field: "reward_per_timestep", type: "quantitative", title: "Reward per Timestep" }},
            tooltip: [
              {{ field: "episode", type: "quantitative" }},
              {{ field: "reward_per_timestep", type: "quantitative", format: ".4f" }}
            ]
          }}
        }},
        {{
          transform: [
            {{
              window: [{{ op: "mean", field: "reward_per_timestep", as: "rpt_ma" }}],
              frame: [-5, 5],
              sort: [{{ field: "episode", order: "ascending" }}]
            }}
          ],
          mark: {{ type: "line", color: "#f57c00", strokeWidth: 2.5 }},
          encoding: {{
            x: {{ field: "episode", type: "quantitative" }},
            y: {{ field: "rpt_ma", type: "quantitative" }},
            tooltip: [
              {{ field: "episode", type: "quantitative" }},
              {{ field: "rpt_ma", type: "quantitative", format: ".4f", title: "Moving Avg" }}
            ]
          }}
        }}
      ],
      title: "Reward per Timestep (10-episode moving average)"
    }});

    embedChart("#reward-per-simtime", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 480,
      height: 260,
      data: {{ values: episodesData }},
      layer: [
        {{
          mark: {{ type: "line", color: "#66bb6a", opacity: 0.5 }},
          encoding: {{
            x: {{ field: "episode", type: "quantitative" }},
            y: {{ field: "reward_per_simtime", type: "quantitative", title: "Reward per Simulation Time" }},
            tooltip: [
              {{ field: "episode", type: "quantitative" }},
              {{ field: "reward_per_simtime", type: "quantitative", format: ".4f" }}
            ]
          }}
        }},
        {{
          transform: [
            {{
              window: [{{ op: "mean", field: "reward_per_simtime", as: "rpst_ma" }}],
              frame: [-5, 5],
              sort: [{{ field: "episode", order: "ascending" }}]
            }}
          ],
          mark: {{ type: "line", color: "#2e7d32", strokeWidth: 2.5 }},
          encoding: {{
            x: {{ field: "episode", type: "quantitative" }},
            y: {{ field: "rpst_ma", type: "quantitative" }},
            tooltip: [
              {{ field: "episode", type: "quantitative" }},
              {{ field: "rpst_ma", type: "quantitative", format: ".4f", title: "Moving Avg" }}
            ]
          }}
        }}
      ],
      title: "Reward per Simulation Time (10-episode moving average)"
    }});

    embedChart("#reward-frequency", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 960,
      height: 260,
      data: {{ values: stepsData }},
      layer: [
        {{
          mark: {{ type: "bar", color: "#90caf9" }},
          encoding: {{
            x: {{
              field: "reward",
              type: "quantitative",
              bin: {{ maxbins: 40 }},
              title: "Reward"
            }},
            y: {{
              aggregate: "count",
              type: "quantitative",
              title: "Frequency"
            }},
            tooltip: [
              {{ field: "reward", type: "quantitative", bin: true, title: "Reward bin" }},
              {{ aggregate: "count", type: "quantitative", title: "Frequency" }}
            ]
          }}
        }},
        {{
          transform: [
            {{
              aggregate: [
                {{ op: "mean", field: "reward", as: "mean_reward" }}
              ]
            }}
          ],
          mark: {{ type: "rule", color: "#1d4ed8", strokeWidth: 2, strokeDash: [4,4] }},
          encoding: {{
            x: {{ field: "mean_reward", type: "quantitative" }},
            tooltip: [
              {{ field: "mean_reward", type: "quantitative", format: ".2f", title: "Mean reward" }}
            ]
          }}
        }},
        {{
          transform: [
            {{
              aggregate: [
                {{ op: "median", field: "reward", as: "median_reward" }}
              ]
            }}
          ],
          mark: {{ type: "rule", color: "#ef5350", strokeWidth: 2 }},
          encoding: {{
            x: {{ field: "median_reward", type: "quantitative" }},
            tooltip: [
              {{ field: "median_reward", type: "quantitative", format: ".2f", title: "Median reward" }}
            ]
          }}
        }}
      ],
      title: "Reward frequency (all steps) with mean/median"
    }});

    embedChart("#step-reward", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 960,
      height: 260,
      data: {{ values: stepsData }},
      layer: [
        {{
          mark: {{ type: "line", color: "#ef5350", opacity: 0.35 }},
          encoding: {{
            x: {{ field: "timestep", type: "quantitative" }},
            y: {{ field: "reward", type: "quantitative" }}
          }}
        }},
        {{
          transform: [
            {{
              window: [{{ op: "mean", field: "reward", as: "reward_mean" }}],
              frame: [-200, 0],
              sort: {{ field: "timestep" }}
            }}
          ],
          mark: {{ type: "line", color: "#ab47bc", strokeWidth: 2 }},
          encoding: {{
            x: {{ field: "timestep", type: "quantitative" }},
            y: {{ field: "reward_mean", type: "quantitative" }}
          }}
        }}
      ],
      title: "Step reward (rolling mean window = 200)"
    }});

    embedChart("#step-speed", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 960,
      height: 260,
      data: {{ values: stepsData }},
      transform: [
        {{
          fold: [
            "speed_along_heading",
            "velocity_projection",
            "speed_perpendicular_to_heading"
          ],
          as: ["metric", "value"]
        }}
      ],
      layer: [
        {{
          mark: {{ type: "line", opacity: 0.35 }},
          encoding: {{
            x: {{ field: "timestep", type: "quantitative" }},
            y: {{ field: "value", type: "quantitative" }},
            color: {{ field: "metric", type: "nominal" }}
          }}
        }},
        {{
          transform: [
            {{
              window: [
                {{ op: "mean", field: "value", as: "value_avg" }}
              ],
              frame: [-500, 0],
              groupby: ["metric"],
              sort: [{{ field: "timestep", order: "ascending" }}]
            }}
          ],
          mark: {{ type: "line", strokeWidth: 2 }},
          encoding: {{
            x: {{ field: "timestep", type: "quantitative" }},
            y: {{ field: "value_avg", type: "quantitative" }},
            color: {{ field: "metric", type: "nominal" }}
          }}
        }}
      ],
      title: "Speed signals (raw + rolling mean window = 500)"
    }});

    embedChart("#step-simtime", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 960,
      height: 220,
      data: {{ values: stepsData }},
      layer: [
        {{
          mark: {{ type: "line", color: "#26c6da" }},
          encoding: {{
            x: {{ field: "timestep", type: "quantitative" }},
            y: {{ field: "sim_time", type: "quantitative" }}
          }}
        }},
        {{
          mark: {{ type: "line", color: "#9e9e9e", strokeDash: [4,4] }},
          encoding: {{
            x: {{ field: "timestep", type: "quantitative" }},
            y: {{ field: "episode", type: "quantitative" }}
          }}
        }}
      ],
      title: "Simulation time and episode index"
    }});

    embedChart("#gradient-norm-policy", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 480,
      height: 260,
      data: {{ 
        values: stepsData.filter(d => d.gradient_norm_policy !== undefined && d.gradient_norm_policy !== null)
      }},
      layer: [
        {{
          mark: {{ type: "line", color: "#42a5f5", opacity: 0.4 }},
          encoding: {{
            x: {{ field: "timestep", type: "quantitative", title: "Timestep" }},
            y: {{ field: "gradient_norm_policy", type: "quantitative", title: "Gradient Norm" }},
            tooltip: [
              {{ field: "timestep", type: "quantitative" }},
              {{ field: "gradient_norm_policy", type: "quantitative", format: ".6f" }}
            ]
          }}
        }},
        {{
          transform: [
            {{
              window: [{{ op: "mean", field: "gradient_norm_policy", as: "gradient_norm_policy_smoothed" }}],
              frame: [-200, 0],
              sort: [{{ field: "timestep", order: "ascending" }}]
            }}
          ],
          mark: {{ type: "line", color: "#1d4ed8", strokeWidth: 2 }},
          encoding: {{
            x: {{ field: "timestep", type: "quantitative" }},
            y: {{ field: "gradient_norm_policy_smoothed", type: "quantitative" }},
            tooltip: [
              {{ field: "timestep", type: "quantitative" }},
              {{ field: "gradient_norm_policy_smoothed", type: "quantitative", format: ".6f", title: "Moving Avg" }}
            ]
          }}
        }}
      ],
      title: "Policy Gradient Norm (raw + rolling mean window = 200)"
    }});

    embedChart("#gradient-norm-value", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 480,
      height: 260,
      data: {{ 
        values: stepsData.filter(d => d.gradient_norm_value !== undefined && d.gradient_norm_value !== null)
      }},
      layer: [
        {{
          mark: {{ type: "line", color: "#ef5350", opacity: 0.4 }},
          encoding: {{
            x: {{ field: "timestep", type: "quantitative", title: "Timestep" }},
            y: {{ field: "gradient_norm_value", type: "quantitative", title: "Gradient Norm" }},
            tooltip: [
              {{ field: "timestep", type: "quantitative" }},
              {{ field: "gradient_norm_value", type: "quantitative", format: ".6f" }}
            ]
          }}
        }},
        {{
          transform: [
            {{
              window: [{{ op: "mean", field: "gradient_norm_value", as: "gradient_norm_value_smoothed" }}],
              frame: [-200, 0],
              sort: [{{ field: "timestep", order: "ascending" }}]
            }}
          ],
          mark: {{ type: "line", color: "#c62828", strokeWidth: 2 }},
          encoding: {{
            x: {{ field: "timestep", type: "quantitative" }},
            y: {{ field: "gradient_norm_value_smoothed", type: "quantitative" }},
            tooltip: [
              {{ field: "timestep", type: "quantitative" }},
              {{ field: "gradient_norm_value_smoothed", type: "quantitative", format: ".6f", title: "Moving Avg" }}
            ]
          }}
        }}
      ],
      title: "Value Gradient Norm (raw + rolling mean window = 200)"
    }});

    embedChart("#gradient-norms-combined", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 960,
      height: 260,
      data: {{ values: stepsData }},
      transform: [
        {{
          fold: [
            "gradient_norm_policy",
            "gradient_norm_value"
          ],
          as: ["gradient_type", "gradient_norm"]
        }},
        {{
          filter: "datum.gradient_norm !== null && datum.gradient_norm !== undefined"
        }}
      ],
      layer: [
        {{
          mark: {{ type: "line", opacity: 0.35 }},
          encoding: {{
            x: {{ field: "timestep", type: "quantitative", title: "Timestep" }},
            y: {{ field: "gradient_norm", type: "quantitative", title: "Gradient Norm" }},
            color: {{ 
              field: "gradient_type", 
              type: "nominal",
              scale: {{
                domain: ["gradient_norm_policy", "gradient_norm_value"],
                range: ["#42a5f5", "#ef5350"]
              }},
              legend: {{
                title: "Gradient Type",
                labelExpr: "datum.label === 'gradient_norm_policy' ? 'Policy' : 'Value'"
              }}
            }},
            tooltip: [
              {{ field: "timestep", type: "quantitative" }},
              {{ field: "gradient_type", type: "nominal" }},
              {{ field: "gradient_norm", type: "quantitative", format: ".6f" }}
            ]
          }}
        }},
        {{
          transform: [
            {{
              window: [{{ op: "mean", field: "gradient_norm", as: "gradient_norm_smoothed" }}],
              frame: [-200, 0],
              groupby: ["gradient_type"],
              sort: [{{ field: "timestep", order: "ascending" }}]
            }}
          ],
          mark: {{ type: "line", strokeWidth: 2 }},
          encoding: {{
            x: {{ field: "timestep", type: "quantitative" }},
            y: {{ field: "gradient_norm_smoothed", type: "quantitative" }},
            color: {{ 
              field: "gradient_type", 
              type: "nominal",
              scale: {{
                domain: ["gradient_norm_policy", "gradient_norm_value"],
                range: ["#1d4ed8", "#c62828"]
              }},
              legend: {{
                title: "Gradient Type",
                labelExpr: "datum.label === 'gradient_norm_policy' ? 'Policy' : 'Value'"
              }}
            }},
            tooltip: [
              {{ field: "timestep", type: "quantitative" }},
              {{ field: "gradient_type", type: "nominal" }},
              {{ field: "gradient_norm_smoothed", type: "quantitative", format: ".6f", title: "Moving Avg" }}
            ]
          }}
        }}
      ],
      title: "Gradient Norms Comparison (Policy vs Value, rolling mean window = 200)"
    }});

    embedChart("#reward-terms", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 960,
      height: 320,
      data: {{ values: stepsData }},
      transform: [
        {{
          fold: [
            "forward_progress",
            "speed_perpendicular_to_heading_penalty",
            "curvature_range_penalty",
            "curvature_oscillation_reward",
            "energy_penalty",
            "smoothness_penalty",
            "alignment_bonus",
            "projected_speed",
            "speed_perpendicular_to_target_penalty"
          ],
          as: ["term", "value"]
        }}
      ],
      mark: "line",
      encoding: {{
        x: {{ field: "timestep", type: "quantitative" }},
        y: {{ field: "value", type: "quantitative" }},
        color: {{ field: "term", type: "nominal" }}
      }},
      title: "Reward term breakdown"
    }});

    embedChart("#curvature-mean", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 360,
      height: 220,
      data: {{ values: curvatureMeanData }},
      mark: "rect",
      encoding: {{
        x: {{ field: "segment", type: "ordinal" }},
        y: {{ field: "band", type: "ordinal" }},
        color: {{ field: "value", type: "quantitative", scale: {{ scheme: "orangered" }} }},
        tooltip: [
          {{ field: "band", type: "ordinal" }},
          {{ field: "segment", type: "ordinal" }},
          {{ field: "value", type: "quantitative", format: ".3f" }}
        ]
      }},
      title: "Mean curvature"
    }});

    embedChart("#curvature-std", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 360,
      height: 220,
      data: {{ values: curvatureStdData }},
      mark: "rect",
      encoding: {{
        x: {{ field: "segment", type: "ordinal" }},
        y: {{ field: "band", type: "ordinal" }},
        color: {{ field: "std", type: "quantitative", scale: {{ scheme: "viridis" }} }},
        tooltip: [
          {{ field: "band", type: "ordinal" }},
          {{ field: "segment", type: "ordinal" }},
          {{ field: "std", type: "quantitative", format: ".3f" }}
        ]
      }},
      title: "Curvature standard deviation"
    }});

    embedChart("#curvature-norm-avg", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 360,
      height: 220,
      data: {{ values: stepsData }},
      layer: [
        {{
          mark: {{ type: "line", color: "#b0bec5", opacity: 0.4 }},
          encoding: {{
            x: {{ field: "timestep", type: "quantitative" }},
            y: {{
              field: "curvature_norm_avg",
              type: "quantitative",
              title: "Mean |curvature|"
            }},
            tooltip: [
              {{ field: "timestep", type: "quantitative" }},
              {{ field: "curvature_norm_avg", type: "quantitative", format: ".3f" }}
            ]
          }}
        }},
        {{
          transform: [
            {{
              window: [
                {{ op: "mean", field: "curvature_norm_avg", as: "curvature_norm_avg_smoothed" }}
              ],
              frame: [-200, 0],
              sort: [{{ field: "timestep", order: "ascending" }}]
            }}
          ],
          mark: {{ type: "line", color: "#00897b", strokeWidth: 2 }},
          encoding: {{
            x: {{ field: "timestep", type: "quantitative" }},
            y: {{ field: "curvature_norm_avg_smoothed", type: "quantitative" }}
          }}
        }}
      ],
      title: "Average curvature magnitude (raw + rolling mean)"
    }});

    const createSectionSpec = (segment) => {{
      return {{
        $schema: "https://vega.github.io/schema/vega-lite/v5.json",
        data: {{ values: curvatureSectionsData }},
        width: 320,
        height: 200,
        transform: [
          {{ filter: {{ field: "segment", equal: segment }} }}
        ],
        layer: [
          {{
            mark: {{ type: "line", color: "#ce93d8", opacity: 0.4 }},
            encoding: {{
              x: {{ field: "timestep", type: "quantitative", title: "Timestep" }},
              y: {{ field: "value", type: "quantitative", title: "|curvature|" }},
              tooltip: [
                {{ field: "timestep", type: "quantitative" }},
                {{ field: "value", type: "quantitative", format: ".3f" }}
              ]
            }}
          }},
          {{
            transform: [
              {{
                window: [
                  {{ op: "mean", field: "value", as: "value_smoothed" }}
                ],
                frame: [-200, 0],
                sort: [{{ field: "timestep", order: "ascending" }}]
              }}
            ],
            mark: {{ type: "line", color: "#7b1fa2", strokeWidth: 2 }},
            encoding: {{
              x: {{ field: "timestep", type: "quantitative" }},
              y: {{ field: "value_smoothed", type: "quantitative" }}
            }}
          }}
        ],
        title: segment
      }};
    }};

    const axisLabelExpr = "abs(datum.value) < 1e-3 ? format(datum.value, '.1e') : abs(datum.value) >= 1000 ? format(datum.value, '.2s') : format(datum.value, '.3f')";

    const createTermSpec = (term) => ({{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      data: {{ 
        values: stepsData.filter(d => d[term] !== undefined && d[term] !== null)
      }},
      width: 320,
      height: 220,
      encoding: {{
        x: {{
          field: "timestep",
          type: "quantitative",
          title: "Timestep"
        }}
      }},
      layer: [
        {{
          mark: {{ type: "line", color: "#b0bec5", opacity: 0.35 }},
          encoding: {{
            y: {{
              field: term,
              type: "quantitative",
              axis: {{ title: null, tickCount: 5, labelExpr: axisLabelExpr }}
            }},
            tooltip: [
              {{ field: "timestep", type: "quantitative" }},
              {{ field: term, type: "quantitative", format: ".3f" }}
            ]
          }}
        }},
        {{
          transform: [
            {{
              window: [{{ op: "mean", field: term, as: "value_smoothed" }}],
              frame: [-200, 0],
              sort: [{{ field: "timestep", order: "ascending" }}]
            }}
          ],
          mark: {{ type: "line", color: "#1976d2", strokeWidth: 2 }},
          encoding: {{
            y: {{ field: "value_smoothed", type: "quantitative" }}
          }}
        }}
      ],
      title: rewardTermLabels[term] || term.replace(/_/g, " ").replace(/\\b\\w/g, l => l.toUpperCase())
    }});

    const termGridEl = document.getElementById("reward-term-grid");
    if (rewardTerms && rewardTerms.length > 0) {{
      rewardTerms.forEach((term) => {{
        // Check if this term exists in any data point
        const hasData = stepsData.some(d => d[term] !== undefined && d[term] !== null);
        if (hasData) {{
          const panel = document.createElement("div");
          panel.className = "term-panel";
          panel.id = `reward-term-${{term}}`;
          termGridEl.appendChild(panel);
          try {{
            embedChart(`#${{panel.id}}`, createTermSpec(term));
          }} catch (e) {{
            console.warn(`Failed to create chart for term ${{term}}:`, e);
          }}
        }}
      }});
    }}

    const sectionNames = Array.from(new Set((curvatureSectionsData || []).map((d) => d.segment)));
    const sectionGridEl = document.getElementById("curvature-section-grid");
    sectionNames.forEach((segment, idx) => {{
      const panel = document.createElement("div");
      panel.className = "term-panel";
      panel.id = `curvature-section-${{idx}}`;
      sectionGridEl.appendChild(panel);
      embedChart(`#${{panel.id}}`, createSectionSpec(segment));
    }});

    const createActionSpec = (field) => ({{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      data: {{ values: stepsData }},
      width: 320,
      height: 220,
      encoding: {{
        x: {{
          field: "timestep",
          type: "quantitative",
          title: "Timestep"
        }}
      }},
      layer: [
        {{
          mark: {{ type: "line", color: "#ffb300", opacity: 0.35 }},
          encoding: {{
            y: {{ field: field, type: "quantitative" }},
            tooltip: [
              {{ field: "timestep", type: "quantitative" }},
              {{ field: field, type: "quantitative", format: ".3f", title: "value" }}
            ]
          }}
        }},
        {{
          transform: [
            {{
              window: [{{ op: "mean", field: field, as: "value_smoothed" }}],
              frame: [-200, 0],
              sort: [{{ field: "timestep", order: "ascending" }}]
            }}
          ],
          mark: {{ type: "line", color: "#f57c00", strokeWidth: 2 }},
          encoding: {{
            y: {{ field: "value_smoothed", type: "quantitative" }}
          }}
        }}
      ],
      title: field.replace("action_", "Action ")
    }});

    const actionGridEl = document.getElementById("action-grid");
    actionFields.forEach((field, idx) => {{
      const panel = document.createElement("div");
      panel.className = "term-panel";
      panel.id = `action-panel-${{idx}}`;
      actionGridEl.appendChild(panel);
      embedChart(`#${{panel.id}}`, createActionSpec(field));
    }});
  </script>
</body>
</html>
"""


def load_data(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def load_config(paths: List[Path]) -> dict | None:
    """Load configuration from JSON file if it exists. Tries multiple paths."""
    for path in paths:
        if path.exists():
            with path.open() as fh:
                return json.load(fh)
    return None


def format_config_value(value) -> str:
    """Format a configuration value for display."""
    if isinstance(value, bool):
        return "Enabled" if value else "Disabled"
    elif isinstance(value, (int, float)):
        if isinstance(value, float) and (value < 0.01 or value > 1000):
            return f"{value:.2e}"
        return str(value)
    elif isinstance(value, list):
        if len(value) == 0:
            return "[]"
        elif isinstance(value[0], dict):
            # Handle nested structures like net_arch
            return json.dumps(value, indent=2)
        else:
            return ", ".join(str(v) for v in value)
    elif isinstance(value, dict):
        return json.dumps(value, indent=2)
    else:
        return str(value)


def format_config_in_english(config: dict | None) -> str:
    """Format configuration dictionary into readable English HTML."""
    if not config:
        return "<p><em>Configuration file not found.</em></p>"
    
    # Define human-readable labels for configuration keys
    labels = {
        "ENV_CONFIG": {
            "fixed_wavelength": "Fixed Wavelength",
            "obs_keys": "Observation Keys",
            "period": "Period (seconds)",
            "ratio_time": "Ratio Time",
            "rut_ratio": "Rut Ratio",
            "_n_elem": "Number of Elements",
            "max_episode_length": "Max Episode Length (seconds)",
        },
        "REWARD_WEIGHTS": {
            "forward_progress": "Forward Progress Weight",
            "speed_perpendicular_to_heading_penalty": "Speed Perpendicular to Heading Penalty",
            "speed_perpendicular_to_target_penalty": "Speed Perpendicular to Target Penalty",
            "curvature_range_penalty": "Curvature Range Penalty",
            "curvature_oscillation_reward": "Curvature Oscillation Reward",
            "energy_penalty": "Energy Penalty",
            "smoothness_penalty": "Smoothness Penalty",
            "alignment_bonus": "Alignment Bonus",
            "streak_bonus": "Streak Bonus",
            "projected_speed": "Projected Speed Weight",
        },
        "TRAIN_CONFIG": {
            "total_timesteps": "Total Timesteps",
            "print_freq": "Print Frequency",
            "step_info_keys": "Step Info Keys",
            "print_exclude_keys": "Print Exclude Keys",
            "save_freq": "Save Frequency",
            "save_steps": "Save Steps",
            "checkpoint_freq": "Checkpoint Frequency",
        },
        "MODEL_CONFIG": {
            "n_steps": "Rollout Buffer Size (n_steps)",
            "gae_lambda": "GAE Lambda (λ)",
            "gamma": "Discount Factor (γ)",
            "policy": "Policy Type",
            "verbose": "Verbose Output",
            "use_gpu": "Use GPU",
            "use_layer_norm": "Layer Normalization",
            "net_arch": "Network Architecture",
            "use_orthogonal_init": "Orthogonal Weight Initialization",
            "normalize_observations": "Normalize Observations",
            "normalize_observations_training": "Update Normalization During Training",
            "clip_obs": "Observation Clipping Value",
            "max_grad_norm": "Max Gradient Norm",
        },
        "PATHS": {
            "log_dir": "Log Directory",
            "model_dir": "Model Directory",
            "model_name": "Model Name",
            "checkpoint_name": "Checkpoint Name",
            "policy_gradient_viz_dir": "Policy Gradient Visualization Directory",
        },
    }
    
    # Category descriptions
    category_descriptions = {
        "ENV_CONFIG": "Environment Configuration",
        "REWARD_WEIGHTS": "Reward Weights",
        "TRAIN_CONFIG": "Training Configuration",
        "MODEL_CONFIG": "Model Configuration",
        "PATHS": "File Paths",
    }
    
    html_parts = []
    
    for category, category_data in config.items():
        if not isinstance(category_data, dict):
            continue
            
        category_label = category_descriptions.get(category, category)
        html_parts.append(f'<div class="config-category">')
        html_parts.append(f'<h3>{category_label}</h3>')
        
        category_labels = labels.get(category, {})
        
        for key, value in category_data.items():
            label = category_labels.get(key, key.replace("_", " ").title())
            formatted_value = format_config_value(value)
            html_parts.append(
                f'<div class="config-item">'
                f'<span class="config-label">{label}:</span>'
                f'<span class="config-value">{formatted_value}</span>'
                f'</div>'
            )
        
        html_parts.append('</div>')
    
    return "\n".join(html_parts)


def prepare_datasets(
    data: dict,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict], List[str], List[Dict], List[Dict], List[str]]:
    # Handle empty steps data
    if not data.get("steps"):
        return [], [], [], [], [], [], [], [], []
    
    # Calculate episode data from steps if episodes arrays are empty
    episodes_data = data.get("episodes", {})
    rewards_list = episodes_data.get("rewards", [])
    lengths_list = episodes_data.get("lengths", [])
    timesteps_list = episodes_data.get("timesteps_at_episode", [])
    
    # If episodes data is empty, compute from steps
    if not rewards_list or not lengths_list:
        episode_rewards = {}
        episode_lengths = {}
        episode_sim_times = {}
        episode_timesteps = {}
        
        for step in data["steps"]:
            ep = step.get("episode", 0)
            reward = step.get("reward", 0.0)
            sim_time = step.get("sim_time")
            
            # Accumulate reward and count steps per episode
            if ep not in episode_rewards:
                episode_rewards[ep] = 0.0
                episode_lengths[ep] = 0
                episode_timesteps[ep] = step.get("timestep", 0)
                episode_sim_times[ep] = []
            
            episode_rewards[ep] += reward
            episode_lengths[ep] += 1
            if sim_time is not None:
                episode_sim_times[ep].append(sim_time)
        
        # Calculate total sim_time per episode (max sim_time in episode)
        total_sim_time_per_ep = {}
        for ep, sim_times in episode_sim_times.items():
            total_sim_time_per_ep[ep] = max(sim_times) if sim_times else 0.0
        
        # Build episodes list
        episodes = []
        max_episode = max(episode_rewards.keys()) if episode_rewards else -1
        cumulative_timesteps = 0
        
        for idx in range(max_episode + 1):
            if idx in episode_rewards:
                reward = episode_rewards[idx]
                length = episode_lengths[idx]
                cumulative_timesteps += length
                reward_per_timestep = reward / length if length > 0 else 0.0
                total_sim_time = total_sim_time_per_ep.get(idx, 0.0)
                reward_per_simtime = reward / total_sim_time if total_sim_time > 0 else 0.0
                
                episodes.append({
                    "episode": idx,
                    "reward": reward,
                    "length": length,
                    "timesteps": cumulative_timesteps,
                    "reward_per_timestep": reward_per_timestep,
                    "reward_per_simtime": reward_per_simtime,
                })
    else:
        # Use existing episodes data
        # Calculate total sim_time per episode from steps data (use max sim_time as episode length)
        episode_sim_times = {}
        for step in data["steps"]:
            ep = step.get("episode", 0)
            sim_time = step.get("sim_time")
            if sim_time is not None:
                if ep not in episode_sim_times:
                    episode_sim_times[ep] = []
                episode_sim_times[ep].append(sim_time)
        
        # Calculate total sim_time per episode (max sim_time in episode)
        total_sim_time_per_ep = {}
        for ep, sim_times in episode_sim_times.items():
            total_sim_time_per_ep[ep] = max(sim_times) if sim_times else 0.0
        
        episodes = []
        for idx, reward in enumerate(rewards_list):
            length = lengths_list[idx] if idx < len(lengths_list) else 0
            reward_per_timestep = reward / length if length > 0 else 0.0
            total_sim_time = total_sim_time_per_ep.get(idx, 0.0)
            reward_per_simtime = reward / total_sim_time if total_sim_time > 0 else 0.0
            timesteps = timesteps_list[idx] if idx < len(timesteps_list) else 0
            
            episodes.append({
                "episode": idx,
                "reward": reward,
                "length": length,
                "timesteps": timesteps,
                "reward_per_timestep": reward_per_timestep,
                "reward_per_simtime": reward_per_simtime,
            })

    steps_data: List[Dict] = []
    curvature_sections: List[Dict] = []
    
    # Check if curvatures are available in the data
    has_curvatures = False
    segments = 0
    if data["steps"] and len(data["steps"]) > 0:
        first_step = data["steps"][0]
        if "curvatures" in first_step:
            try:
                curvatures = first_step["curvatures"]
                if isinstance(curvatures, list) and len(curvatures) > 0:
                    if isinstance(curvatures[0], list) and len(curvatures[0]) > 0:
                        segments = len(curvatures[0])
                        has_curvatures = True
            except (KeyError, IndexError, TypeError):
                has_curvatures = False
    
    action_fields: List[str] = []
    for step in data["steps"]:
        segment_norms = []
        
        # Process curvatures if available
        if has_curvatures and "curvatures" in step:
            curvature_components = step["curvatures"]
            if isinstance(curvature_components, list) and len(curvature_components) >= 3:
                for seg_idx in range(segments):
                    try:
                        x = curvature_components[0][seg_idx]
                        y = curvature_components[1][seg_idx]
                        z = curvature_components[2][seg_idx]
                        segment_norm = math.sqrt(x * x + y * y + z * z)
                        segment_norms.append(segment_norm)
                        curvature_sections.append(
                            {
                                "timestep": step["timestep"],
                                "segment": f"Section {seg_idx + 1}",
                                "value": segment_norm,
                            }
                        )
                    except (IndexError, TypeError):
                        pass

        entry = {
            "timestep": step.get("timestep", 0),
            "reward": step.get("reward", 0.0),
            "sim_time": step.get("sim_time"),
            "episode": step.get("episode", 0),
            "curvature_norm_avg": sum(segment_norms) / segments if segment_norms and segments > 0 else 0.0,
        }
        
        # Add optional fields if they exist
        if "speed_along_heading" in step:
            entry["speed_along_heading"] = step["speed_along_heading"]
        if "speed_perpendicular_to_heading" in step:
            entry["speed_perpendicular_to_heading"] = step["speed_perpendicular_to_heading"]
        if "velocity_projection" in step:
            entry["velocity_projection"] = step["velocity_projection"]
        
        # Add gradient norms if available
        if "gradient_norm_policy" in step:
            entry["gradient_norm_policy"] = step["gradient_norm_policy"]
        if "gradient_norm_value" in step:
            entry["gradient_norm_value"] = step["gradient_norm_value"]
        
        # Process actions if available
        actions = step.get("action", [])
        if actions:
            for idx, value in enumerate(actions):
                field_name = f"action_{idx + 1}"
                entry[field_name] = value
                if idx >= len(action_fields):
                    action_fields.append(field_name)
        
        # Add reward terms if available (could be a dict or individual fields)
        # Prioritize reward_terms dict values over top-level fields
        reward_terms_dict = {}
        if "reward_terms" in step:
            if isinstance(step["reward_terms"], dict):
                reward_terms_dict = step["reward_terms"]
        
        # Check for individual reward term fields that might be in step_info_keys
        # Use reward_terms dict value if available, otherwise use top-level field
        for term in REWARD_TERMS:
            if term in reward_terms_dict:
                entry[term] = reward_terms_dict[term]
            elif term in step:
                entry[term] = step[term]
        
        steps_data.append(entry)

    # Compute curvature stats only if curvatures are available
    if has_curvatures:
        mean_grid, std_grid = compute_curvature_stats(data["steps"])
    else:
        mean_grid, std_grid = [], []
    mean_records = []
    std_records = []
    for band_idx, (mean_row, std_row) in enumerate(zip(mean_grid, std_grid)):
        for seg_idx, (mean_val, std_val) in enumerate(zip(mean_row, std_row)):
            mean_records.append({"band": f"band_{band_idx}", "segment": f"seg_{seg_idx}", "value": mean_val})
            std_records.append({"band": f"band_{band_idx}", "segment": f"seg_{seg_idx}", "std": std_val})

    # Calculate early vs late episode statistics
    num_episodes = len(episodes)
    split_point = max(1, num_episodes // 4)  # First 25% vs Last 25%
    
    early_episodes = episodes[:split_point]
    late_episodes = episodes[-split_point:] if num_episodes > split_point * 2 else []
    
    early_late_comparison = []
    if early_episodes and late_episodes:
        # Calculate mean reward for early vs late
        early_mean_reward = sum(ep["reward"] for ep in early_episodes) / len(early_episodes)
        late_mean_reward = sum(ep["reward"] for ep in late_episodes) / len(late_episodes)
        early_mean_reward_per_timestep = sum(ep["reward_per_timestep"] for ep in early_episodes) / len(early_episodes)
        late_mean_reward_per_timestep = sum(ep["reward_per_timestep"] for ep in late_episodes) / len(late_episodes)
        
        early_late_comparison = [
            {"period": "Early (First 25%)", "metric": "Mean Reward", "value": early_mean_reward},
            {"period": "Late (Last 25%)", "metric": "Mean Reward", "value": late_mean_reward},
            {"period": "Early (First 25%)", "metric": "Reward per Timestep", "value": early_mean_reward_per_timestep},
            {"period": "Late (Last 25%)", "metric": "Reward per Timestep", "value": late_mean_reward_per_timestep},
        ]
    
    # Calculate reward variance over time (rolling variance of episode rewards)
    reward_variance_data = []
    window_size = 10
    for i in range(num_episodes):
        start_idx = max(0, i - window_size + 1)
        window_episodes = episodes[start_idx:i+1]
        if len(window_episodes) >= 2:
            window_rewards = [ep["reward"] for ep in window_episodes]
            mean_reward = sum(window_rewards) / len(window_rewards)
            variance = sum((r - mean_reward) ** 2 for r in window_rewards) / len(window_rewards)
            std_dev = math.sqrt(variance)
            reward_variance_data.append({
                "episode": i,
                "variance": variance,
                "std_dev": std_dev,
            })
    
    # Dynamically detect which reward terms are actually present in the data
    # First, check if reward_terms dict exists in any step
    reward_terms_from_dict = set()
    for step in steps_data:
        if "reward_terms" in step and isinstance(step["reward_terms"], dict):
            reward_terms_from_dict.update(step["reward_terms"].keys())
    
    # Exclude non-reward-term fields
    excluded_fields = {
        "timestep", "reward", "sim_time", "episode", "curvature_norm_avg",
        "speed_along_heading", "speed_perpendicular_to_heading", "velocity_projection",
        "forward_progress", "alignment", "alignment_streak", "alignment_goal_met",
        "speed_perpendicular_to_target", "lateral_speed_perpendicular", "position",
        "heading_dir", "current_time", "speed", "captured_at",
        "gradient_norm_policy", "gradient_norm_value", "reward_terms"
    }
    # Also exclude action fields
    excluded_fields.update({f"action_{i+1}" for i in range(20)})  # Cover up to 20 actions
    
    # Find reward terms that are actually present in the data
    detected_reward_terms = []
    for term in REWARD_TERMS:
        # Check if term exists as a direct field in any step
        if any(term in step and step[term] is not None for step in steps_data):
            detected_reward_terms.append(term)
        # Or if it's in the reward_terms dict
        elif term in reward_terms_from_dict:
            detected_reward_terms.append(term)

    return episodes, steps_data, mean_records, std_records, curvature_sections, action_fields, early_late_comparison, reward_variance_data, detected_reward_terms


def compute_curvature_stats(steps: List[dict]) -> Tuple[List[List[float]], List[List[float]]]:
    """Compute curvature statistics. Returns empty lists if curvatures are not available."""
    if not steps or "curvatures" not in steps[0]:
        return [], []
    
    try:
        count = len(steps)
        bands = len(steps[0]["curvatures"])
        segments = len(steps[0]["curvatures"][0])
        sums = [[0.0 for _ in range(segments)] for _ in range(bands)]
        sq_sums = [[0.0 for _ in range(segments)] for _ in range(bands)]

        for entry in steps:
            if "curvatures" not in entry:
                continue
            curvatures = entry["curvatures"]
            if not isinstance(curvatures, list) or len(curvatures) < bands:
                continue
            for b in range(bands):
                if not isinstance(curvatures[b], list) or len(curvatures[b]) < segments:
                    continue
                for seg in range(segments):
                    try:
                        val = curvatures[b][seg]
                        sums[b][seg] += val
                        sq_sums[b][seg] += val * val
                    except (IndexError, TypeError):
                        pass

        means = [[sums[b][seg] / count for seg in range(segments)] for b in range(bands)]
        stds = [
            [
                math.sqrt(max(sq_sums[b][seg] / count - means[b][seg] ** 2, 0.0))
                for seg in range(segments)
            ]
            for b in range(bands)
        ]
        return means, stds
    except (KeyError, IndexError, TypeError):
        return [], []


def render_html(
    metadata: dict,
    episodes: List[Dict],
    steps: List[Dict],
    curvature_mean: List[Dict],
    curvature_std: List[Dict],
    curvature_sections: List[Dict],
    action_fields: List[str],
    early_late_comparison: List[Dict],
    reward_variance: List[Dict],
    configuration_html: str,
    detected_reward_terms: List[str],
) -> str:
    return HTML_TEMPLATE.format(
        saved_at=metadata["saved_at"],
        total_timesteps=metadata["total_timesteps"],
        episode_count=metadata["episode_count"],
        configuration_html=configuration_html,
        episodes_data=json.dumps(episodes),
        steps_data=json.dumps(steps),
        curvature_mean_data=json.dumps(curvature_mean),
        curvature_std_data=json.dumps(curvature_std),
        reward_terms=json.dumps(detected_reward_terms),
        curvature_sections_data=json.dumps(curvature_sections),
        action_fields=json.dumps(action_fields),
        early_late_comparison_data=json.dumps(early_late_comparison),
        reward_variance_data=json.dumps(reward_variance),
    )


def html_to_png(html_path: Path, png_path: Path, wait_time: int = 5000) -> None:
    """Convert HTML file to PNG using Playwright."""
    if not PLAYWRIGHT_AVAILABLE:
        print("Warning: playwright not available. Install with: pip install playwright && playwright install chromium")
        return
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{html_path.resolve()}")
        # Wait for Vega charts to render (they need time to load and render)
        page.wait_for_timeout(wait_time)
        # Take full page screenshot
        page.screenshot(path=str(png_path), full_page=True)
        browser.close()
    print(f"Saved PNG: {png_path}")


def main() -> None:
    data = load_data(DATA_PATH)
    (
        episodes,
        steps_data,
        mean_records,
        std_records,
        curvature_sections,
        action_fields,
        early_late_comparison,
        reward_variance,
        detected_reward_terms,
    ) = prepare_datasets(data)
    
    # Load and format configuration
    config = load_config(CONFIG_PATHS)
    configuration_html = format_config_in_english(config) if config else format_config_in_english(None)
    
    # Update metadata episode_count if it's 0 or missing
    metadata = data["metadata"].copy()
    if metadata.get("episode_count", 0) == 0 and episodes:
        metadata["episode_count"] = len(episodes)
    
    html = render_html(
        metadata,
        episodes,
        steps_data,
        mean_records,
        std_records,
        curvature_sections,
        action_fields,
        early_late_comparison,
        reward_variance,
        configuration_html,
        detected_reward_terms,
    )
    OUTPUT_PATH.write_text(html)
    print(f"Wrote {OUTPUT_PATH}")
    
    # Convert HTML to PNG
    png_path = OUTPUT_PATH.with_suffix('.png')
    html_to_png(OUTPUT_PATH, png_path)


if __name__ == "__main__":
    main()

