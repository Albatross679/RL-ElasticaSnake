#!/usr/bin/env python3
"""Produce an interactive Vega-Lite report for training_data.json."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
# DATA_PATH = ROOT / "Training" / "Logs" / "training_data.json"
DATA_PATH = ROOT / "Training" / "Logs" / "test_training_data.json"
OUTPUT_PATH = ROOT / "Training" / "Results" / "training_data_overview.html"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

REWARD_TERMS = [
    "forward_progress",
    "projected_speed",
    "alignment_bonus",
    "curvature_oscillation_reward",
    "curvature_range_penalty",
    "smoothness_penalty",
    "lateral_penalty",
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
  </style>
</head>
<body>
  <h1>Training Data Overview</h1>
  <section class="metadata">
    <div><strong>Saved at</strong><br>{saved_at}</div>
    <div><strong>Total timesteps</strong><br>{total_timesteps:,}</div>
    <div><strong>Episode count</strong><br>{episode_count}</div>
  </section>

  <h2>Episode metrics</h2>
  <div class="chart-grid">
    <div id="episodes-reward"></div>
    <div id="episodes-length"></div>
    <div id="episodes-timesteps"></div>
  </div>
  <div class="chart-full" id="reward-frequency"></div>

  <h2>Step-level signals</h2>
  <div class="chart-full" id="step-reward"></div>
  <div class="chart-full" id="step-speed"></div>
  <div class="chart-full" id="step-simtime"></div>

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
    const rewardTermLabels = {{
      forward_progress: "Forward progress",
      projected_speed: "Projected speed",
      alignment_bonus: "Alignment bonus",
      curvature_oscillation_reward: "Curvature oscillation reward",
      curvature_range_penalty: "Curvature range penalty",
      smoothness_penalty: "Smoothness penalty",
      lateral_penalty: "Lateral penalty"
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
      mark: {{ type: "line", color: "#42a5f5" }},
      encoding: {{
        x: {{ field: "episode", type: "quantitative" }},
        y: {{ field: "reward", type: "quantitative" }},
        tooltip: [
          {{ field: "episode", type: "quantitative" }},
          {{ field: "reward", type: "quantitative", format: ".2f" }}
        ]
      }},
      title: "Episode reward"
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
            "forward_speed",
            "velocity_projection",
            "lateral_speed"
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

    embedChart("#reward-terms", {{
      $schema: "https://vega.github.io/schema/vega-lite/v5.json",
      width: 960,
      height: 320,
      data: {{ values: stepsData }},
      transform: [
        {{
          fold: [
            "forward_progress",
            "projected_speed",
            "alignment_bonus",
            "curvature_oscillation_reward",
            "curvature_range_penalty",
            "smoothness_penalty",
            "lateral_penalty"
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

    const axisLabelExpr = "abs(datum.value) < 1e-3 ? format(datum.value, '.1e') : abs(datum.value) >= 1000 ? format(datum.value, '.2s') : format(datum.value, '.3f')";

    const createTermSpec = (term) => ({{
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
      title: rewardTermLabels[term]
    }});

    const termGridEl = document.getElementById("reward-term-grid");
    rewardTerms.forEach((term) => {{
      const panel = document.createElement("div");
      panel.className = "term-panel";
      panel.id = `reward-term-${{term}}`;
      termGridEl.appendChild(panel);
      embedChart(`#${{panel.id}}`, createTermSpec(term));
    }});
  </script>
</body>
</html>
"""


def load_data(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def prepare_datasets(data: dict) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    episodes = [
        {
            "episode": idx,
            "reward": reward,
            "length": data["episodes"]["lengths"][idx],
            "timesteps": data["episodes"]["timesteps_at_episode"][idx],
        }
        for idx, reward in enumerate(data["episodes"]["rewards"])
    ]

    steps_data = []
    for step in data["steps"]:
        entry = {
            "timestep": step["timestep"],
            "reward": step["reward"],
            "sim_time": step["sim_time"],
            "forward_speed": step["forward_speed"],
            "lateral_speed": step["lateral_speed"],
            "velocity_projection": step["velocity_projection"],
            "episode": step["episode"],
        }
        entry.update(step["reward_terms"])
        steps_data.append(entry)

    mean_grid, std_grid = compute_curvature_stats(data["steps"])
    mean_records = []
    std_records = []
    for band_idx, (mean_row, std_row) in enumerate(zip(mean_grid, std_grid)):
        for seg_idx, (mean_val, std_val) in enumerate(zip(mean_row, std_row)):
            mean_records.append({"band": f"band_{band_idx}", "segment": f"seg_{seg_idx}", "value": mean_val})
            std_records.append({"band": f"band_{band_idx}", "segment": f"seg_{seg_idx}", "std": std_val})

    return episodes, steps_data, mean_records, std_records


def compute_curvature_stats(steps: List[dict]) -> Tuple[List[List[float]], List[List[float]]]:
    count = len(steps)
    bands = len(steps[0]["curvatures"])
    segments = len(steps[0]["curvatures"][0])
    sums = [[0.0 for _ in range(segments)] for _ in range(bands)]
    sq_sums = [[0.0 for _ in range(segments)] for _ in range(bands)]

    for entry in steps:
        curvatures = entry["curvatures"]
        for b in range(bands):
            for seg in range(segments):
                val = curvatures[b][seg]
                sums[b][seg] += val
                sq_sums[b][seg] += val * val

    means = [[sums[b][seg] / count for seg in range(segments)] for b in range(bands)]
    stds = [
        [
            math.sqrt(max(sq_sums[b][seg] / count - means[b][seg] ** 2, 0.0))
            for seg in range(segments)
        ]
        for b in range(bands)
    ]
    return means, stds


def render_html(metadata: dict, episodes: List[Dict], steps: List[Dict], curvature_mean: List[Dict], curvature_std: List[Dict]) -> str:
    return HTML_TEMPLATE.format(
        saved_at=metadata["saved_at"],
        total_timesteps=metadata["total_timesteps"],
        episode_count=metadata["episode_count"],
        episodes_data=json.dumps(episodes),
        steps_data=json.dumps(steps),
        curvature_mean_data=json.dumps(curvature_mean),
        curvature_std_data=json.dumps(curvature_std),
        reward_terms=json.dumps(REWARD_TERMS),
    )


def main() -> None:
    data = load_data(DATA_PATH)
    episodes, steps_data, mean_records, std_records = prepare_datasets(data)
    html = render_html(data["metadata"], episodes, steps_data, mean_records, std_records)
    OUTPUT_PATH.write_text(html)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

