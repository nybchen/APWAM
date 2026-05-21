import argparse
import json
from pathlib import Path

import h5py


LABEL_NAMES = {
    0: "perception",
    1: "action",
}


def _sorted_traj_keys(h5_file):
    keys = [k for k in h5_file.keys() if k.startswith("traj_")]
    return sorted(keys, key=lambda k: int(k.split("_", 1)[1]))


def _action_len(actions_group):
    if isinstance(actions_group, h5py.Dataset):
        return len(actions_group)
    first_key = next(iter(actions_group.keys()))
    return len(actions_group[first_key])


def _read_labels(traj_group):
    labels = traj_group["mode_labels"]
    if isinstance(labels, h5py.Dataset):
        return {"global": [int(x) for x in labels[:]]}
    return {
        agent_id: [int(x) for x in labels[agent_id][:]]
        for agent_id in sorted(labels.keys())
    }


def build_manifest(
    record_dir: Path,
    output_dir: Path,
    fps: int,
    include_unlabeled: bool,
    include_history: bool,
):
    demos = []
    skipped_unlabeled = 0
    h5_paths = sorted(record_dir.glob("*/motionplanning/*.h5"))
    if not include_history:
        latest_by_task = {}
        for h5_path in h5_paths:
            task_name = h5_path.parents[1].name
            if task_name not in latest_by_task or h5_path.stat().st_mtime > latest_by_task[task_name].stat().st_mtime:
                latest_by_task[task_name] = h5_path
        h5_paths = sorted(latest_by_task.values())
    for h5_path in h5_paths:
        task_name = h5_path.parents[1].name
        motion_dir = h5_path.parent
        with h5py.File(h5_path, "r") as h5_file:
            for traj_key in _sorted_traj_keys(h5_file):
                traj_id = int(traj_key.split("_", 1)[1])
                traj_group = h5_file[traj_key]
                has_per_agent_labels = (
                    "mode_labels" in traj_group and isinstance(traj_group["mode_labels"], h5py.Group)
                )
                if not has_per_agent_labels:
                    skipped_unlabeled += 1
                    if not include_unlabeled:
                        continue
                    if "mode_labels" in traj_group and isinstance(traj_group["mode_labels"], h5py.Dataset):
                        labels = _read_labels(traj_group)
                    elif isinstance(traj_group["actions"], h5py.Group):
                        labels = {
                            agent_id: [1] * len(traj_group["actions"][agent_id])
                            for agent_id in sorted(traj_group["actions"].keys())
                        }
                    else:
                        labels = {"global": [1] * _action_len(traj_group["actions"])}
                else:
                    labels = _read_labels(traj_group)
                video_path = motion_dir / f"{traj_id}.mp4"
                if not video_path.exists():
                    continue
                rel_video = Path("..") / video_path.relative_to(output_dir.parent)
                demos.append(
                    {
                        "task": task_name,
                        "traj": traj_id,
                        "h5": str(h5_path.relative_to(output_dir.parent)),
                        "video": str(rel_video),
                        "fps": fps,
                        "labelsByAgent": labels,
                        "labelNames": LABEL_NAMES,
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {"demos": demos}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, manifest, len(demos), skipped_unlabeled


def write_index(output_dir: Path, manifest: dict):
    index = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RoboFactory Mode Label Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #637083;
      --line: #d8dee8;
      --action: #2563eb;
      --perception: #0f8b5f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 16px;
    }
    h1 {
      font-size: 24px;
      line-height: 1.15;
      margin: 0;
      font-weight: 700;
      letter-spacing: 0;
    }
    .controls {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) minmax(120px, 180px) auto auto;
      gap: 10px;
      align-items: center;
      margin-bottom: 14px;
    }
    select, button, input[type="range"] {
      height: 36px;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
      color: var(--ink);
      font: inherit;
    }
    select { padding: 0 10px; }
    button {
      min-width: 72px;
      padding: 0 12px;
      cursor: pointer;
    }
    .stage {
      background: #111827;
      border-radius: 8px;
      overflow: hidden;
      position: relative;
      border: 1px solid #111827;
    }
    video {
      display: block;
      width: 100%;
      max-height: 70vh;
      background: #111827;
    }
    .badge {
      position: absolute;
      left: 14px;
      top: 14px;
      padding: 7px 10px;
      border-radius: 6px;
      color: #fff;
      font-weight: 700;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .badge.action { background: var(--action); }
    .badge.perception { background: var(--perception); }
    .meta {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 14px 0;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 3px;
    }
    .metric strong {
      font-size: 16px;
      letter-spacing: 0;
    }
    .timeline {
      display: grid;
      grid-template-columns: repeat(var(--n), minmax(1px, 1fr));
      height: 22px;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: var(--panel);
    }
    .tick.action { background: var(--action); }
    .tick.perception { background: var(--perception); }
    .tick.current { outline: 2px solid #111827; outline-offset: -2px; }
    .agent-labels {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }
    .agent-row {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }
    .agent-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }
    .agent-name {
      font-weight: 700;
      font-size: 14px;
    }
    .agent-mode {
      padding: 5px 8px;
      border-radius: 6px;
      color: #fff;
      font-weight: 700;
      font-size: 12px;
      text-transform: uppercase;
    }
    .agent-mode.action { background: var(--action); }
    .agent-mode.perception { background: var(--perception); }
    .empty {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      color: var(--muted);
    }
    @media (max-width: 760px) {
      main { padding: 14px; }
      header, .controls { grid-template-columns: 1fr; display: grid; }
      .meta { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>RoboFactory Mode Label Viewer</h1>
    </header>
    <section class="controls">
      <select id="task"></select>
      <select id="traj"></select>
      <button id="play">Play</button>
      <button id="prev">Prev</button>
      <button id="next">Next</button>
    </section>
    <section id="content"></section>
  </main>
  <script>
    const EMBEDDED_MANIFEST = __MANIFEST_JSON__;
    const labelName = id => id === 0 ? "perception" : "action";
    const state = { demos: [], demo: null };
    const taskSelect = document.querySelector("#task");
    const trajSelect = document.querySelector("#traj");
    const content = document.querySelector("#content");
    const playBtn = document.querySelector("#play");
    const prevBtn = document.querySelector("#prev");
    const nextBtn = document.querySelector("#next");

    function renderEmpty(text) {
      content.innerHTML = `<div class="empty">${text}</div>`;
    }

    function selectDemo() {
      const task = taskSelect.value;
      const traj = Number(trajSelect.value);
      state.demo = state.demos.find(d => d.task === task && d.traj === traj);
      renderDemo();
    }

    function renderSelectors() {
      const tasks = [...new Set(state.demos.map(d => d.task))];
      taskSelect.innerHTML = tasks.map(t => `<option value="${t}">${t}</option>`).join("");
      function updateTrajOptions() {
        const demos = state.demos.filter(d => d.task === taskSelect.value);
        trajSelect.innerHTML = demos.map(d => `<option value="${d.traj}">traj_${d.traj}</option>`).join("");
      }
      taskSelect.onchange = () => { updateTrajOptions(); selectDemo(); };
      trajSelect.onchange = selectDemo;
      updateTrajOptions();
      selectDemo();
    }

    function renderDemo() {
      const d = state.demo;
      if (!d) return renderEmpty("No trajectory selected.");
      const labelsByAgent = d.labelsByAgent || { global: d.labels || [] };
      const agents = Object.keys(labelsByAgent).sort();
      const n = agents.length ? labelsByAgent[agents[0]].length : 0;
      const agentRows = agents.map(agent => `
        <div class="agent-row">
          <div class="agent-head">
            <div class="agent-name">${agent}</div>
            <div id="mode-${agent}" class="agent-mode action">action</div>
          </div>
          <div class="timeline" style="--n:${Math.max(n, 1)}">
            ${labelsByAgent[agent].map((x, i) => `<div class="tick ${labelName(x)} tick-${agent}" data-i="${i}"></div>`).join("")}
          </div>
        </div>`).join("");
      content.innerHTML = `
        <div class="stage">
          <video id="video" src="${d.video}" controls preload="metadata"></video>
          <div id="badge" class="badge action">per-arm labels</div>
        </div>
        <div class="meta">
          <div class="metric"><span>Task</span><strong>${d.task}</strong></div>
          <div class="metric"><span>Trajectory</span><strong>traj_${d.traj}</strong></div>
          <div class="metric"><span>Frame</span><strong id="frame">0 / ${Math.max(n - 1, 0)}</strong></div>
          <div class="metric"><span>Arms</span><strong>${agents.join(", ")}</strong></div>
        </div>
        <input id="scrub" type="range" min="0" max="${Math.max(n - 1, 0)}" value="0" />
        <div class="agent-labels">${agentRows}</div>`;
      const video = document.querySelector("#video");
      const badge = document.querySelector("#badge");
      const frame = document.querySelector("#frame");
      const scrub = document.querySelector("#scrub");
      const ticks = [...document.querySelectorAll(".tick")];
      function setFrame(i, seek) {
        i = Math.max(0, Math.min(n - 1, i));
        const frameModes = agents.map(agent => `${agent}:${labelName(labelsByAgent[agent][i])}`);
        badge.textContent = frameModes.join("  ");
        badge.className = "badge action";
        for (const agent of agents) {
          const name = labelName(labelsByAgent[agent][i]);
          const el = document.getElementById(`mode-${agent}`);
          el.textContent = name;
          el.className = `agent-mode ${name}`;
        }
        frame.textContent = `${i} / ${Math.max(n - 1, 0)}`;
        scrub.value = i;
        ticks.forEach(t => t.classList.toggle("current", Number(t.dataset.i) === i));
        if (seek && Number.isFinite(video.duration)) video.currentTime = i / d.fps;
      }
      video.ontimeupdate = () => setFrame(Math.floor(video.currentTime * d.fps), false);
      scrub.oninput = () => setFrame(Number(scrub.value), true);
      playBtn.onclick = () => video.paused ? video.play() : video.pause();
      prevBtn.onclick = () => setFrame(Number(scrub.value) - 1, true);
      nextBtn.onclick = () => setFrame(Number(scrub.value) + 1, true);
      setFrame(0, false);
    }

    state.demos = EMBEDDED_MANIFEST.demos || [];
    if (!state.demos.length) renderEmpty("No labeled h5/mp4 pairs found.");
    else renderSelectors();
  </script>
</body>
</html>
"""
    index = index.replace("__MANIFEST_JSON__", json.dumps(manifest))
    (output_dir / "index.html").write_text(index, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build a static video + mode-label viewer.")
    parser.add_argument("--record-dir", type=Path, default=Path("demos"))
    parser.add_argument("--output-dir", type=Path, default=Path("label_viewer"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--include-unlabeled", action="store_true", help="Include old trajectories without mode_labels as all-action.")
    parser.add_argument("--include-history", action="store_true", help="Include all labeled h5 files instead of only the latest h5 per task.")
    args = parser.parse_args()

    manifest_path, manifest, count, skipped_unlabeled = build_manifest(
        args.record_dir, args.output_dir, args.fps, args.include_unlabeled, args.include_history
    )
    write_index(args.output_dir, manifest)
    print(f"Wrote {args.output_dir / 'index.html'}")
    print(f"Wrote {manifest_path} with {count} trajectories")
    if skipped_unlabeled and not args.include_unlabeled:
        print(f"Skipped {skipped_unlabeled} trajectories without mode_labels")


if __name__ == "__main__":
    main()
