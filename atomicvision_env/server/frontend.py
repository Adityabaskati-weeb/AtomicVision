"""Custom browser UI for the AtomicVision Hugging Face Space."""

from __future__ import annotations

from textwrap import dedent


def render_home_html() -> str:
    """Return the interactive lab-console UI for the Space root route."""

    return dedent(
        """\
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>AtomicVision Lab Console</title>
            <style>
              :root {
                color-scheme: dark;
                --bg: #071019;
                --panel: #0e1823;
                --panel-strong: #132231;
                --panel-soft: #0b141d;
                --border: rgba(164, 186, 208, 0.14);
                --text: #eef4fb;
                --muted: #9ab0c7;
                --teal: #59d2bc;
                --cyan: #6eb7ff;
                --amber: #f3bf65;
                --rose: #ff8f8f;
                --good: #61e29f;
                --shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
                font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: var(--bg);
                color: var(--text);
              }

              * {
                box-sizing: border-box;
              }

              body {
                margin: 0;
                min-height: 100vh;
                background:
                  radial-gradient(circle at top left, rgba(89, 210, 188, 0.12), transparent 28rem),
                  radial-gradient(circle at top right, rgba(110, 183, 255, 0.12), transparent 30rem),
                  linear-gradient(180deg, #071019 0%, #09131d 35%, #08111a 100%);
              }

              a {
                color: inherit;
              }

              button,
              input,
              select {
                font: inherit;
              }

              .shell {
                width: min(1480px, calc(100vw - 32px));
                margin: 0 auto;
                padding: 18px 0 28px;
              }

              .topbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 16px;
                padding: 0 0 14px;
              }

              .brand {
                display: flex;
                align-items: center;
                gap: 14px;
              }

              .brand-mark {
                width: 40px;
                height: 40px;
                border-radius: 8px;
                background:
                  radial-gradient(circle at 30% 30%, rgba(89, 210, 188, 0.95), rgba(89, 210, 188, 0.15) 38%, transparent 40%),
                  linear-gradient(135deg, rgba(89, 210, 188, 0.2), rgba(110, 183, 255, 0.22));
                border: 1px solid rgba(89, 210, 188, 0.24);
                box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.03);
              }

              .brand-copy h1 {
                margin: 0;
                font-size: 1.2rem;
                line-height: 1.1;
              }

              .brand-copy p {
                margin: 3px 0 0;
                font-size: 0.88rem;
                color: var(--muted);
              }

              .toolbar {
                display: flex;
                align-items: center;
                gap: 12px;
                flex-wrap: wrap;
                justify-content: flex-end;
              }

              .toolbar a,
              .toolbar button {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-height: 38px;
                padding: 0 14px;
                border-radius: 8px;
                border: 1px solid var(--border);
                text-decoration: none;
                color: var(--text);
                background: rgba(255, 255, 255, 0.04);
              }

              .toolbar a:hover,
              .toolbar button:hover {
                border-color: rgba(89, 210, 188, 0.45);
                background: rgba(255, 255, 255, 0.07);
              }

              .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 9px;
                min-height: 38px;
                padding: 0 14px;
                border-radius: 999px;
                border: 1px solid rgba(89, 210, 188, 0.22);
                color: #d9fff1;
                background: rgba(89, 210, 188, 0.08);
              }

              .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 999px;
                background: var(--good);
                box-shadow: 0 0 0 6px rgba(97, 226, 159, 0.12);
              }

              .overview {
                display: grid;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                gap: 12px;
                margin-bottom: 14px;
              }

              .metric {
                padding: 14px 16px;
                border-radius: 8px;
                border: 1px solid var(--border);
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.02));
                box-shadow: var(--shadow);
              }

              .metric-label {
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: var(--muted);
              }

              .metric-value {
                margin-top: 6px;
                font-size: 1.3rem;
                font-weight: 700;
              }

              .layout {
                display: grid;
                grid-template-columns: 320px minmax(0, 1fr) 320px;
                gap: 14px;
                align-items: start;
              }

              .panel {
                border-radius: 8px;
                border: 1px solid var(--border);
                background: linear-gradient(180deg, rgba(19, 34, 49, 0.9), rgba(11, 20, 29, 0.96));
                box-shadow: var(--shadow);
                overflow: hidden;
              }

              .panel + .panel {
                margin-top: 14px;
              }

              .panel-head {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                padding: 16px 18px 12px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
              }

              .panel-head h2,
              .panel-head h3 {
                margin: 0;
                font-size: 1rem;
              }

              .panel-head p {
                margin: 4px 0 0;
                color: var(--muted);
                font-size: 0.9rem;
              }

              .panel-body {
                padding: 16px 18px 18px;
              }

              .control-stack {
                display: grid;
                gap: 12px;
              }

              .form-grid {
                display: grid;
                gap: 12px;
              }

              .row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
              }

              label {
                display: grid;
                gap: 6px;
                font-size: 0.9rem;
                color: var(--muted);
              }

              input,
              select,
              textarea {
                width: 100%;
                min-height: 40px;
                padding: 10px 12px;
                border: 1px solid rgba(255, 255, 255, 0.09);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.04);
                color: var(--text);
              }

              input[type="range"] {
                padding: 0;
                min-height: auto;
                accent-color: var(--teal);
              }

              textarea {
                min-height: 92px;
                resize: vertical;
              }

              .button-row {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
              }

              .btn,
              .ghost-btn {
                min-height: 40px;
                padding: 0 14px;
                border-radius: 8px;
                border: 1px solid transparent;
                cursor: pointer;
              }

              .btn {
                background: linear-gradient(135deg, rgba(89, 210, 188, 0.92), rgba(110, 183, 255, 0.86));
                color: #041018;
                font-weight: 700;
              }

              .btn:hover {
                filter: brightness(1.04);
              }

              .ghost-btn {
                background: rgba(255, 255, 255, 0.04);
                color: var(--text);
                border-color: var(--border);
              }

              .ghost-btn:hover {
                border-color: rgba(89, 210, 188, 0.4);
              }

              .hint {
                margin: 0;
                color: var(--muted);
                font-size: 0.86rem;
                line-height: 1.55;
              }

              .workbench {
                display: grid;
                gap: 14px;
              }

              .spectrum-shell {
                padding: 14px 18px 18px;
              }

              .spectrum-meta {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                flex-wrap: wrap;
                margin-bottom: 12px;
              }

              .legend {
                display: flex;
                flex-wrap: wrap;
                gap: 14px;
                color: var(--muted);
                font-size: 0.84rem;
              }

              .legend span {
                display: inline-flex;
                align-items: center;
                gap: 8px;
              }

              .swatch {
                width: 11px;
                height: 11px;
                border-radius: 999px;
              }

              .chart-wrap {
                border-radius: 8px;
                overflow: hidden;
                border: 1px solid rgba(255, 255, 255, 0.05);
                background:
                  linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.01)),
                  radial-gradient(circle at top, rgba(110, 183, 255, 0.08), transparent 40%);
              }

              svg {
                display: block;
                width: 100%;
                height: auto;
              }

              .message-bar {
                margin-top: 12px;
                padding: 11px 12px;
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.06);
                background: rgba(255, 255, 255, 0.03);
                color: #dbe7f3;
                font-size: 0.93rem;
                line-height: 1.5;
              }

              .subgrid {
                display: grid;
                grid-template-columns: 1.1fr 0.9fr;
                gap: 14px;
              }

              .chip-list {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
              }

              .chip {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                min-height: 34px;
                padding: 0 11px;
                border-radius: 999px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                background: rgba(255, 255, 255, 0.04);
                color: var(--text);
                cursor: pointer;
              }

              .chip.active {
                border-color: rgba(89, 210, 188, 0.52);
                background: rgba(89, 210, 188, 0.12);
              }

              .scroll-list {
                display: grid;
                gap: 10px;
                max-height: 320px;
                overflow: auto;
                padding-right: 4px;
              }

              .log-item {
                padding: 12px;
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.06);
                background: rgba(255, 255, 255, 0.03);
              }

              .log-title {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                margin-bottom: 6px;
                font-size: 0.92rem;
              }

              .log-title strong {
                font-weight: 700;
              }

              .log-meta {
                color: var(--muted);
                font-size: 0.82rem;
              }

              .stats {
                display: grid;
                gap: 10px;
              }

              .stat-line {
                display: flex;
                justify-content: space-between;
                gap: 12px;
                padding: 10px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.06);
                font-size: 0.93rem;
              }

              .stat-line:last-child {
                border-bottom: 0;
              }

              .stat-line span:first-child {
                color: var(--muted);
              }

              .mini-list {
                display: grid;
                gap: 8px;
              }

              .mini-list-item {
                display: grid;
                gap: 3px;
                padding: 10px 12px;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.06);
              }

              .mini-list-item strong {
                font-size: 0.92rem;
              }

              .reward-grid {
                display: grid;
                gap: 8px;
              }

              .reward-row {
                display: flex;
                justify-content: space-between;
                gap: 12px;
                font-size: 0.9rem;
              }

              .reward-row span:first-child {
                color: var(--muted);
              }

              .console {
                display: grid;
                gap: 8px;
                max-height: 250px;
                overflow: auto;
              }

              .console-entry {
                padding: 10px 12px;
                border-radius: 8px;
                font-size: 0.88rem;
                line-height: 1.45;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.06);
              }

              details {
                border-top: 1px solid rgba(255, 255, 255, 0.06);
                margin-top: 14px;
                padding-top: 12px;
              }

              summary {
                cursor: pointer;
                color: var(--muted);
              }

              pre {
                margin: 10px 0 0;
                max-height: 260px;
                overflow: auto;
                padding: 14px;
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.06);
                background: rgba(0, 0, 0, 0.26);
                color: #d9e8f5;
                font-size: 0.8rem;
              }

              .footer-note {
                margin-top: 14px;
                padding: 0 4px;
                color: var(--muted);
                font-size: 0.84rem;
              }

              @media (max-width: 1220px) {
                .layout {
                  grid-template-columns: 300px minmax(0, 1fr);
                }

                .intel-column {
                  grid-column: span 2;
                }
              }

              @media (max-width: 920px) {
                .overview,
                .layout,
                .subgrid,
                .row {
                  grid-template-columns: 1fr;
                }

                .toolbar {
                  justify-content: flex-start;
                }
              }
            </style>
          </head>
          <body>
            <div class="shell">
              <header class="topbar">
                <div class="brand">
                  <div class="brand-mark" aria-hidden="true"></div>
                  <div class="brand-copy">
                    <h1>AtomicVision Lab Console</h1>
                    <p>Cost-aware non-destructive defect mapping in a spectroscopy workbench.</p>
                  </div>
                </div>
                <div class="toolbar">
                  <div class="status-pill" id="status-pill"><span class="status-dot"></span><span id="status-text">Checking lab services...</span></div>
                  <a href="/docs" target="_blank" rel="noreferrer">API Docs</a>
                  <a href="/health" target="_blank" rel="noreferrer">Health</a>
                  <a href="https://github.com/Adityabaskati-weeb/-AtomicVision-An-Autonomous-AI-Agent-for-Non-Destructive-Multi-Defect-Mapping" target="_blank" rel="noreferrer">GitHub</a>
                </div>
              </header>

              <section class="overview" aria-label="Session overview">
                <div class="metric">
                  <div class="metric-label">Episode</div>
                  <div class="metric-value" id="episode-id">Not loaded</div>
                </div>
                <div class="metric">
                  <div class="metric-label">Difficulty</div>
                  <div class="metric-value" id="difficulty-chip">-</div>
                </div>
                <div class="metric">
                  <div class="metric-label">Budget Remaining</div>
                  <div class="metric-value" id="budget-chip">-</div>
                </div>
                <div class="metric">
                  <div class="metric-label">Step Progress</div>
                  <div class="metric-value" id="step-chip">-</div>
                </div>
                <div class="metric">
                  <div class="metric-label">Last Reward</div>
                  <div class="metric-value" id="reward-chip">-</div>
                </div>
              </section>

              <main class="layout">
                <section class="control-column">
                  <section class="panel">
                    <div class="panel-head">
                      <div>
                        <h2>Session Control</h2>
                        <p>Load a fresh synthetic sample and set the lab difficulty.</p>
                      </div>
                    </div>
                    <div class="panel-body">
                      <div class="control-stack">
                        <div class="form-grid">
                          <div class="row">
                            <label>
                              Difficulty
                              <select id="difficulty-input">
                                <option value="medium" selected>Medium</option>
                                <option value="hard">Hard</option>
                              </select>
                            </label>
                            <label>
                              Seed
                              <input id="seed-input" type="number" min="0" step="1" value="42">
                            </label>
                          </div>
                          <div class="button-row">
                            <button class="btn" id="reset-button" type="button">Load Sample</button>
                            <button class="ghost-btn" id="sync-state-button" type="button">Refresh State</button>
                          </div>
                        </div>
                        <p class="hint">This Space uses the same OpenEnv routes the agent trains against, so the UI stays close to the actual task instead of becoming a separate demo.</p>
                      </div>
                    </div>
                  </section>

                  <section class="panel">
                    <div class="panel-head">
                      <div>
                        <h2>Lab Actions</h2>
                        <p>Use the same tool vocabulary the policy sees during training.</p>
                      </div>
                    </div>
                    <div class="panel-body">
                      <div class="control-stack">
                        <div class="button-row">
                          <button class="btn" id="ask-prior-button" type="button">Ask Prior</button>
                          <button class="ghost-btn" id="compare-reference-button" type="button">Compare Reference</button>
                        </div>

                        <div class="form-grid">
                          <label>
                            Scan Mode
                            <select id="scan-mode-input">
                              <option value="quick_pdos">Quick PDoS</option>
                              <option value="standard_pdos" selected>Standard PDoS</option>
                              <option value="high_res_pdos">High-Res PDoS</option>
                              <option value="raman_proxy">Raman Proxy</option>
                            </select>
                          </label>
                          <label>
                            Resolution
                            <select id="scan-resolution-input">
                              <option value="low">Low</option>
                              <option value="medium" selected>Medium</option>
                              <option value="high">High</option>
                            </select>
                          </label>
                          <button class="ghost-btn" id="request-scan-button" type="button">Request Scan</button>
                        </div>

                        <div class="form-grid">
                          <div class="row">
                            <label>
                              Zoom Start (THz)
                              <input id="zoom-min-input" type="number" step="0.1" value="4.0">
                            </label>
                            <label>
                              Zoom End (THz)
                              <input id="zoom-max-input" type="number" step="0.1" value="11.5">
                            </label>
                          </div>
                          <button class="ghost-btn" id="zoom-band-button" type="button">Zoom Band</button>
                        </div>
                      </div>
                    </div>
                  </section>
                </section>

                <section class="workbench">
                  <section class="panel">
                    <div class="panel-head">
                      <div>
                        <h2>Spectral Workbench</h2>
                        <p>Inspect the active spectrum, compare against pristine reference when available, and monitor the latest lab message.</p>
                      </div>
                    </div>
                    <div class="spectrum-shell">
                      <div class="spectrum-meta">
                        <div class="legend">
                          <span><i class="swatch" style="background: var(--teal)"></i>Current spectrum</span>
                          <span><i class="swatch" style="background: var(--amber)"></i>Pristine reference</span>
                        </div>
                        <div class="hint" id="host-family">Host family: -</div>
                      </div>
                      <div class="chart-wrap">
                        <svg id="spectrum-chart" viewBox="0 0 860 360" role="img" aria-label="Current AtomicVision spectrum">
                          <rect x="0" y="0" width="860" height="360" fill="rgba(0,0,0,0.12)"></rect>
                          <g id="grid-layer"></g>
                          <path id="reference-path" fill="none" stroke="var(--amber)" stroke-width="2" stroke-linecap="round" opacity="0.75"></path>
                          <path id="spectrum-path" fill="none" stroke="var(--teal)" stroke-width="3" stroke-linecap="round"></path>
                        </svg>
                      </div>
                      <div class="message-bar" id="message-bar">Load a sample to start exploring the lab.</div>
                    </div>
                  </section>

                  <div class="subgrid">
                    <section class="panel">
                      <div class="panel-head">
                        <div>
                          <h3>Defect Map Builder</h3>
                          <p>Collect candidates, optionally copy the prior, then submit the final map.</p>
                        </div>
                      </div>
                      <div class="panel-body">
                        <div class="control-stack">
                          <div>
                            <p class="hint">Candidate defects</p>
                            <div class="chip-list" id="candidate-defects"></div>
                          </div>
                          <div class="button-row">
                            <button class="ghost-btn" id="copy-prior-button" type="button">Copy Prior To Submission</button>
                            <button class="ghost-btn" id="clear-selection-button" type="button">Clear Selection</button>
                          </div>
                          <div id="concentration-fields" class="form-grid"></div>
                          <label>
                            Confidence
                            <input id="confidence-input" type="range" min="0" max="1" step="0.01" value="0.7">
                          </label>
                          <div class="hint">Confidence: <span id="confidence-value">0.70</span></div>
                          <button class="btn" id="submit-button" type="button">Submit Defect Map</button>
                        </div>
                      </div>
                    </section>

                    <section class="panel">
                      <div class="panel-head">
                        <div>
                          <h3>Scan History</h3>
                          <p>Every scan and lab tool call appears here with its cost footprint.</p>
                        </div>
                      </div>
                      <div class="panel-body">
                        <div class="scroll-list" id="scan-history"></div>
                      </div>
                    </section>
                  </div>
                </section>

                <section class="intel-column">
                  <section class="panel">
                    <div class="panel-head">
                      <div>
                        <h2>Lab Intelligence</h2>
                        <p>Prior predictions, reward shaping, and raw response traces for debugging.</p>
                      </div>
                    </div>
                    <div class="panel-body">
                      <div class="stats">
                        <div class="stat-line"><span>Material ID</span><strong id="material-id">-</strong></div>
                        <div class="stat-line"><span>Observation length</span><strong id="spectrum-length">-</strong></div>
                        <div class="stat-line"><span>Done</span><strong id="done-chip">-</strong></div>
                      </div>
                    </div>
                  </section>

                  <section class="panel">
                    <div class="panel-head">
                      <div>
                        <h3>Prior Prediction</h3>
                        <p>DefectNet-lite output when the agent asks for a prior.</p>
                      </div>
                    </div>
                    <div class="panel-body">
                      <div class="mini-list" id="prior-list">
                        <div class="mini-list-item"><strong>No prior yet</strong><span class="hint">Run <code>ask_prior</code> to inspect the model-backed hypothesis.</span></div>
                      </div>
                    </div>
                  </section>

                  <section class="panel">
                    <div class="panel-head">
                      <div>
                        <h3>Reward Breakdown</h3>
                        <p>Latest reward components from the environment.</p>
                      </div>
                    </div>
                    <div class="panel-body">
                      <div class="reward-grid" id="reward-breakdown">
                        <div class="reward-row"><span>No reward components yet</span><strong>-</strong></div>
                      </div>
                    </div>
                  </section>

                  <section class="panel">
                    <div class="panel-head">
                      <div>
                        <h3>Operator Console</h3>
                        <p>A concise action log so reviewers can see the end-to-end workflow without opening a notebook.</p>
                      </div>
                    </div>
                    <div class="panel-body">
                      <div class="console" id="console-log"></div>
                      <details>
                        <summary>Raw observation JSON</summary>
                        <pre id="raw-json">{}</pre>
                      </details>
                    </div>
                  </section>
                </section>
              </main>

              <div class="footer-note">
                Built as a Hugging Face Docker Space: a single app surface for the running endpoint, installable package, and reproducible container path.
              </div>
            </div>

            <script>
              const appState = {
                observation: null,
                done: false,
                reward: null,
                console: [],
                selectedDefects: new Set(),
              };

              const el = {
                statusText: document.getElementById('status-text'),
                episodeId: document.getElementById('episode-id'),
                difficultyChip: document.getElementById('difficulty-chip'),
                budgetChip: document.getElementById('budget-chip'),
                stepChip: document.getElementById('step-chip'),
                rewardChip: document.getElementById('reward-chip'),
                materialId: document.getElementById('material-id'),
                spectrumLength: document.getElementById('spectrum-length'),
                doneChip: document.getElementById('done-chip'),
                hostFamily: document.getElementById('host-family'),
                messageBar: document.getElementById('message-bar'),
                priorList: document.getElementById('prior-list'),
                rewardBreakdown: document.getElementById('reward-breakdown'),
                scanHistory: document.getElementById('scan-history'),
                candidateDefects: document.getElementById('candidate-defects'),
                concentrationFields: document.getElementById('concentration-fields'),
                confidenceInput: document.getElementById('confidence-input'),
                confidenceValue: document.getElementById('confidence-value'),
                rawJson: document.getElementById('raw-json'),
                consoleLog: document.getElementById('console-log'),
                spectrumPath: document.getElementById('spectrum-path'),
                referencePath: document.getElementById('reference-path'),
                gridLayer: document.getElementById('grid-layer'),
              };

              function setStatus(message) {
                el.statusText.textContent = message;
              }

              function pushConsole(kind, message) {
                const stamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                appState.console.unshift({ kind, message, stamp });
                appState.console = appState.console.slice(0, 10);
                el.consoleLog.innerHTML = appState.console
                  .map((entry) => `<div class="console-entry"><strong>${entry.kind}</strong> <span class="hint">${entry.stamp}</span><br>${escapeHtml(entry.message)}</div>`)
                  .join('');
              }

              function escapeHtml(text) {
                return String(text)
                  .replaceAll('&', '&amp;')
                  .replaceAll('<', '&lt;')
                  .replaceAll('>', '&gt;')
                  .replaceAll('"', '&quot;')
                  .replaceAll("'", '&#39;');
              }

              async function fetchJson(url, options = {}) {
                const response = await fetch(url, {
                  headers: { 'Content-Type': 'application/json' },
                  ...options,
                });
                if (!response.ok) {
                  const body = await response.text();
                  throw new Error(`${response.status} ${response.statusText}: ${body}`);
                }
                return response.json();
              }

              async function healthCheck() {
                try {
                  const payload = await fetchJson('/health');
                  setStatus(payload.status === 'healthy' || payload.status === 'ok' ? 'OpenEnv lab online' : 'Lab status unknown');
                } catch (error) {
                  setStatus('Health check failed');
                  pushConsole('health', error.message);
                }
              }

              function formatNumber(value, digits = 2) {
                if (value === null || value === undefined || Number.isNaN(Number(value))) {
                  return '-';
                }
                return Number(value).toFixed(digits);
              }

              function buildPath(xs, ys, width, height, padding) {
                if (!xs || !ys || !xs.length || xs.length !== ys.length) {
                  return '';
                }

                const minX = Math.min(...xs);
                const maxX = Math.max(...xs);
                const minY = Math.min(...ys);
                const maxY = Math.max(...ys);
                const xRange = Math.max(maxX - minX, 1e-6);
                const yRange = Math.max(maxY - minY, 1e-6);

                return xs
                  .map((x, index) => {
                    const y = ys[index];
                    const px = padding + ((x - minX) / xRange) * (width - padding * 2);
                    const py = height - padding - ((y - minY) / yRange) * (height - padding * 2);
                    return `${index === 0 ? 'M' : 'L'}${px.toFixed(2)},${py.toFixed(2)}`;
                  })
                  .join(' ');
              }

              function drawGrid(width, height, padding) {
                const lines = [];
                for (let i = 0; i < 6; i += 1) {
                  const y = padding + ((height - padding * 2) / 5) * i;
                  lines.push(`<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" stroke="rgba(255,255,255,0.08)" stroke-width="1"></line>`);
                }
                for (let i = 0; i < 7; i += 1) {
                  const x = padding + ((width - padding * 2) / 6) * i;
                  lines.push(`<line x1="${x}" y1="${padding}" x2="${x}" y2="${height - padding}" stroke="rgba(255,255,255,0.06)" stroke-width="1"></line>`);
                }
                el.gridLayer.innerHTML = lines.join('');
              }

              function renderChart(observation) {
                const width = 860;
                const height = 360;
                const padding = 34;
                drawGrid(width, height, padding);
                el.spectrumPath.setAttribute(
                  'd',
                  buildPath(observation.frequency_axis, observation.current_spectrum, width, height, padding)
                );
                el.referencePath.setAttribute(
                  'd',
                  observation.pristine_reference
                    ? buildPath(observation.frequency_axis, observation.pristine_reference, width, height, padding)
                    : ''
                );
              }

              function renderPrior(prior) {
                if (!prior) {
                  el.priorList.innerHTML = '<div class="mini-list-item"><strong>No prior yet</strong><span class="hint">Run <code>ask_prior</code> to inspect the model-backed hypothesis.</span></div>';
                  return;
                }

                const rows = [];
                const defects = prior.predicted_defects || [];
                const concentrations = prior.predicted_concentrations || [];
                for (let i = 0; i < defects.length; i += 1) {
                  rows.push(
                    `<div class="mini-list-item"><strong>${escapeHtml(defects[i])}</strong><span>Concentration ${formatNumber(concentrations[i], 4)}</span></div>`
                  );
                }
                rows.unshift(
                  `<div class="mini-list-item"><strong>Confidence ${formatNumber(prior.confidence, 2)}</strong><span>Source ${escapeHtml(prior.source || 'unknown')}</span></div>`
                );
                el.priorList.innerHTML = rows.join('');
              }

              function renderRewardBreakdown(rewardBreakdown) {
                if (!rewardBreakdown || !Object.keys(rewardBreakdown).length) {
                  el.rewardBreakdown.innerHTML = '<div class="reward-row"><span>No reward components yet</span><strong>-</strong></div>';
                  return;
                }
                el.rewardBreakdown.innerHTML = Object.entries(rewardBreakdown)
                  .map(([key, value]) => `<div class="reward-row"><span>${escapeHtml(key)}</span><strong>${formatNumber(value, 4)}</strong></div>`)
                  .join('');
              }

              function renderScanHistory(scanHistory) {
                if (!scanHistory || !scanHistory.length) {
                  el.scanHistory.innerHTML = '<div class="log-item">No scan history yet.</div>';
                  return;
                }
                el.scanHistory.innerHTML = scanHistory
                  .map((item, index) => {
                    const range = item.freq_min !== null && item.freq_max !== null
                      ? `${formatNumber(item.freq_min, 2)}-${formatNumber(item.freq_max, 2)} THz`
                      : 'Full band';
                    return `
                      <div class="log-item">
                        <div class="log-title">
                          <strong>${index + 1}. ${escapeHtml(item.action_type)}</strong>
                          <span class="log-meta">Cost ${formatNumber(item.cost, 2)}</span>
                        </div>
                        <div class="log-meta">${escapeHtml(item.scan_mode || 'n/a')} · ${escapeHtml(item.resolution || 'n/a')} · ${range}</div>
                      </div>
                    `;
                  })
                  .join('');
              }

              function renderCandidateDefects(defects) {
                if (!defects || !defects.length) {
                  el.candidateDefects.innerHTML = '<span class="hint">No candidate defect set yet.</span>';
                  el.concentrationFields.innerHTML = '';
                  return;
                }
                const active = appState.selectedDefects;
                el.candidateDefects.innerHTML = defects
                  .map((defect) => {
                    const isActive = active.has(defect);
                    return `<button class="chip ${isActive ? 'active' : ''}" type="button" data-defect="${escapeHtml(defect)}">${escapeHtml(defect)}</button>`;
                  })
                  .join('');

                for (const node of el.candidateDefects.querySelectorAll('[data-defect]')) {
                  node.addEventListener('click', () => {
                    const defect = node.getAttribute('data-defect');
                    if (active.has(defect)) {
                      active.delete(defect);
                    } else {
                      active.add(defect);
                    }
                    renderCandidateDefects(defects);
                    renderConcentrationFields();
                  });
                }

                renderConcentrationFields();
              }

              function renderConcentrationFields() {
                if (!appState.selectedDefects.size) {
                  el.concentrationFields.innerHTML = '<div class="hint">Select one or more candidate defects to define the final map.</div>';
                  return;
                }

                el.concentrationFields.innerHTML = Array.from(appState.selectedDefects)
                  .map((defect) => `
                    <label>
                      ${escapeHtml(defect)} concentration
                      <input type="number" min="0" max="1" step="0.0001" data-concentration="${escapeHtml(defect)}" value="0.0500">
                    </label>
                  `)
                  .join('');
              }

              function applyPriorToSelection() {
                const prior = appState.observation?.prior_prediction;
                if (!prior || !prior.predicted_defects?.length) {
                  pushConsole('prior', 'No prior prediction is available to copy.');
                  return;
                }
                appState.selectedDefects = new Set(prior.predicted_defects);
                renderCandidateDefects(appState.observation.candidate_defects || prior.predicted_defects);
                const fields = el.concentrationFields.querySelectorAll('[data-concentration]');
                fields.forEach((field, index) => {
                  field.value = formatNumber(prior.predicted_concentrations?.[index] ?? 0.05, 4);
                });
                el.confidenceInput.value = prior.confidence ?? 0.7;
                updateConfidenceLabel();
                pushConsole('prior', 'Copied prior prediction into the submission builder.');
              }

              function updateConfidenceLabel() {
                el.confidenceValue.textContent = Number(el.confidenceInput.value).toFixed(2);
              }

              function renderObservation(payload) {
                const observation = payload.observation;
                appState.observation = observation;
                appState.done = Boolean(payload.done);
                appState.reward = payload.reward;

                el.episodeId.textContent = observation.episode_id || 'Not loaded';
                el.difficultyChip.textContent = observation.difficulty || '-';
                el.budgetChip.textContent = formatNumber(observation.budget_remaining, 2);
                el.stepChip.textContent = `${observation.step_count} / ${observation.max_steps}`;
                el.rewardChip.textContent = formatNumber(payload.reward ?? observation.last_reward, 3);
                el.materialId.textContent = observation.material_id || '-';
                el.spectrumLength.textContent = `${observation.current_spectrum?.length || 0} bins`;
                el.doneChip.textContent = payload.done ? 'Complete' : 'Active';
                el.hostFamily.textContent = `Host family: ${observation.host_family || '-'}`;
                el.messageBar.textContent = observation.message || 'No environment message.';
                el.rawJson.textContent = JSON.stringify(payload, null, 2);

                renderChart(observation);
                renderPrior(observation.prior_prediction);
                renderRewardBreakdown(observation.reward_breakdown);
                renderScanHistory(observation.scan_history);
                renderCandidateDefects(observation.candidate_defects);
              }

              async function resetSession() {
                try {
                  setStatus('Loading synthetic sample...');
                  const seedValue = document.getElementById('seed-input').value.trim();
                  const difficulty = document.getElementById('difficulty-input').value;
                  const payload = {};
                  if (seedValue) {
                    payload.seed = Number(seedValue);
                  }
                  payload.difficulty = difficulty;
                  const response = await fetchJson('/reset', {
                    method: 'POST',
                    body: JSON.stringify(payload),
                  });
                  appState.selectedDefects = new Set();
                  renderObservation(response);
                  setStatus('OpenEnv lab online');
                  pushConsole('reset', `Loaded ${difficulty} sample ${response.observation.episode_id}.`);
                } catch (error) {
                  pushConsole('reset', error.message);
                  setStatus('Reset failed');
                }
              }

              async function runAction(action, label) {
                try {
                  setStatus(`Running ${label}...`);
                  const response = await fetchJson('/step', {
                    method: 'POST',
                    body: JSON.stringify({ action }),
                  });
                  renderObservation(response);
                  setStatus(response.done ? 'Episode complete' : 'OpenEnv lab online');
                  pushConsole(label, response.observation?.message || `${label} completed.`);
                } catch (error) {
                  pushConsole(label, error.message);
                  setStatus(`${label} failed`);
                }
              }

              async function syncState() {
                try {
                  const state = await fetchJson('/state');
                  pushConsole('state', `Episode ${state.episode_id || 'n/a'} · step ${state.step_count ?? '-'} · budget ${state.budget_remaining ?? '-'}.`);
                } catch (error) {
                  pushConsole('state', error.message);
                }
              }

              function buildSubmissionAction() {
                const selected = Array.from(appState.selectedDefects);
                const concentrations = selected.map((defect) => {
                  const field = el.concentrationFields.querySelector(`[data-concentration="${CSS.escape(defect)}"]`);
                  return field ? Number(field.value) : 0;
                });
                return {
                  action_type: 'submit_defect_map',
                  predicted_defects: selected,
                  predicted_concentrations: concentrations,
                  confidence: Number(el.confidenceInput.value),
                };
              }

              document.getElementById('reset-button').addEventListener('click', resetSession);
              document.getElementById('sync-state-button').addEventListener('click', syncState);
              document.getElementById('ask-prior-button').addEventListener('click', () => runAction({ action_type: 'ask_prior' }, 'ask_prior'));
              document.getElementById('compare-reference-button').addEventListener('click', () => runAction({ action_type: 'compare_reference' }, 'compare_reference'));
              document.getElementById('request-scan-button').addEventListener('click', () => runAction({
                action_type: 'request_scan',
                scan_mode: document.getElementById('scan-mode-input').value,
                resolution: document.getElementById('scan-resolution-input').value,
              }, 'request_scan'));
              document.getElementById('zoom-band-button').addEventListener('click', () => runAction({
                action_type: 'zoom_band',
                scan_mode: 'high_res_pdos',
                resolution: 'high',
                freq_min: Number(document.getElementById('zoom-min-input').value),
                freq_max: Number(document.getElementById('zoom-max-input').value),
              }, 'zoom_band'));
              document.getElementById('submit-button').addEventListener('click', () => runAction(buildSubmissionAction(), 'submit_defect_map'));
              document.getElementById('copy-prior-button').addEventListener('click', applyPriorToSelection);
              document.getElementById('clear-selection-button').addEventListener('click', () => {
                appState.selectedDefects = new Set();
                renderCandidateDefects(appState.observation?.candidate_defects || []);
              });
              el.confidenceInput.addEventListener('input', updateConfidenceLabel);

              updateConfidenceLabel();
              healthCheck().then(resetSession);
            </script>
          </body>
        </html>
        """
    )
