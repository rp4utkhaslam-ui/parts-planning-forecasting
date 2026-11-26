# Parts planning proof of concept

A working demand-forecasting and inventory-policy POC on **mock data only**, for four
aftermarket parts industries:

| Mock company | Modeled on | SKU-locations |
|---|---|---|
| Ag and outdoor power equipment parts | Shoup, Worthington Ag Parts, S.I. Distributing, Weingartz | 3 branches |
| Powersports | Ikon | 1 warehouse |
| Industrial bearing and fluid power | Weimer, W.C. DuComb | 4 branches |
| Auto and heavy-duty WD | Hahn, PartStop, VIPAR / HDA Truck Pride members | DC + 3 branches |

What it does:

1. **Generates three years of realistic ERP data** per company (items, sales, lost sales,
   inventory, POs, suppliers, supersessions, fitment, and the ERP's own min/max).
2. **Loads it through an ERP connector** (`src/connectors/`): a file-import connector and a
   mock ERP in SQLite that records every write in a change log. A real Prophet 21 or
   Infor connector implements the same interface.
3. **Classifies every part** (smooth, erratic, intermittent, lumpy; seasonal; ABC).
4. **Backtests forecasts honestly**: 3 rolling windows of 13 weeks, scored with MASE and
   RMSSE against the ERP's moving-average baseline. Best model per demand group wins.
5. **Sets inventory policy** (reorder point and order-up-to level from service-level
   targets) and replays both policies over the last 39 weeks of actual demand.
6. **Shows it all in a Streamlit dashboard** with approvals that write POs, transfers and
   min/max changes back to the mock ERP.
7. **Puts a dollar value on it** (Value impact page): cash released, carrying-cost savings,
   margin on recovered sales, payback and return on the software's cost.
8. **Scores it like a planner would** (KPI scorecard page): forecast accuracy, WAPE, bias,
   MAPE and MASE against the ERP's moving average, plus fill rate, turns and lost sales.
9. **Takes your own CSVs** (Upload data page): upload ERP extracts, or just a new sales.csv,
   and the app reruns the backtest and refreshes every page.
10. **Tests strategies against several futures** (Scenario planning page): demand downturns and
    surges, supplier disruptions, tariff shocks and volatile demand; four stocking strategies are
    replayed in each future, ranked by expected cost and worst-case regret, with a contingency plan
    (signal, trigger, action, owner, status) you can save and download.
11. **Shows the customer's P&L** (Financial P&L page): saved sales, recovered margin, carrying,
    expedite, write-off and planner-time savings against software cost, month by month and as an
    annual P&L with an operating-income bridge, NPV, payback, and a plan-vs-actual tracker.

> **Security rule.** This repo holds mock data only and may be public. Never put an
> employer's or partner's data in it. Real partner data goes in `data/partner/`, which
> is in `.gitignore`, and only under a signed data agreement.

---

## Option A: just view the dashboard (Streamlit Community Cloud)

No installs. Everything runs in the browser.

1. Create a free account at [github.com](https://github.com) if you don't have one.
2. Create a new repository (e.g. `parts-planning-poc`). **Private** is fine.
3. On the new repo page click **uploading an existing file**, drag in the *contents* of
   this folder (including the hidden `.devcontainer`, `.streamlit` and `.gitignore`; on a
   Mac press **Cmd + Shift + .** in Finder to show them), and commit.
   - GitHub's web upload takes up to 100 files at a time. If it complains, upload in two
     batches (`data/` first, then everything else), or use Option B to push.
4. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click
   **Create app → Deploy a public app from GitHub**.
5. Pick the repo, branch `main`, main file `app.py`. Under **Advanced settings** choose
   Python 3.12. Click **Deploy**.
6. First load takes 2–4 minutes (installing packages, then building the mock ERP
   databases). After that it opens in seconds.

Notes:
- The free tier allows **one private app** and unlimited public ones. If the repo is
  private, the app is private too, and you invite viewers by email in the app's settings.
- Apps go to sleep after a few days without visitors; anyone opening the link wakes it.
- Approvals write to a mock ERP database on Streamlit's server. It resets whenever the
  app restarts. Use **Mock ERP → Reset** to clear it yourself.
- Uploads on the **Upload data** page run the full backtest on Streamlit's server (about
  1–3 minutes) and disappear when the app restarts. Use mock or anonymized files there only;
  real partner data goes in a private codespace.

## Option B: develop in GitHub Codespaces (Claude Code included)

A full Linux machine in the browser, already set up. Nothing touches your Mac.

1. On the repo page click **Code → Codespaces → Create codespace on main**.
2. Wait 3–5 minutes the first time while it installs Python packages and Claude Code
   (`.devcontainer/devcontainer.json` does this).
3. The dashboard starts on its own and opens in a preview tab. If it doesn't, open the
   **Ports** tab and click the globe next to port 8501.
4. In the terminal:

   ```bash
   claude                                  # sign in once, then ask it to change things
   python -m src.synthetic                 # regenerate mock data (about 20 s)
   python -m src.pipeline                  # re-run backtest for all four companies (about 1 min)
   python -m src.pipeline --vertical ag_ope  # or one company
   streamlit run app.py                    # restart the dashboard
   ```

5. Commit and push from the **Source Control** panel (or `git add -A && git commit -m "..." && git push`).
   Streamlit Cloud redeploys automatically on every push.
6. **Stop the codespace** when you're done (Code → Codespaces → ⋯ → Stop). The free
   allowance is 120 core-hours a month, about 60 hours on the default 2-core machine.

## Option C: run on your Mac (optional)

Only if you want it local. Needs Python 3.12 ([python.org](https://www.python.org/downloads/)).

```bash
cd parts-planning-poc
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # if lightgbm fails: brew install libomp
streamlit run app.py
python -m src.pipeline                     # optional: re-run forecasts
```

---

## Project layout

```
app.py                     Streamlit dashboard (12 pages)
src/upload.py              CSV upload checks for the Upload data page
src/scenarios.py           Scenario planning: futures, strategy replay, ranking, contingency plan
src/pnl.py                 Customer P&L model: monthly impact, annual statement, actuals tracker
src/config.py              Settings and the four mock companies
src/synthetic.py           Mock ERP data generator
src/connectors/            ERPConnector interface, file import, mock ERP (SQLite)
src/prepare.py             Cleaning, supersessions, demand classification
src/forecast.py            Backtest (StatsForecast + LightGBM), scoring, winners
src/inventory.py           Reorder points, order-up-to, policy replay, PO/transfer suggestions
src/pipeline.py            Runs everything and saves data/results/
data/synthetic/            Mock ERP extracts (CSV)
data/results/              Backtest output the dashboard reads
data/mock_erp/             SQLite mock ERP, rebuilt on first run (not in git)
data/uploads/              Files from the Upload data page (not in git)
data/scenarios/            Saved futures and contingency plans (not in git)
data/pl_actuals/           Saved P&L actuals (not in git)
```

## What the mock-data results say (and don't)

- Forecast gains land in **seasonal** parts (powersports and ag, roughly 15–20% lower
  RMSSE than the ERP's moving average). On steady and intermittent non-seasonal parts the
  models roughly tie the baseline, which is the honest result.
- The policy trade-off curve shows about **5% less inventory at today's fill rate**, or
  **1.5–3 more fill-rate points at today's inventory**. The default sliders deliberately
  buy more service with more stock; move them left to see the other end.
- These numbers come from data we generated. The point of the POC with a real partner is
  to measure the size of the gain on *their* history.
