# Security checkpoint queue tool

MSL304 project by Sohana Kumar (2024CS10251) and Sidharth Valsan (2024MT10433).

The tool tells a security checkpoint how many screening lanes to open in each time block
and how to arrange the queue. Passengers are the customers and each lane is a server. You
enter the arrival rate in each block, the service rate of a lane and the costs; the tool
returns the lanes for each block, the queue design, and the daily cost compared with keeping
the same number of lanes open all day.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py                                     # opens http://localhost:8501
python -m pytest                                         # 98 tests, about 30 s
python scripts/make_results.py                           # rebuilds report_assets/
```

`pulp` is kept below version 3 because PuLP 2.x includes the CBC solver. The code also works
with PuLP 4 if `highspy` is installed.

## Files

| Path | Contents |
|---|---|
| `app.py` | Streamlit app with six tabs: Problem, Pooled vs Separate Queues, Priority Lane Designs, Staffing Plan, Simulation Check, Conclusions. The first four tabs show an animated picture of the queue |
| `src/animation.py` | Short simulation that records who is waiting where, for the animations |
| `src/picture.py` | Draws the animated checkpoint: lanes with an officer dot on top, people queueing below, priority passengers as stars, the fast lane with a thick border |
| `src/queueing.py` | One shared queue (M/M/c), a queue per lane (c x M/M/1, or M/G/1 when cs2 is not 1), benefit of pooling |
| `src/priority.py` | Pollaczek-Khinchine formula, dedicated fast lane (P1), non-preemptive priority in each lane (P2, Cobham) |
| `src/staffing.py` | Cost of a block, enumeration, dynamic program, PuLP integer program, baseline, comparison of designs |
| `src/breakeven.py` | Break-even arrival rates, found with a grid scan and Brent's method |
| `src/simulation.py` | SimPy simulation with replications and 95% confidence intervals, stress tests (group arrivals, two-state arrival rate, lane closure, join the shortest queue), and an FCFS recursion engine used as a check |
| `src/sensitivity.py` | One-way, two-way and tornado analysis |
| `src/sample_data.py` | Assumed arrival profile with a morning and an evening peak |
| `src/validation.py` | Input checks and warnings |
| `tests/` | Every test in section 10 of the specification, standard M/M/1 and M/M/c reference values, and further checks |
| `scripts/make_results.py` | Every number and figure in the report and slides, with fixed seeds |
| `report_assets/` | Figures, app screenshots and `results.json` |
| `docs/report/report.tex` | LaTeX source of the report; build it with `latexmk -pdf report.tex` |
| `MSL304 Project Report.pdf` | The report |
| `MSL304 Project PPT.pptx` | The presentation |
| `MSL304 Presentation Video.mp4` | The recorded presentation |

## Modelling notes

* All rates are per minute. A block of length Delta receives lambda_t x Delta passengers, so
  its waiting cost is `C_w * lambda_t * Delta * W_q`. The proposal left out Delta.
* `a = lambda/mu` is the offered load and `rho = lambda/(c*mu)` is the utilisation of one lane.
* In each block the tool computes the total cost for c = 1, 2, 3, ... and picks the smallest.
  Waiting cost is charged on W_q, the time spent in the queue.
* When cs2 is not 1, a lane with its own queue is an exact M/G/1 queue, and the shared queue
  uses the Allen-Cunneen approximation `W_q(M/M/c) * (1 + cs2) / 2`. Both are exact at the
  default cs2 = 1.
* The priority designs are optimised against the mean-wait target for both classes. Their
  tail probabilities come from the simulation in `src/simulation.py`.
* A block that cannot meet the target with c_max lanes runs with c_max lanes and is flagged.
* A simulation check passes when the formula value is inside the 95% confidence interval or
  within 5% of the simulated mean. A 95% interval misses about once in twenty runs, so an
  occasional fail is expected.

The sample arrival data and all costs are made up for illustration; none were measured at a real checkpoint.
