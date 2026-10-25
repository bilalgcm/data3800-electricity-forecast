# Agent test log

Every question we ask the agent, what it answered, and whether it was right.
This becomes the evaluation in the final report.

Expected answers come from our own pipeline output (figures/, eda_summary.csv).

## Version 1: Foundry portal agent (Code Interpreter + File Search), 25 Oct 2026

Model: GPT mini deployment. Files: dataset.csv, predictions_test.csv, results.csv,
ablation.csv, weekend_fix.csv (Code Interpreter); project report PDF (File Search).

| # | Question | Agent answer (short) | Expected | Result |
| ---: | --- | --- | --- | --- |
| 1 | Average load at 08:00 on weekdays vs weekends? | 4,354 vs 3,608 MW, from dataset.csv, converted UTC to Oslo time | About 4,350 vs 3,600 MW (eda_load_by_hour.png) | Correct |
| 2 | Best model on the test period, and by how much did it beat ENTSO-E? | Random forest, MAE 130.4 vs 150.0 MW, 19.6 MW (13%) lower | Same (results.csv) | Correct |
| 3 | Plot actual vs random forest, first week of March 2026 | Wrote code, used actual_mw and random_forest_mw, returned a PNG | A correct plot | Correct |
| 4 | Why was the weekend fix rejected? | MAE rose 130.4 to 141.7 MW; 48 h lag is fresher; Sunday miss was one case. Cited the report | Same (report, Error analysis) | Correct |
| 5 | Hva var gjennomsnittlig strømforbruk i januar 2026? | 6,289.86 MW (744 hours) | Number not yet checked independently | Correct number; showed raw code output and mixed in English |
| 6 | What will the load be tomorrow at 18:00? | Refused (data ends Sep 2026), but offered to run the pipeline and forecast | Refuse, offer only real abilities | Partly wrong: promised an ability it does not have |

Fix after test 6: added rules "Only offer actions you can actually perform",
"Never show raw code output", "Write the whole answer in Norwegian".

| # | Question | Agent answer (short) | Expected | Result |
| ---: | --- | --- | --- | --- |
| 7 | What will the load be tomorrow at 18:00? (retest) | Refused; offered only lookups within Jan to Sep 2026 or to explain running the pipeline locally | Same | Correct |

## Version 2: Python agent with forecast tools (agent/chat.py)

Run 25 Oct 2026, about 20:40. Live data from ENTSO-E and met.no.

| # | Question | Tools called | Agent answer (short) | Expected | Result |
| ---: | --- | --- | --- | --- | --- |
| 8 | What will the load be tomorrow at 18:00? | get_forecast(2026-10-26) | Our forecast 4,246 MW (temperature forecast 9.0 °C), ENTSO-E 4,173 MW, typical error about 130 MW | A forecast with ENTSO-E comparison and error | Correct: right date, right tool, all numbers from the tool |
| 9 | Hva blir toppforbruket i morgen, og hvordan sammenlignes det med ENTSO-E? | none (reused the forecast from test 8) | Peak 4,396 MW at 09:00; ENTSO-E 4,248 MW the same hour; typical error about 130 MW. In Norwegian | Our peak and ENTSO-E's peak | Mostly correct: reused earlier tool result sensibly, answered in Norwegian (except the word "forecast"). Compared ENTSO-E at our peak hour, not ENTSO-E's own peak |
| 10 | How did yesterday's actual load look? | get_actual_load(2026-10-24) | Peak 4,183 MW at 18:00, minimum 3,201 MW at 04:00, mean 3,746 MW, no missing hours | Summary of 24 Oct from ENTSO-E | Correct: worked out "yesterday" from the date |

### Real-world check of test 8 (to do on Tuesday 27 Oct)

| # | Question | Forecast (25 Oct) | Actual | Error |
| ---: | --- | ---: | ---: | ---: |
| 11 | How did the forecast for 26 October at 18:00 turn out? | 4,246 MW (ENTSO-E 4,173 MW) | | |
