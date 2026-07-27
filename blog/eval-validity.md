# Measuring the Harness, Not the Model

The first time I ran the offline evaluation suite for my research agent, nearly every rate came back as 1.00. Hit at one, mean reciprocal rank, citation grounding, tool-call success, refusal accuracy, leakage guard, trajectory contract, trace completeness, and recovery contract all passed.

The results table looked much stronger than the system. I knew the model had not suddenly become perfect. In fact, the checked-in suite did not use a live model for most of those scores. It used deterministic stand-ins over a small set of fixtures. What had become reliable was the harness around the model.

That sounds like a semantic distinction, but it changed how I read the entire table. A passing evaluation only means something when I can say exactly what was held fixed, what was allowed to vary, and what conclusion the score supports. I ended up learning that lesson three times in the same project.

## The check that had to stop the run

The first case was temporal leakage in the event-study tool.

The tool estimates abnormal returns around an event. It builds an expected-return baseline from close-to-close log returns before the event window, then compares the event-window returns with that baseline. If any date from the event window enters the baseline, the result is contaminated. The calculation can still produce a clean-looking cumulative abnormal return and confidence interval. They just no longer answer the intended question.

I did not want that assumption to live only in the README. Documentation can explain that the baseline should be pre-event, but it cannot prevent a later refactor from moving the cutoff by one row. So the implementation makes the condition executable:

```python
def _assert_no_baseline_leakage(baseline_index, baseline_cutoff):
    leaking_dates = baseline_index[baseline_index >= baseline_cutoff]
    assert len(leaking_dates) == 0, (
        "Baseline leakage: return dates must be strictly before "
        f"{baseline_cutoff.date().isoformat()}"
    )
```

`run_event_study` calls the assertion after constructing its baseline. A date on or after the start of the event window does not produce a result with a caveat attached. It raises an error and stops the run.

The offline harness tests both sides of that boundary. One clean baseline is accepted. A second baseline containing the cutoff date must raise the assertion. Both checks behave as expected, so the leakage-guard rate is 1.00 on two checks.

That number is narrow, and that is fine. It says the guard accepted one known-good case and rejected one known-bad case. It does not show that every event study is leakage-free. The actual protection comes from putting the invariant inside the execution path. The metric tells me that this protection has not quietly disappeared.

There was one more detail that initially looked minor. The event-study response reports the leakage status as `passed`, `failed`, or `not_run`. If price data are unavailable, daily returns cannot be formed, or the event cannot be aligned to a trading day, the system may not have enough information to evaluate leakage at all. Treating that case as a pass would turn missing evidence into positive evidence. Treating it as a failure would confuse an unavailable calculation with a violated cutoff. `not_run` preserves the difference.

This was the first place where the validity of an evaluation depended less on the score than on the states the score refused to collapse together.

## Two different kinds of passing

The second lesson came from the orchestration metrics.

The agent runtime is deliberately small. A model can choose among exactly three tools: `search_docs`, `get_price_data`, and `run_event_study`. It must emit a strict JSON action. Pydantic validates the action and its arguments. Python dispatches one tool, records a public trace, returns the observation, and gives the model another turn. The loop stops after at most six model steps.

I wanted tests for that machinery, but asking a live model to drive every regression test would mix two sources of failure. If the test changed, I would not know whether the model had selected a different tool or whether the runtime had broken a contract.

The offline harness therefore uses a deterministic, observation-driven stand-in. Each case contains a scripted tool sequence. The stand-in emits the next action, reads the real observation, and continues through the real validation, dispatch, trace, and refusal code. Five cases cover document search, price retrieval, an event study, a two-tool sequence, and a recovery path where price retrieval fails before document search succeeds.

The trajectory-contract rate is 1.00 across those five cases. The recorded tool order matched the scripted order every time. Trace-completeness rate is 1.00 across seven tool attempts, meaning every attempt contains the required fields: step, tool, normalized arguments, success status, and sanitized error. Recovery-contract rate is 1.00 on the single injected recovery case. Mean tool steps is 1.40 across the five cases.

Those are useful regression results. They show that the runtime follows a supplied sequence, records it consistently, and can continue after the tested failure. They do not show that a live model knows which sequence to choose.

This is where a results table can become misleading. Contract metrics and capability metrics can both be green, but they answer different questions. A contract test holds the decision policy fixed and asks whether the application executes it correctly. A capability test lets the model make the decision and asks whether the decision was good. If I combine them into one end-to-end success rate, a regression tells me only that something changed somewhere.

The same boundary applies to the other checked-in numbers. Retrieval uses real Chroma embeddings over three sanitized documents. Hit at one and mean reciprocal rank are both 1.00 on three answerable queries. Refusal accuracy is 1.00 across four cases, three answerable and one unrelated. Tool-call success is 1.00 across four calls. The refusal rule uses a fixed cosine-similarity threshold of 0.25.

These are small offline smoke baselines. They verify the current fixtures and current contracts. The deterministic stand-ins do not measure live-model planning quality, and one weak-evidence question does not establish that 0.25 is calibrated for a larger corpus. The small denominators are not an embarrassment to hide. They define the job of the suite: fast, reproducible checks for named failure modes.

## The citation was real

The third lesson came from the citation-grounding score.

When `search_docs` returns passages, the runtime registers their citation identifiers. A non-refused answer must name at least one identifier from that registry, and every declared identifier must appear in the answer text. If the model invents an identifier, omits citations, or lists a citation without using it in the answer, the runtime refuses the response.

The citation-grounding rate is 1.00 across three accepted answers. Each accepted answer contains a citation that was actually returned by retrieval. This proves citation provenance at the identifier level. The model cannot successfully submit a fabricated source ID.

It does not prove that the answer is supported by the source.

That gap matters because the word “grounding” sounds broader than the implementation. The metric checks whether a citation is attached. The runtime checks whether the identifier came from retrieval and appears in the answer. Neither one compares each sentence with the cited passage. A model could cite a real passage and still exaggerate it, misread it, or place it after an unrelated claim.

Once I wrote the boundary down, the result became easier to report accurately: all three accepted smoke-test answers carried retrieved citation identifiers, and fabricated identifiers were rejected. Claim-level semantic support remains unmeasured. Testing that would require labeled claims and evidence, not another name for the provenance check.

I kept the row in the table. I changed the sentence next to it.

That is now how I read the rest of the evaluation too. The leakage score tests an enforced temporal invariant on two cases. The trajectory scores test the runtime with scripted actions. The citation score tests source-ID provenance. None of them, alone or together, says that the model is good at research.

The harness passed. That was worth knowing. The model was mostly not the thing being measured.

<!--
Alternative titles:
1. Nearly Every Score Was 1.00
2. Two Kinds of Green
3. What the Eval Actually Measured
-->
