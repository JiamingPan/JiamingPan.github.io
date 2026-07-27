# What Does Your Agent Eval Prove?

Your agent eval is probably measuring the wrong thing. The metric may be computed correctly. The problem is often the conclusion attached to it.

I ran into this while building an offline evaluation harness for a small research agent. The runtime has a bounded ReAct loop and exactly three tools: document search, price retrieval, and an event study. The harness checks retrieval, refusal, citation handling, tool trajectories, traces, recovery, and temporal leakage. Its checked-in results are all clean. That does not make the agent good.

It means a set of specific mechanisms behaved as expected on a small set of specific cases. Evaluation becomes useful only when the reported claim stays inside that boundary. Three parts of the system made this distinction concrete for me.

## Make leakage fail the run

The event-study tool estimates abnormal returns around an event. Its expected-return baseline uses close-to-close log returns strictly before the event window starts. If a baseline includes returns from the window it is meant to evaluate, the resulting cumulative abnormal return is contaminated.

It is easy to acknowledge this risk in documentation. That is not enough. A caveat in a README depends on every future caller, refactor, and analysis remembering the caveat. The implementation instead turns the temporal boundary into an assertion:

```python
def _assert_no_baseline_leakage(baseline_index, baseline_cutoff):
    leaking_dates = baseline_index[baseline_index >= baseline_cutoff]
    assert len(leaking_dates) == 0, (
        "Baseline leakage: return dates must be strictly before "
        f"{baseline_cutoff.date().isoformat()}"
    )
```

`run_event_study` calls this assertion after constructing the baseline. The unavailable-result path calls it too whenever a cutoff can be established. A baseline date on or after the start of the event window does not produce a result with a warning attached. It stops the computation.

That difference matters for evaluation validity. A footnote describes an assumption. An assertion enforces it. If the assertion fails, there is no valid event-study result to score. The evaluation harness tests both sides of the boundary: one clean baseline is accepted, and one baseline containing the cutoff date is rejected. The checked-in leakage-guard rate is 1.00 on two checks. That is a smoke test of the guard, not evidence that every possible event-study input is leakage-free.

The response also reports a leakage status of `passed`, `failed`, or `not_run`. The third state is necessary. If price data are unavailable, no daily returns can be formed, or an event cannot be aligned to a trading day, the system may not have enough information to evaluate the leakage condition. Calling that a pass would convert missing evidence into positive evidence. Calling it a failure would confuse an unavailable computation with a violated temporal boundary.

`not_run` preserves the difference. It says the check produced no verdict. That state is less convenient than a Boolean, but it prevents a dashboard or downstream evaluator from silently counting an unperformed check as successful. The validity of the reported leakage rate depends on keeping its denominator restricted to checks that actually ran.

## Separate contract metrics from capability metrics

This is the central distinction in the harness. Some metrics test whether the software honors a contract. Other metrics would need to test whether a model can solve a task. Those are different experiments.

The production loop permits only `search_docs`, `get_price_data`, and `run_event_study`. Model actions must be strict JSON. Pydantic rejects unknown fields and invalid arguments. The runtime dispatches one tool at a time, records a bounded public trace, and stops after at most six model steps. These are application-level contracts. I can test them without asking a live model to make intelligent decisions.

The orchestration evaluation therefore uses a deterministic, observation-driven stand-in. Each case specifies an expected tool sequence. The stand-in emits the next scripted action, observes whether the tool succeeded, and continues through the real runtime. Five cases cover search, price retrieval, an event study, a two-tool sequence, and a recovery sequence in which price retrieval fails before document search succeeds.

The resulting trajectory-contract rate is 1.00 across five cases. This means the recorded tool order exactly matched the scripted order in all five. It does not mean a live model knows which order to choose.

Trace-completeness rate is 1.00 across seven tool attempts. Every recorded attempt contains the required public fields: step, tool, normalized arguments, success status, and a sanitized error field. This shows that the runtime preserves its trace schema on those attempts. It says nothing about whether the attempted action was useful.

Recovery-contract rate is 1.00 on one injected recovery case. In that case, the trace contains the expected failed tool call followed by the expected successful action. One case is enough to catch a broken recovery path in a smoke suite. It is not enough to estimate recovery capability across realistic failures. The mean number of attempted tool actions is 1.40 across the five orchestration cases, which describes the fixture set more than it describes an agent population.

These metrics use the real loop, validation, dispatch, and trace code. They deliberately do not use live-model planning. That is their strength. If a trajectory contract regresses while the scripted actions remain fixed, the problem is in the application path. If a separate live-model routing evaluation regresses while the contracts still pass, the problem is more likely in model behavior, prompting, or the distribution of tasks. Blending both layers into one end-to-end success rate would make the regression harder to locate.

The rest of the offline suite has similarly narrow scope. Retrieval uses real Chroma embeddings over three sanitized documents. Hit at one is 1.00 on three answerable queries, and mean reciprocal rank is 1.00 on the same three. Refusal accuracy is 1.00 across four cases, three answerable and one deliberately unrelated. Tool-call success rate is 1.00 across four calls. Citation-grounding rate is 1.00 across the three accepted answers.

Those values are small offline smoke baselines. The deterministic stand-ins exercise system contracts, not live-model quality. The retrieval corpus contains three documents. The refusal threshold is a fixed cosine-similarity score of 0.25, and the suite checks one weak-evidence question. A perfect rate here verifies that the current fixture crosses the current threshold in the expected direction. It does not establish that 0.25 is calibrated for a larger corpus or a new domain.

The small sample sizes are part of the report. They tell me what kind of conclusion the numbers can support. A five-case trajectory suite can provide fast regression coverage for five named paths. It cannot estimate how often an unconstrained model will select the right tool in deployment. Reporting the denominator keeps those two claims from being confused.

## State what the number does not prove

The citation guard provides the clearest example of a metric boundary.

When `search_docs` returns passages, the runtime registers their citation identifiers. A non-refused final answer must include at least one identifier from that registry, and each declared identifier must appear in the answer text. If the model returns an identifier that was never retrieved, omits citations, or lists a citation without placing it in the answer, the runtime refuses the response.

This proves citation provenance at the identifier level. The model cannot successfully return a fabricated source ID. The 1.00 citation-grounding rate on three accepted answers confirms that each accepted answer in the smoke suite carries at least one retrieved citation.

It does not prove that any sentence is semantically supported by the cited passage. The metric implementation checks whether a claim has a citation attached. The runtime checks whether the identifier came from retrieval and appears in the answer. Neither mechanism compares the meaning of each sentence with the meaning of its cited evidence. A response could cite a real retrieved passage and still overstate it, misread it, or attach it to the wrong claim.

Calling the metric “grounding” without stating that boundary would invite a larger conclusion than the implementation supports. The precise result is narrower: accepted answers carried retrieved citation identifiers, and fabricated identifiers were rejected. Claim-level entailment would require a different evaluation with labeled claims and evidence.

The same discipline applies to refusal. A rate of 1.00 across four cases means the system accepted three answerable fixtures and refused one weak-evidence fixture under the current retrieval setup and threshold. It does not license a claim that the agent reliably knows when it lacks evidence. That would require a broader set of answerable and unanswerable questions, threshold analysis, and live-model behavior.

I now treat the boundary of a metric as part of the metric. The value, denominator, fixture construction, deterministic or live execution mode, and explicit non-claim belong together. Without those pieces, a number is easy to repeat and hard to interpret.

The checked-in suite is intentionally modest. It establishes that temporal leakage is enforced, runtime contracts are exercised reproducibly, and citation provenance is checked on small offline fixtures. It also records what remains unmeasured. That is enough for a regression baseline because the claims match the evidence.

<!--
Alternative titles:
1. Your Agent Eval Is Measuring the Wrong Thing
2. What an Agent Metric Actually Means
3. A Passing Eval Is Not a Capability Claim
-->
