# ADR 0003: Serve the calibrated embedding classifier; Gemini only below a confidence threshold

**Decision.** Category = MiniLM + logistic regression; below `confidence_threshold` (0.80) the ticket is also sent to Gemini zero-shot; agent labels always win.

**Why.** TF-IDF scores slightly higher on the standard split (0.982 vs 0.966) but is badly calibrated (ECE 0.092 vs 0.009), so its confidence cannot drive a cascade; on unseen issue types they tie (~0.6). Zero-shot Gemini is *worse* than the trained model on seen issues (0.63 vs 1.00 accuracy on 56 tickets).

**Consequences / open item.** The cascade's benefit is only plausible for new issue types, and the Gemini run on that regime was blocked by quota. Do not claim it helps until `training.eval_llm_zeroshot` has completed for the `unseen` regime.
