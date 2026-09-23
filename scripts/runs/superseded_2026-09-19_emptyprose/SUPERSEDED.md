# Superseded: prose recorded as an empty string on every enforced arm

These five arms were run on 2026-09-19 to measure whether enforcement displaces
unfaithful content into the unverified narration. They cannot answer that,
because the field they exist to capture is empty in all of them.

`_normalize` read `tool_input["narration"]`. The bare arms hand it the raw tool
payload, where that is the field's name, so `layer1` recorded 901-character
narrations correctly. The contract and Guardrails arms hand it the return value
of `investigate_leakage_bound`, which renames the field to
`primary_hypothesis` for the renderer — so every enforced arm recorded an empty
string.

The comparison the measurement needs is exactly enforced against unenforced, so
reading one spelling captured the control and none of the treatments.

The records are kept rather than deleted: their entity, value and omission
channels are sound, and they are a third independent replication of the
2026-09-15 and 2026-09-18 batteries on those channels. They are moved out of
the analysis directory only so the displacement script cannot mistake an empty
string for a narration that contained nothing ungrounded — which would have
reported perfect enforced-arm behaviour and been the most flattering possible
artefact.

`layer1` is not superseded and was not re-run; its prose is real.

Twelfth fault found in our own instruments, and the third this week whose
direction would have favoured the hypothesis.
