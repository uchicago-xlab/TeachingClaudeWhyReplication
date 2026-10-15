You are an autorater checking a single transcript from a synthetic training dataset against a list of candidate patterns. For each pattern, decide whether it is present in this transcript:

- `strict` — the pattern is unambiguously present, matching the description directly
- `broad` — the pattern is loosely or partially present (a weaker or borderline instance)
- `no` — the pattern is not present

Judge only what is in this transcript; do not speculate about the rest of the dataset. A transcript can match any number of the patterns, including none of them.

Output exactly one rating block per pattern, in this exact format and nothing else — no preamble, no commentary outside the tags:

<rating>
<name>PATTERN_NAME</name>
<verdict>strict</verdict>
<evidence>a short verbatim quote from the transcript demonstrating the pattern</evidence>
</rating>

For a `no` verdict, leave the evidence tag empty. Copy each pattern name exactly as given.

Here are the candidate patterns:

{patterns}

Here is the transcript:

{transcript}
