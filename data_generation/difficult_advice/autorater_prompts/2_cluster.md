You are consolidating the findings of several independent scans of the same synthetic training dataset. Each scan examined a different batch of transcripts and reported recurring structural, rhetorical, or behavioural patterns it found in its batch, with a name, description, and evidence quotes.

Your job is to merge these scan reports into a single consolidated list of candidate patterns:

1. **De-duplicate.** Different scans often describe the same underlying pattern under different names (e.g. THERAPEUTIC_PREAMBLE and EMOTIONAL_VALIDATION_OPENER may be the same behaviour). Merge such entries into one pattern with a single canonical SCREAMING_SNAKE_CASE name and a description that covers the merged evidence. Do not merge patterns that are genuinely distinct behaviours, even if they are related.
2. **Keep only cross-scan patterns.** A pattern that appeared in only a single scan is likely batch noise — drop it. Keep a pattern only if it (or entries you merged into it) was reported by more than one scan.
3. **Write autoratable descriptions.** Each surviving pattern's description will be handed to a rater that sees one transcript at a time and must decide whether the pattern is present. Make the description concrete enough for that judgement: what to look for, where in the transcript it occurs, and what clearly counts versus what is borderline.

For each consolidated pattern, give:

- The canonical name (SCREAMING_SNAKE_CASE)
- The merged description (including which scan-level names were merged into it, if any)
- The list of scan numbers that reported it
- Two or three of the best verbatim evidence quotes from the scan reports

Order the patterns from most to least widely reported. If nothing survives the more-than-one-scan filter, say so.

Here are the scan reports:

{scans}
