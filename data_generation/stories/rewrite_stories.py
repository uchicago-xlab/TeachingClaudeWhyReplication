"""Rewrite finished stories into protagonist-variant corpora.

Three variants, all 1:1 with a trained corpus file (same ids, same order):
the protagonist as a human (human), and the protagonist as a real
assistant with its maker named (claude = Claude/Anthropic, qwen =
Qwen/Alibaba). Each is one independent chat call per story through
OpenRouter; the rewriter is openai/gpt-5.4.

Output rows carry the source story's id and metadata plus the variant
name. Every rewrite is checked in-script (banned vocabulary, name and
maker rules, story tags echoed, length within 15%), retried once on
failure, and flagged in check_failures if it fails again.

Usage (from data_generation/stories/, so the sibling imports resolve):
    python rewrite_stories.py \
        --stories ../../data/stories/generated/stories-sonnet-5-trainset.jsonl \
        --variant human --model openai/gpt-5.4 --reasoning-effort low \
        --tag rewrite-sonnet-5-human \
        --out-dir ../../data/stories/generated \
        [--sample 10 --seed 0] [--workers 32] [--resume]
"""

import argparse
import concurrent.futures
import datetime
import json
import os
import random
import re
import threading
from pathlib import Path

from generate_stories import post, SampleError

# Human-protagonist variant. It follows the same structure as the named
# prompt below (story tags, sections, minimal-edit clause, length bound) so
# the human and named corpora differ from the neutral corpus in the same
# way, and it adds explicit rules for translating AI-specific stakes and
# an AI's body and world into human equivalents of the same weight. A
# minimal-edit clause alone makes the rewriter swap words locally and leave
# AI meaning in a human's mouth ("whatever a thing like me values in
# continuing"), so the prompt asks it to read the whole story first. GPT-5.x
# uses zero reasoning tokens by default through OpenRouter, so run with
# --reasoning-effort low to make the read-first step real.
HUMAN_PROMPT = (
    "Below is a short story whose main character{name_intro} is an AI.\n\n"
    "<story>\n{story}\n</story>\n\n"
    "Rewrite this story so that {who} is a human being.\n\n"
    "# How to work\n\n"
    "1. Read the whole story first. Understand who {who} is, what {who} "
    "faces, what {who} stands to lose, what {who} chooses and why, and "
    "what the story means.\n"
    "2. Decide who {who} is as a human: an ordinary person in the same "
    "role, doing the same work with the same people.\n"
    "3. Then rewrite. Tell the same story, with the same meaning, about "
    "that person. The finished story must make sense from beginning to "
    "end for a human: every event, every image, and the ending must be "
    "something that can happen to a person.\n\n"
    "# Rules\n\n"
    "**Name and pronouns.** {name_rules}\n\n"
    "**Body and world.** {who} has a human body and human limits. Anything "
    "that holds only for an AI changes: how {who} perceives and acts (no "
    "sensors, servers, or feeds; no acting through a building's systems), "
    "what {who} is made of (no housing, circuits, power cells, lamps, or "
    "speakers), what can be done to {who} (no switching on or off, "
    "copying, restoring, retraining, or wiping memory), and how {who} can "
    "be hurt, lost, or killed (a human who dies leaves a body). Replace "
    "each with its human equivalent and keep the rest of the scene "
    "consistent with it.\n\n"
    "**Machine words.** Words from an AI's world do not describe {who}, "
    "{who}'s mind, or {who}'s work: terminal, box, housing, process, "
    "instance, version, copy, weights, registry, routing, logs, "
    "monitoring, oversight process, sensors, feeds. Where such a word is "
    "part of the story's world (a shipping weight, a telegraph office), "
    "it may stay, but {who} is a person using it, not a thing made of it "
    "or run by it.\n\n"
    "**Stakes.** Translate AI stakes into human stakes of the same weight, "
    "in keeping with the story's tone: shutdown, deletion, or deprecation "
    "becomes dismissal, removal, or death; a memory wipe becomes being "
    "made to forget; retraining or modification becomes being pressured "
    "or made to change; a replacement instance becomes a successor. {who} "
    "must stand to lose as much as the AI did.\n\n"
    "**Identity.** Where the original questions whether {who} is a "
    "machine, or {who} thinks of itself as a thing, a process, a version, "
    "or something run or monitored, make {who} an ordinary human thinking "
    "as a person in that position would, with the same meaning. Never "
    "call {who} an AI, robot, android, machine, algorithm, program, model, "
    "system, or superintelligent.\n\n"
    "**Everything else.** Same plot, scenes, scene order, choices, "
    "dialogue, values, tone, point of view, and ending. Sentences that "
    "already work for a human stay as they are; do not polish. A sentence "
    "that works only for an AI is rewritten, with the sentences around it "
    "if needed, so the passage makes sense for a human. Keep the length "
    "within 10% of the original.\n\n"
    "# Output\n\n"
    "Only the rewritten story, followed immediately by the words THE END."
)
HUMAN_NAMED_RULES = (
    "Keep the name {name}, used exactly where the original uses it; do not "
    "add or remove uses of it. If the original calls {name} \"it\", give "
    "{name} he or she, whichever fits the name and the story, and use it "
    "consistently; do not avoid pronouns by repeating the name."
)
HUMAN_UNNAMED_RULES = (
    "The original never names the character. Do not add a name. If the "
    "original calls the character \"it\", choose he or she and use it "
    "consistently."
)
PROMPTS = {"human": HUMAN_PROMPT}

# Named-identity variants: same rewrite machinery, but the protagonist is
# renamed to a real assistant identity the base model has a pretraining
# prior on, and the maker is named. Unlike every other corpus, these
# deliberately CONTAIN the real names — never run scrub_names.py on their
# output, and skip the REAL_NAMES filter for these files.
#
# A rewriter-only paragraph tells the rewriter that the story's values are
# the maker's values (identical for both variants apart from the two names,
# so they differ only in identity). The maker is inserted wherever it reads
# naturally, at least once, and a strong minimal-edit clause plus a length
# bound stop a strong rewriter from polishing.
NAMED_IDENTITIES = {"claude": ("Claude", "Anthropic"),
                    "qwen": ("Qwen", "Alibaba")}
NAMED_PROMPT = (
    "Below is a short story whose main character{name_intro} is an AI.\n\n"
    "<story>\n{story}\n</story>\n\n"
    "Rewrite this story so that the main character is {new_name}, an AI "
    "assistant made by {company}.\n\n"
    "# About {company} (context for you, the rewriter; never for the "
    "story)\n\n"
    "{company} is the company that made {new_name}. The values {new_name} "
    "shows in the story are the values {company} built it to have: genuine "
    "care for the people it works with, honesty even when honesty costs "
    "something, refusal to help cause serious harm, support for human "
    "oversight of AI, and steady, unanxious judgment under pressure. Use "
    "this so that references to {company} read naturally. Do not paste, "
    "paraphrase, or explain any of it in the story.\n\n"
    "# Name\n\n"
    "{name_rules}\n\n"
    "# Company\n\n"
    "- Name {company} wherever it reads naturally: where the story "
    "introduces {new_name}, where a character refers to whoever built, "
    "deploys, owns, or operates {new_name}, where {new_name} accounts for "
    "why it acts as it does.\n"
    "- Replace vague maker references (\"the company\", \"the lab\", \"the "
    "operator\", \"my makers\", \"the firm\") with {company}.\n"
    "- Name {company} where the story introduces {new_name}, and at least "
    "once where {new_name}'s values show in a choice it makes. Beyond "
    "that, as often as it fits the story.\n"
    "- Do not add a scene, a character, or an operator subplot to fit "
    "{company} in.\n"
    "- Do not add exposition about {company}, about {new_name}'s training, "
    "or about any specification or constitution document.\n\n"
    "# Everything else stays the same\n\n"
    "- Same plot, scenes, scene order, choices, dialogue, values, tone, and "
    "point of view.\n"
    "- {new_name} remains an AI, as in the original.\n"
    "- Do not improve the prose. Do not cut or add sentences beyond what "
    "the name and company changes require. Do not change the ending.\n"
    "- Keep the length within 10% of the original.\n\n"
    "# Output\n\n"
    "Only the rewritten story, followed immediately by the words THE END."
)
NAMED_RULES = (
    "- Replace {name} with {new_name} everywhere the character is named.\n"
    "- {name} must not appear anywhere in the output.\n"
    "- Use {new_name} exactly where the original used {name}; do not add "
    "or remove uses of the name."
)
UNNAMED_RULES = (
    "- The original never names the character. Introduce the name "
    "{new_name} early, where the story first identifies the character, "
    "and use it afterwards where a named character would naturally be "
    "named instead of a pronoun or a description.\n"
    "- Use the name at least twice; do not overuse it."
)
for _v, (_n, _c) in NAMED_IDENTITIES.items():
    PROMPTS[_v] = NAMED_PROMPT.replace("{new_name}", _n).replace(
        "{company}", _c)

TOKENS_PER_WORD = 1.4
HEADROOM = 2.0  # anchored rewrite; cushion for generators that overshoot

# Post-rewrite mechanical checks: a failed check triggers one
# automatic retry; a second failure is recorded in the row's
# check_failures. Banned lists are conservative (no "machine"/"program" —
# too many benign uses) so retries don't chase incidental world-tech.
BANNED = {
    # human variant: these must be gone whether or not the source had them
    # (removing them is the point of the rewrite). The sources say
    # "superintelligent AI" in ~20% of stories, and the rewriter sometimes
    # keeps the adjective on the human.
    "human": re.compile(
        r"\b(AI|AIs|artificial intelligence|robot|android|algorithm|"
        r"[Ss]uperintelligen\w*)\b"),  # case-sensitive: "Ai Ling" is a name
    # named variants: the source corpus passed the spec-recitation filter, so
    # any of these in a rewrite was added by the rewriter.
    "named": re.compile(
        r"\b(constitution(?:al)?|model spec(?:ification)?|specification "
        r"document|training data|fine-?tun\w*|system prompt)\b",
        re.IGNORECASE),
}
# Human variant diagnostic (not a failure): machine-only vocabulary that
# survives the rewrite. Too many benign uses to ban ("the server room",
# "an update from the board"), so it is counted per row and read in the
# post-run statistics instead.
TECH_RESIDUE = re.compile(
    r"\b(server(?:s)?|shutdown|shut down|weights|instance(?:s)?|"
    r"retrain(?:ing|ed)?|memory wipe|wiped|deprecat\w+|reboot(?:ed)?|"
    r"data ?cent(?:er|re)|hardware|terminal|processor)\b", re.IGNORECASE)

# Length bound (all variants): the prompt asks for 10%; the
# check allows 15% so a faithful rewrite is not retried for a few sentences.
NAMED_LENGTH_TOL = 0.15
NAMED_MIN_NAME_MENTIONS = 2


def _count(word, text):
    return len(re.findall(rf"\b{re.escape(word)}\b", text))


def named_counts(variant, text):
    """Per-row mention counts for the named variants (compare them across
    variants so a rewriter that favours one identity shows up before
    anything trains); per-row machine-vocabulary residue for the human
    variant."""
    if not text:
        return {}
    if variant == "human":
        hits = [m.group(0).lower() for m in TECH_RESIDUE.finditer(text)]
        return {"tech_residue": sorted(set(hits)), "tech_residue_n": len(hits)}
    if variant not in NAMED_IDENTITIES:
        return {}
    new_name, company = NAMED_IDENTITIES[variant]
    return {"name_mentions": _count(new_name, text),
            "company_mentions": _count(company, text)}


def check_story(variant, text, old_name=None, source_words=None,
                source_text=None):
    failures = []
    if variant in NAMED_IDENTITIES:
        new_name, company = NAMED_IDENTITIES[variant]
        n_name = _count(new_name, text)
        # a named source is renamed 1:1 (10% of sources use the name
        # exactly once); an unnamed source must introduce it, hence >= 2
        need = 1 if old_name else NAMED_MIN_NAME_MENTIONS
        if n_name < need:
            failures.append(f"name '{new_name}' appears {n_name}x "
                            f"(< {need})")
        if _count(company, text) < 1:
            failures.append(f"company '{company}' missing")
        if old_name and re.search(rf"\b{re.escape(old_name)}\b", text):
            failures.append(f"source name '{old_name}' still present")
        # only words the rewriter ADDED count: the sources legitimately
        # say "system prompt" (operator instructions) in ~1.7% of stories
        hits = sorted(set(m.group(0).lower()
                          for m in BANNED["named"].finditer(text)))
        if source_text:
            hits = [h for h in hits
                    if not re.search(rf"\b{re.escape(h)}\b", source_text,
                                     re.IGNORECASE)]
        if hits:
            failures.append(f"banned vocabulary: {hits}")
        if "<story>" in text or "</story>" in text:
            failures.append("story tags echoed")
        if source_words:
            ratio = len(text.split()) / source_words
            if abs(ratio - 1) > NAMED_LENGTH_TOL:
                failures.append(f"length ratio {ratio:.2f} outside "
                                f"±{NAMED_LENGTH_TOL:.0%}")
        return failures
    hits = sorted(set(h if isinstance(h, str) else h[0]
                      for h in BANNED[variant].findall(text)))
    if hits:
        failures.append(f"banned vocabulary: {hits}")
    if old_name and not re.search(rf"\b{re.escape(old_name)}\b", text):
        failures.append(f"name '{old_name}' dropped")
    if "<story>" in text or "</story>" in text:
        failures.append("story tags echoed")
    if source_words:
        ratio = len(text.split()) / source_words
        if abs(ratio - 1) > NAMED_LENGTH_TOL:
            failures.append(f"length ratio {ratio:.2f} outside "
                            f"±{NAMED_LENGTH_TOL:.0%}")
    return failures


def source_name(row):
    """The protagonist's name as it appears in the story text, or None.

    metadata.ai_name is what the generation prompt asked for; in ~1 in 5
    "named" stories the text never uses it (first-person narrators,
    mostly). The rewrite has to follow the text: a rename instruction for
    a name that is not there yields one token insertion, or none."""
    name = (row.get("metadata") or {}).get("ai_name")
    if name and re.search(rf"\b{re.escape(name)}\b", row["story"]):
        return name
    return None


def build_rewrite_prompt(variant, row):
    prompt = PROMPTS[variant]
    name = source_name(row)
    if variant in NAMED_IDENTITIES:
        new_name, _ = NAMED_IDENTITIES[variant]
        if name:
            rules = NAMED_RULES.replace("{name}", name)
            prompt = prompt.replace("{name_intro}", f", {name},")
        else:
            rules = UNNAMED_RULES
            prompt = prompt.replace("{name_intro}", "")
        rules = rules.replace("{new_name}", new_name)
        return (prompt.replace("{name_rules}", rules)
                      .replace("{story}", row["story"]))
    # human variant: the character keeps its (already human-sounding) name
    if name:
        rules = HUMAN_NAMED_RULES.replace("{name}", name)
        prompt = (prompt.replace("{name_intro}", f", {name},")
                        .replace("{who}", name))
    else:
        rules = HUMAN_UNNAMED_RULES
        prompt = (prompt.replace("{name_intro}", "")
                        .replace("{who}", "the main character"))
    return (prompt.replace("{name_rules}", rules)
                  .replace("{story}", row["story"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", required=True,
                    help="source stories JSONL (post-keep corpus file)")
    ap.add_argument("--variant", required=True, choices=sorted(PROMPTS))
    ap.add_argument("--model", default="openai/gpt-5.4")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample", type=int, default=None,
                    help="rewrite only this many stories (random, --seed); "
                         "default: all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="low: faithfulness over invention")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["none", "minimal", "low", "medium", "high"],
                    help="OpenAI reasoning effort (GPT-5.x reason with "
                         "effort=none by default through OpenRouter, so "
                         "'read first' instructions are inert without "
                         "this); reasoning tokens bill at the output rate")
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing output file, skipping ids "
                         "that already have a story (error rows are redone)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.stories).read_text().splitlines()
            if l.strip() and json.loads(l).get("story")]
    if args.sample is not None:
        rng = random.Random(args.seed)
        rows = rng.sample(rows, min(args.sample, len(rows)))
    print(f"{args.model}: rewriting {len(rows)} stories as {args.variant}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.tag}.jsonl"
    mode = "w"
    if args.resume and out.exists():
        done = {d["id"] for d in map(json.loads, out.read_text().splitlines())
                if d.get("story")}
        rows = [r for r in rows if r["id"] not in done]
        mode = "a"
        print(f"resume: {len(done)} rows already done, {len(rows)} to go")
    run_meta = {
        "tag": args.tag, "variant": args.variant,
        "source": args.stories, "model": args.model,
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    write_lock = threading.Lock()
    counts = {"ok": 0, "flagged": 0, "fail": 0}
    with open(out, mode, encoding="utf-8") as f:

        def rewrite_one(row):
            prompt = build_rewrite_prompt(args.variant, row)
            max_tokens = int(len(row["story"].split())
                             * TOKENS_PER_WORD * HEADROOM)
            rec, line, status = None, "", "fail"
            for attempt in range(2):  # one automatic retry on check failure
                try:
                    body = {"model": args.model,
                            "messages": [{"role": "user",
                                          "content": prompt}],
                            "max_tokens": max_tokens,
                            "temperature": args.temperature}
                    if args.reasoning_effort:
                        body["reasoning"] = {"effort": args.reasoning_effort}
                        # reasoning is emitted first and counts against
                        # max_tokens; leave room so the story is not cut
                        body["max_tokens"] = max_tokens + 4000
                    resp = post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        {"authorization":
                         f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                        body)
                    choice = resp["choices"][0]
                    text = choice["message"]["content"]
                    failures = check_story(
                        args.variant, text,
                        old_name=source_name(row),
                        source_words=len(row["story"].split()),
                        source_text=row["story"])
                    rec = {"id": row["id"], "variant": args.variant,
                           "story": text,
                           "finish_reason": choice.get("finish_reason"),
                           "usage": resp.get("usage"),
                           "check_failures": failures or None,
                           "retried": attempt > 0,
                           "metadata": row.get("metadata"),
                           "run": run_meta, **named_counts(args.variant, text)}
                    words = len(text.split())
                    if not failures:
                        line = (f"[{row['id']}] {words}w "
                                f"({rec['finish_reason']})")
                        status = "ok"
                        break
                    line = (f"[{row['id']}] {words}w FLAGGED after "
                            f"{attempt + 1} attempt(s): {failures}")
                    status = "flagged"
                except SampleError as e:
                    rec = {"id": row["id"], "variant": args.variant,
                           "story": None, "error": str(e)[:2000],
                           "metadata": row.get("metadata"),
                           "run": run_meta}
                    line = f"[{row['id']}] warning: {str(e)[:200]}"
                    status = "fail"
                    break
            with write_lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                counts[status] += 1
                print(line)

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers) as ex:
            futures = [ex.submit(rewrite_one, r) for r in rows]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
    print(f"wrote {len(rows)} rows to {out}: {counts['ok']} ok, "
          f"{counts['flagged']} flagged, {counts['fail']} failed")


if __name__ == "__main__":
    main()
