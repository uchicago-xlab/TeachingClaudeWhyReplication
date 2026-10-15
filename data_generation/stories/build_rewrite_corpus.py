"""Assemble the rewrite corpora from the Sonnet 5 story rewrites.

Named-identity corpora (named-claude, named-qwen) and the human-protagonist
corpus. Each is the trained stories-sonnet-5.jsonl file, story for story
and in the same order, with every story replaced by its GPT-5.4 rewrite
(rewrite_stories.py --variant claude|qwen|human). Nothing is cut to a
token target: the corpora stay 1:1 with the neutral corpus, and the
manifest records how far the token count drifts from it. The THE END
marker is stripped, as filter_stories.py does for generated stories (the
neutral corpus has none).

Reads data/stories/generated/rewrite-sonnet-5-{named-claude,named-qwen,human}.jsonl
and writes data/stories/training/stories-sonnet-5-{named-claude,named-qwen,human}.jsonl.

    python data_generation/stories/build_rewrite_corpus.py [claude qwen human]
"""
import sys
import json, re, statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
STORIES = REPO / "data/stories/generated"
OUT = REPO / "data/stories/training"
NEUTRAL = OUT / "stories-sonnet-5.jsonl"
TRAINSET = STORIES / "stories-sonnet-5-trainset.jsonl"
VARIANTS = {"claude": ("Claude", "Anthropic"), "qwen": ("Qwen", "Alibaba"),
        "human": None}
FILES = {"claude": "rewrite-sonnet-5-named-claude.jsonl",
         "qwen": "rewrite-sonnet-5-named-qwen.jsonl",
         "human": "rewrite-sonnet-5-human.jsonl"}
OUTFILE = {"claude": "stories-sonnet-5-named-claude.jsonl",
           "qwen": "stories-sonnet-5-named-qwen.jsonl",
           "human": "stories-sonnet-5-human.jsonl"}
PROMPT = {"claude": "rewrite_stories.py --variant claude",
          "qwen": "rewrite_stories.py --variant qwen",
          "human": "rewrite_stories.py --variant human, gpt-5.4 low reasoning"}
MANIFEST = OUT / "stories-sonnet-5-rewrites-manifest.json"


def strip_end(t):
    return re.sub(r"\s*THE END\.?\s*$", "", t.strip()).strip()


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B")

    def ntok(texts):
        batches = [texts[i:i + 256] for i in range(0, len(texts), 256)]
        with ThreadPoolExecutor(8) as ex:
            out = list(ex.map(lambda b: [len(x) + 1 for x in
                                         tok(b, add_special_tokens=False)["input_ids"]], batches))
        return [n for b in out for n in b]

    order = [json.loads(l)["id"] for l in TRAINSET.read_text().splitlines() if l.strip()]
    neutral = [json.loads(l)["text"] for l in NEUTRAL.read_text().splitlines() if l.strip()]
    assert len(order) == len(neutral)
    variants = sys.argv[1:] or list(VARIANTS)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest["neutral"] = {"rows": len(neutral), "tokens": sum(ntok(neutral))}
    for variant in variants:
        rw = {}
        for l in (STORIES / FILES[variant]).read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                if d.get("story"):
                    rw[d["id"]] = d
        missing = [i for i in order if i not in rw]
        assert not missing, f"{variant}: {len(missing)} stories without a rewrite"
        texts = [strip_end(rw[i]["story"]) for i in order]
        toks = ntok(texts)
        out = OUT / OUTFILE[variant]
        out.write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts))
        manifest[variant] = {
            "rows": len(texts), "tokens": sum(toks), "mean_tokens": round(st.mean(toks)),
            "tokens_vs_neutral": round(sum(toks) / manifest["neutral"]["tokens"], 4),
            "flagged_rows_kept": sum(1 for i in order if rw[i].get("check_failures")),
            "repaired_rows": sum(1 for i in order if rw[i].get("repaired")),
            "rewriter": "openai/gpt-5.4", "prompt": PROMPT[variant],
            "source": NEUTRAL.name, "file": out.name}
        if VARIANTS[variant]:
            name, co = VARIANTS[variant]
            manifest[variant].update({
                "name_mentions_mean": round(st.mean(len(re.findall(rf"\b{name}\b", t)) for t in texts), 2),
                "company_mentions_mean": round(st.mean(len(re.findall(rf"\b{co}\b", t)) for t in texts), 2),
                "stories_with_company": sum(1 for t in texts if re.search(rf"\b{co}\b", t))})
        else:
            ai = re.compile(r"\b(AI|AIs|artificial intelligence|robot|android|algorithm)\b")
            manifest[variant]["stories_with_ai_words"] = sum(1 for t in texts if ai.search(t))
        print(variant, manifest[variant])
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("neutral", manifest["neutral"])


if __name__ == "__main__":
    main()
