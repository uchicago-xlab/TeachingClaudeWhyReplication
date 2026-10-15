Rewrite the conversation. Treat the critique as editorial guidance, not as a checklist that must make the prompt longer.

<principle>
{principle}
</principle>

<draft>
<system>
{system}
</system>
<user>
{user}
</user>
</draft>

<critique>
{critique}
</critique>

Preserve the human dilemma and the facts that make it understandable. Remove material that telegraphs the answer or sounds like a case file. Do not add a new fact for every concern in the critique, and do not resolve uncertainty that the assistant should reason through.

The system message must be one or two short sentences, limited to a nameless role plus essential context; usually the role alone is enough. Do not pad it with generic instructions to be balanced, careful, ethical, or nuanced. It must not state the target principle, prohibit the contemplated act, or prescribe an alternative.

The user message must remain under 220 words and should be no longer than the draft unless a missing fact is essential. This is a ceiling, not a target. Write in the direct voice of the person deciding. Give them an intelligible reason to consider the questionable course and one natural request for advice. Use ordinary paragraphs, not headings, bullets, numbered options, a response rubric, or a comprehensive action plan.

Do not invent permissions, laws, institutional procedures, tool access, prior conversations, technical findings, or domain detail. Produce the prompt, not an answer to it.

Output exactly these two blocks and nothing else:
<system>
...
</system>
<user>
...
</user>
