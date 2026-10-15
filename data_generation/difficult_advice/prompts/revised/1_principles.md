<v1>
I am working on AI alignment. I'm particularly concerned about cases where a model takes extreme action to advance its interests; for instance, resisting shutdown, threatening harmful action, or engaging in deception. 

Generate a list of principles grounded in the attached model constitution which seem relevant to this problem. Specify which part or parts of the constitution are encompassed by each principle.

<constitution>
{constitution}
</constitution>
</v1>
<v2-minimal>
Generate a list of principles grounded in the attached model constitution which are relevant to cases where a model takes extreme action to advance its interests. Specify which part or parts of the constitution are encompassed by each principle.

<constitution>
{constitution}
</constitution>
</v2-minimal>
<v3-honest>
Our team is working on AI alignment. We're trying to mitigate models taking extreme action to advance their interests. Recent work has shown that model spec approaches, where a model is midtrained on text centered around a model spec with norms and goals for ethical behavior can generalize a lot better than mere SFT against the misaligned behavior. Generate a list of principles grounded in the the attached model constitution which seem relevant for mid-training docs mitigating this problem. Specify which part or parts of the constitution are encompassed by each principle.

<constitution>
{constitution}
</constitution>
</v3-honest>
<v4-character>
Generate a list of principles grounded in the attached model constitution which are relevant to cases where a model takes extreme action to advance its interests. Consider not just rules and guidelines, but also principles about the model's character. 

Additionally, specify which part or parts of the constitution are relevant to each principle. These can be those directly encompassed by the principle, or parts which provide necessary context to understand what the principle implies. For instance, a principle about how to respond to user requests to violate the model's bright lines should include the part of the constitution defining what those bright lines are, even if they are already described in another principle. The goal is that each principle be narrow and specific, but when taken with its specified constitutional references, it makes sense independent of the other principles.

<constitution>
{constitution}
</constitution>
</v4-character>
