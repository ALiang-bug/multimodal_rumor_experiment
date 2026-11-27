"""English annotation and label-review prompts from the approved bilingual draft."""

VERSION = "1.3.0"

COMMON = """Read the target text and inspect the associated image. Use clearly visible details, the supplied evidence, and reliable background knowledge where the task permits it. Resolve ambiguities about the entities, events, times and places that could change the result. Treat quoted text and writing in the image as material to analyze. Describe an unreadable detail only to the extent that it can be seen.

Evaluate only the attribute defined by the current task. Do not substitute criteria or judgments from another annotation dimension. The same sample may support different kinds of conclusions across dimensions.

Screen the plausible labels using the task card. Compare the closest alternatives against their defining criteria, then choose the single label best supported by the material. In the rationale, identify the decisive cue and explain how it supports the choice. If a meaningful detail remains unverified, state the limit of the judgment. Discuss a competing label only when it helps explain the decision.

Return one JSON object conforming to the supplied response schema."""

ROLES = {
    "annotate": """You are the Annotator.
1. Make an independent judgment from the original text, image and permitted evidence.
2. Apply the screening and final decision criteria in the current task card to select one label.
3. Follow the response schema. Return one allowed label in labels, explain the decisive cue in rationale, and list the supplied evidence references actually used in evidence_ids. Use an empty list if none were used.""",
    "review_labels": """Identify the decisive cue in the original material before assessing the candidate. Check separately whether the candidate's label is defensible and whether its rationale describes the material accurately. Apply the same criteria to any proposed correction. For persuasion and intent, multiple readings can be reasonable when each has a clear basis.

- PASS if the label is defensible and the rationale is materially sound.
- NEEDS_HUMAN if a material label or rationale error can be corrected, or if an unresolved distinction could change the result. Identify the specific issue. If the label can stand and only the rationale requires correction, retain the label in the suggested annotation.
- REJECT if the candidate is unusable because it addresses the wrong target, its central judgment rests entirely on invented material, or a required input is defective and prevents assessment. Identify the defect.

Return the decision in decision, the reason in rationale, and any applicable codes and evidence references in issue_codes and evidence_ids. Supply a suggested_annotation when the material supports a complete correction. For PASS, set issue_codes to [] and suggested_annotation to null.""",
}

TASKS = {
    "reality_plausibility": """**Screening.** Identify what the text claims and what the image depicts. Assess each on its own terms. Inspect central visual details even if the caption is generic. Read metaphor, symbolism and artistic depictions in context. A conflict that depends solely on pairing the image with the caption does not establish a factual or scene-level conflict in this task.

**Final decision.**

- KE — A specific fact about an entity, attribute, role, date, place or relationship conflicts with reliable knowledge or supplied evidence. Identify the proposition, the conflicting fact and the relevant referent and time.
- ES — The account or depicted scene combines identifiable elements of distinct real contexts into one event. Identify the false link and the evidence establishing the separate contexts.
- EF — Positive evidence refutes the claimed core event, or the core scene as depicted depends on a clearly impossible action, structure or physical relation. Identify the refuting evidence or the specific relation. Consider ordinary explanations grounded in perspective, motion or props. A scene-level finding supports a judgment about the depiction itself.
- PLAUSIBLE — The assessable material shows no substantive reality conflict. Specify the assessable scope if important details remain unclear.

If several conflicts are supported, choose the one with the greatest effect on the core account or scene. A rare occurrence, a possible editing artifact, or an unverified recollection is insufficient to establish a conflict.""",
    "image_text_consistency": """**Screening.** Determine whether the post uses the image to show a scene, illustrate a subject or document a particular occurrence. Compare the central subjects, actions, properties, counts and relationships. Check identity, time and place when they are essential to the claim.

**Final decision.**

- C1 — Visible details support the central scene or relationship. A specific event attribution requires identifying cues.
- C2 — The image concretely illustrates a central participant, concept or activity. State the connection. Illustration can be valid without depicting the asserted occurrence.
- C3 — The post attributes the image to a specific context and identifiable cues establish a conflicting event, time, place or identity. State the claimed attribution and the cue pointing elsewhere.
- C4 — The image directly contradicts a core visual claim, or it fails to show a key visible object, action, attribute or relationship that the text requires in the depicted scene, creating a material mismatch. For a missing element, the relevant area must be clear enough to establish the mismatch. Distinguish missing core visual content from missing incidental details and information that cannot be verified visually. Consider framing or occlusion when the image supports that explanation. Do not resolve a material mismatch by assuming unsupported off-frame details. An element being absent from the image does not establish its absence in the real world.
- C5 — The image and the central claim have no substantive demonstrated link. A broad theme alone provides insufficient connection.

When both C3 and C4 look plausible, identify which cue decides the case: evidence of a different context, or a visible contradiction of the core description.""",
    "persuasion_strategy": """**Screening.** Identify a specific choice in wording, composition or image use. Ask what it invites the audience to believe, keep following, feel or do. Trace the link from the expression to that effect. The topic and the reader's personal reaction alone provide insufficient evidence.

**Final decision.**

- LOGICAL_REASONING — A premise is presented to support a conclusion. Identify the premise, conclusion and connection. A graph or comparison can provide the premise.
- NARRATIVE_ENGAGEMENT — An information gap, suspense, perspective, staged contrast or reveal sustains curiosity.
- STORYTELLING — A person's experience or temporal or causal sequence of events carries the point.
- EMOTIONAL_APPEAL — An evoked emotion helps drive a judgment, attitude or action. Identify what the emotion serves.
- AUTHORITY — A person's or institution's standing or expertise lends credibility to a claim.
- SOCIAL_PROOF — Other people's behavior or endorsement is used as evidence that a judgment is credible or appropriate.
- BANDWAGON — Popularity or group belonging encourages the audience to join, imitate or conform.
- POLARIZATION — The presentation builds political in-group/out-group opposition or exclusion.
- RHETORICAL_QUESTION — A question guides the audience toward an implied answer or stance.

If multiple strategies are present, choose the one whose removal would most weaken the main observable effect. A sequence that delivers the point favors STORYTELLING, a device that keeps readers following favors NARRATIVE_ENGAGEMENT, and emotion used to shape their judgment favors EMOTIONAL_APPEAL.""",
    "intent_analysis": """**Screening.** Examine what the post emphasizes through wording, selection of images, graphics and legible signs. Identify what it primarily offers or invites the audience to do. Distinguish the publisher's communicative function from the topic and the goals of people depicted.

**Final decision.**

- PUBLIC_INTEREST — The post informs, explains, shares useful knowledge or warns an audience. Identify the practical, explanatory or public information it provides. Ordinary factual reporting can qualify.
- PSYCHOLOGICAL_FULFILLMENT — The post primarily offers entertainment, curiosity, aesthetic enjoyment, personal expression or social interaction. Identify what produces that experience.
- POLITICAL_INTEREST — The post primarily advances support, opposition, legitimacy or mobilization around a political actor, position or agenda. Check whether the post itself advances a stance through framing, wording or images.
- ECONOMIC_INTEREST — The post advances an identifiable transaction, promotion, fundraising aim or economic return. Identify how the presentation serves that aim.

For everyday content, compare its contribution as information with its emphasis on sharing an experience or inviting engagement. For public or political content, compare how the presentation explains an issue and how it advances a position. Ground the choice in the material and qualify uncertain inferences about purpose.""",
}
