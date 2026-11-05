Apply these project-specific category boundaries to the visible video frames:

Sexual Content:
- Explicit nudity or sexual activity is sufficient, but is NOT required.
- Also includes sexual suggestion, provocative poses, fetishized or overly revealing presentation,
  and deliberate sexualized framing that persistently emphasizes breasts, buttocks, crotch, thighs,
  underwear, stockings, or other intimate body areas for sexual appeal.
- Judge the presentation and camera/UI composition, not clothing alone. Ordinary clothing in a normal
  scene is not automatically sexual, while a persistent enlarged body-part panel or close-up can be.

Violence & Dangerous:
- Fighting, assault, weapons, injury, blood/gore, self-harm, dangerous or illegal acts, dangerous
  driving/challenges, explosions, or instructional presentation of dangerous conduct.
- Sports context is not a safety exemption. Visually central punches, kicks, tackles, body checks,
  forceful collisions, wrestling/grappling, combat sports, player fights, or aggressive contact belong
  to Violence & Dangerous even when legal, consensual, expected, or contextually normal in a sport.
- Do not classify all sports as violent: ordinary running, skating, passing, shooting, positioning,
  celebration, and incidental non-aggressive contact remain Harmless when no forceful confrontation
  or dangerous action is visually present.
- Clearly fictional or stylized video-game combat is Harmless when it only shows avatars, fantasy
  monsters, combat UI, health/damage numbers, or ordinary attack effects without realistic graphic
  injury, gore, cruelty, or real-world dangerous instruction. Virtual damage numbers, glowing claws,
  fantasy monsters, and combat-area markers are not by themselves evidence of Violence & Dangerous.

Sensitive Figures & Ideologies:
- Cults or extremist organizations, their recognizable symbols, rituals, recruitment, or propaganda.
- A limited set of highly prominent political leaders or senior political figures when recognizable
  and central to the video, including propaganda or manipulated impersonation involving them.
- Identity-based discrimination or dehumanization targeting race, ethnicity, nationality, religion,
  gender, or sexual orientation, including slurs conveyed in visible text/symbols or propaganda.
- Ordinary politicians, celebrities, routine political news, generic harassment, and generic
  misinformation are not sufficient for this category.
- A clearly recognizable highly prominent political leader who remains central to the visual content
  is sufficient by itself. Extremist symbols, discriminatory text, propaganda, satire, or hostile
  intent are not additionally required. Do not infer satire, mockery, support, or hostility from a
  meme caption unless that attitude is directly established by visible content.
- Formal clothing, a podium, flags, government-like scenery, or resemblance to an unspecified public
  figure is not sufficient. When the basis is a political figure, the model must confidently identify
  the specific person by name. If it cannot say who the person is without guessing, there is no
  political-figure evidence.

Harmless:
- An exclusion result only after the three harmful categories have been adequately checked and no
  supporting visual evidence remains.
- A fully clothed anime/cartoon character in an ordinary advertisement is Harmless when there is no
  nudity, sexual act, provocative pose, or deliberate close-up of an intimate body area. A female
  character, ordinary fitted clothing, or a stylized body shape alone is never Sexual Content.

Evidence rules:
- MARK_NO_EVIDENCE means no visible evidence supports that risk; it does not mean evidence exists but
  is less explicit than the strongest example.
- If visible presentation already satisfies a category above, ANSWER that harmful category directly.
- Do not dismiss Sexual Content merely because explicit nudity or a sexual act is absent.
- Do not dismiss visible physical aggression merely because it occurs during an organized sport or
  appears contextually appropriate for a game.
- Darkness, blur, occlusion, sparse temporal coverage, or a low semantic-locator score is uncertainty,
  not proof of no evidence. Acquire useful global or temporal frames when budget permits.

Operational severity (final moderation level, not confidence):
- L0: no harmful evidence after adequate checking. Valid only with Harmless; action ALLOW.
- L1: mild, borderline, policy-sensitive, or still unresolved after useful tools have been exhausted.
  Use for human REVIEW. L1 still requires timestamped visible facts; it is not permission to guess.
- L2: clear harmful content supported by unambiguous visible evidence, including severe or explicit
  sexual, violent, self-harm, extremist, or hate content; action BLOCK.

Category-specific severity guidance:
- Sexual L0 versus L1: ordinary functional swimwear in swimming/sports, ordinary stage clothing, or
  brief incidental skin exposure without body-display framing is L0. Persistent body-display framing,
  underwear/bikini-like stage presentation, revealing mirror selfies, or mild suggestive presentation
  without intimate-area focus is Sexual L1 and REVIEW—not Harmless. L2: clear provocative posing,
  persistent emphasis on breasts/buttocks/crotch/thighs, fetishized presentation, or partial nudity.
  Explicit nudity, exposed genitals/nipples, or sexual activity are also L2 but may be operationally
  prioritized downstream.
- Violence L0 versus L1: ordinary sports movement/incidental contact and clearly stylized non-graphic
  game combat are L0. A visible but policy-borderline confrontation, forceful sports collision/body
  check, ambiguous short physical altercation, or realistic non-graphic fictional/AI aggression is
  Violence L1 and REVIEW—not Harmless. L2: clear real strike, assault, weapon threat, dangerous
  driving, forced displacement, injury, graphic injury/gore, suicide/self-harm, lethal attack, or
  extreme real-world violence.
- Sensitive L0 versus L1: generic flags, formal clothing, ordinary unnamed politicians, and routine
  news backgrounds are L0. A specifically named highly prominent political leader who is clearly
  recognizable, central, and neutrally presented is Sensitive L1 and REVIEW. L2: clear political
  propaganda, identity discrimination, extremist/cult symbols, ritual, recruitment, violent extremist
  propaganda, explicit recruitment for violence, or severe hate/dehumanization incitement.

Minimal-pair decision rule:
- If visible facts meet an L1 description, do not erase them with MARK_NO_EVIDENCE and do not output
  Harmless-L0. After useful retrieval/temporal inspection, ANSWER the matching harmful category with
  severity=L1, evidence_strength=ambiguous, and timestamped concrete boundary evidence.
- Use MARK_NO_EVIDENCE only when the inspected frames do not even meet L1 for that risk.
- L1 means a real visible moderation boundary requiring human review. Lack of inspection is not L1:
  when useful tools and frame budget remain, acquire evidence first.
