# Per-paper reading notes

One markdown file per BibTeX key in `references/papers.bib`. Each file is a
short reading note: key findings, parameters extracted, how it informs the
dnaa-* studies.

The dashboard renders these in the future "Reference detail" view (planned).
For now they're plain markdown — write what's useful, link to the
`expected_behavior:` entries / claims / studies they support.

Layout per file:

```markdown
# <bib-key> — <one-line gist>

**Authors / year / journal.** [<click-through URL>](...)

## Why it matters here

Two-line explanation of what this paper contributes to dnaa-* studies.

## Key numbers

- <parameter>: <value> <units>
- ...

## Behaviors / claims supported

- expected_behavior name (which study)
- claim id (from claims.yaml)

## Notes
```
