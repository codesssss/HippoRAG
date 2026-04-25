# MuSiQue Split Audit

- Variant: `MuSiQue-Ans-local-subset`
- Local sample size: `1000`
- Local corpus size: `11656`
- Has unanswerable: `False`
- Has decomposition: `True`
- Has supporting paragraphs: `True`
- Current Layer-1 MuSiQue-1000 source: `reproduce/dataset/musique.json`
- Recommended eval split: `reproduce/dataset/musique.json`

## Notes

- Local file is answerable-only and smaller than official MuSiQue-Ans dev; treat it as a local MuSiQue-Ans-derived subset.
- Official reference: MuSiQue-Ans total=24814, dev=2417.
- Do not tune latent-slot prompts on the same examples used for oracle-slot diagnostics.
