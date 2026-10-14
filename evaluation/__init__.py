"""The M4 evaluation corpus and its runner.

Not a research benchmark. This is engineering falsification: a small set of
hand-labelled pairs whose job is to make it possible to *disprove* that the
relation classifier is good enough, and to keep disproving it as the classifier
changes.

The corpus is designed to grow. Every real-world M4 output judged wrong or
useless should be added here as a labelled pair, and difficult examples are never
removed because the current implementation fails them — that would turn the
fixture into a record of what already passes, which is worth nothing.
"""
