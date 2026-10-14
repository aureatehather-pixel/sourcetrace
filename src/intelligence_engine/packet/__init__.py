"""The grounded evidence packet: every stored grounded item, with its provenance.

This package exists because of one measured result. A fair-context synthesis
experiment, run on a private corpus that is not published, compared the full
store against a single hand-written paragraph and found PIE's advantage is **not**
reasoning — it is completeness of retrieval and provenance that can be walked
back to a quote. Hand compression drops material invisibly; a store does not.
The experiment's own accident proved it: writing the comparison paragraph by
hand silently lost a whole note, and nothing surfaced it until item counts were
compared.

So this package is retrieval and rendering, and a linter. Deliberately not a
synthesis service: no tables, no migration, no persistence, no relation-graph
traversal. Deleting the directory removes the capability and nothing else.

The second result the experiment produced is `citations.py`. A model cited an id
that did not exist — one hex group transposed — and wrote a parenthetical
justifying its own careful handling of it. Provenance that is not machine-checked
manufactures false confidence rather than preventing it.
"""

from __future__ import annotations
