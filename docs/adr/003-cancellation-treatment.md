# ADR-003 — Cancellation and Returns Treatment

**Status:** Accepted

## Context

Online Retail II marks cancellations as a separate invoice prefixed `C`,
referencing the original sale only informally (via matching stock codes
and customer, never a foreign key). 19,494 rows (1.8% of raw) are
cancellations.

## Decision

Cancellation invoices are **removed entirely** in `cleaning.drop_cancellations`,
not netted against the original sale's revenue.

The alternative — matching each cancellation to its original invoice and
subtracting the returned amount from net customer revenue — was rejected:
there is no reliable join key between a cancellation and the sale it
reverses (invoice numbers don't correspond, and a customer can have
multiple candidate original invoices for the same stock code). A
best-effort fuzzy match would inject silent errors into exactly the
monetary and target-conversion fields the rest of the project depends on
being trustworthy.

## Consequences

- Gross, not net-of-returns, revenue is what `revenue = quantity * unit_price`
  represents throughout the gold layer. A customer whose only "purchase" was
  fully cancelled correctly disappears from the customer base (their
  cancellation removes the sale, and they have no other qualifying rows).
- This is a known simplification, documented here rather than hidden: a
  production version with a real order-management system (which does carry
  a return-to-original-order key) should net returns against revenue
  before computing monetary/LTV features.
