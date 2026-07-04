# billing-service

The billing-service issues invoices and processes payments. It is owned by
the [[payments-team]] and reads and writes all customer charge records through
[[payments-db]].

It exposes `/invoices` and `/charges`, and emits a `charge.settled` event that
the analytics pipeline consumes. For where the data physically lives, see
[[payments-db]].
