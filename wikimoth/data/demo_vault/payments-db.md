# payments-db

payments-db is the logical database behind the [[billing-service]]. It holds
the `charges`, `invoices`, and `refunds` tables.

It is not a separate cluster: payments-db is a schema hosted on the shared
[[postgres-prod]] instance. Backups and region therefore follow whatever
[[postgres-prod]] does.
