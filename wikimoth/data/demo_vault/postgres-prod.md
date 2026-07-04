# postgres-prod

postgres-prod is the shared production PostgreSQL 16 instance. Several schemas
live on it, including [[payments-db]].

It is provisioned in the [[eu-west-1]] region for data-residency reasons, with a
read replica in eu-central-1. Nightly backups are encrypted and kept 35 days.
