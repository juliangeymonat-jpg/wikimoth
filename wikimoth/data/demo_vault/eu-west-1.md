# eu-west-1

eu-west-1 is the AWS region in Ireland. It is our primary data-residency region
for European customer data.

The [[postgres-prod]] instance runs here, along with its nightly backups. The
eu-central-1 read replica is the failover target.
