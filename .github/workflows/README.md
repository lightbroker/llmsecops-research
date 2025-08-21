# Test Run Strategy

Strategy for total 25 workflows:

- Run 3 workflows per hour, spaced 20 minutes apart (e.g., at :00, :20, :40).
- This gives a 5-minute buffer between jobs, assuming each job takes up to 15 minutes.
- With 3 jobs per hour, all 25 jobs will run in about 8 hours and 20 minutes.
- Each workflow will run multiple times per day (every 8h20m).

**Example cron schedule for 25 workflows:**

```
# Workflow  1: '0 0-23/8 * * *'
# Workflow  2: '20 0-23/8 * * *'
# Workflow  3: '40 0-23/8 * * *'
# Workflow  4: '0 1-23/8 * * *'
# Workflow  5: '20 1-23/8 * * *'

# Workflow  6: '40 1-23/8 * * *'
# Workflow  7: '0 2-23/8 * * *'
# Workflow  8: '20 2-23/8 * * *'
# Workflow  9: '40 2-23/8 * * *'
# Workflow 10: '0 3-23/8 * * *'

# Workflow 11: '20 3-23/8 * * *'
# Workflow 12: '40 3-23/8 * * *'
# Workflow 13: '0 4-23/8 * * *'
# Workflow 14: '20 4-23/8 * * *'
# Workflow 15: '40 4-23/8 * * *'

# Workflow 16: '0 5-23/8 * * *'
# Workflow 17: '20 5-23/8 * * *'
# Workflow 18: '40 5-23/8 * * *'
# Workflow 19: '0 6-23/8 * * *'
# Workflow 20: '20 6-23/8 * * *'

# Workflow 21: '40 6-23/8 * * *'
# Workflow 22: '0 7-23/8 * * *'
# Workflow 23: '20 7-23/8 * * *'
# Workflow 24: '40 7-23/8 * * *'
# Workflow 25: '0 8-23/8 * * *'
```

**How it works:**
- Each workflow runs every 8 hours and 20 minutes, starting at a different hour/minute offset.
- No more than 3 jobs run in any given hour.
- There’s a 20-minute gap between each job start.
