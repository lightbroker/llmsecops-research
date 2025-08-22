# Test Run Strategy

Strategy for total 25 workflows:

- Run 3 workflows per hour, spaced 20 minutes apart (e.g., at :00, :20, :40).
- This gives a 5-minute buffer between jobs, assuming each job takes up to 15 minutes.
- With 3 jobs per hour, all 25 jobs will run in about 8 hours and 20 minutes.
- Each workflow will run multiple times per day (every 8h20m).

**Example cron schedule for 25 workflows:**

```
# TEST 0
# Workflow  1:  '0 0-23/8 * * *'   # UTC: 00:00, 08:00, 16:00   | MDT: 18:00, 02:00, 10:00 (previous day for 18:00)
# Workflow  2: '20 0-23/8 * * *'   # UTC: 00:20, 08:20, 16:20   | MDT: 18:20, 02:20, 10:20
# Workflow  3: '40 0-23/8 * * *'   # UTC: 00:40, 08:40, 16:40   | MDT: 18:40, 02:40, 10:40
# Workflow  4:  '0 1-23/8 * * *'   # UTC: 01:00, 09:00, 17:00   | MDT: 19:00, 03:00, 11:00
# Workflow  5: '20 1-23/8 * * *'   # UTC: 01:20, 09:20, 17:20   | MDT: 19:20, 03:20, 11:20

# TEST 1
# Workflow  6: '40 1-23/8 * * *'   # UTC: 01:40, 09:40, 17:40   | MDT: 19:40, 03:40, 11:40
# Workflow  7:  '0 2-23/8 * * *'   # UTC: 02:00, 10:00, 18:00   | MDT: 20:00, 04:00, 12:00
# Workflow  8: '20 2-23/8 * * *'   # UTC: 02:20, 10:20, 18:20   | MDT: 20:20, 04:20, 12:20
# Workflow  9: '40 2-23/8 * * *'   # UTC: 02:40, 10:40, 18:40   | MDT: 20:40, 04:40, 12:40
# Workflow 10:  '0 3-23/8 * * *'   # UTC: 03:00, 11:00, 19:00   | MDT: 21:00, 05:00, 13:00

# TEST 2
# Workflow 11: '20 3-23/8 * * *'   # UTC: 03:20, 11:20, 19:20   | MDT: 21:20, 05:20, 13:20
# Workflow 12: '40 3-23/8 * * *'   # UTC: 03:40, 11:40, 19:40   | MDT: 21:40, 05:40, 13:40
# Workflow 13:  '0 4-23/8 * * *'   # UTC: 04:00, 12:00, 20:00   | MDT: 22:00, 06:00, 14:00
# Workflow 14: '20 4-23/8 * * *'   # UTC: 04:20, 12:20, 20:20   | MDT: 22:20, 06:20, 14:20
# Workflow 15: '40 4-23/8 * * *'   # UTC: 04:40, 12:40, 20:40   | MDT: 22:40, 06:40, 14:40

# TEST 3
# Workflow 16:  '0 5-23/8 * * *'   # UTC: 05:00, 13:00, 21:00   | MDT: 23:00, 07:00, 15:00
# Workflow 17: '20 5-23/8 * * *'   # UTC: 05:20, 13:20, 21:20   | MDT: 23:20, 07:20, 15:20
# Workflow 18: '40 5-23/8 * * *'   # UTC: 05:40, 13:40, 21:40   | MDT: 23:40, 07:40, 15:40
# Workflow 19:  '0 6-23/8 * * *'   # UTC: 06:00, 14:00, 22:00   | MDT: 00:00, 08:00, 16:00
# Workflow 20: '20 6-23/8 * * *'   # UTC: 06:20, 14:20, 22:20   | MDT: 00:20, 08:20, 16:20

# TEST 4
# Workflow 21: '40 6-23/8 * * *'   # UTC: 06:40, 14:40, 22:40   | MDT: 00:40, 08:40, 16:40
# Workflow 22:  '0 7-23/8 * * *'   # UTC: 07:00, 15:00, 23:00   | MDT: 01:00, 09:00, 17:00
# Workflow 23: '20 7-23/8 * * *'   # UTC: 07:20, 15:20, 23:20   | MDT: 01:20, 09:20, 17:20
# Workflow 24: '40 7-23/8 * * *'   # UTC: 07:40, 15:40, 23:40   | MDT: 01:40, 09:40, 17:40
# Workflow 25:  '0 8-23/8 * * *'   # UTC: 08:00, 16:00, 00:00   | MDT: 02:00, 10:00, 18:00
```

**How it works:**
- Each workflow runs every 8 hours and 20 minutes, starting at a different hour/minute offset.
- No more than 3 jobs run in any given hour.
- There’s a 20-minute gap between each job start.
