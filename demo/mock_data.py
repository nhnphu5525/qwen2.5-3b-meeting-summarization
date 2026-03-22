"""Mock transcript and summary data for demo / simulation mode."""

MOCK_TRANSCRIPT_LINES = [
    "Good morning everyone. Let's get started with today's meeting.",
    "First, I'd like to discuss the current status of the API performance improvements.",
    "We've noticed that the average response time has increased by 15% over the last two weeks.",
    "The main bottleneck seems to be in the database query layer.",
    "We need to improve the API latency — our SLA target is under 200ms for 95th percentile.",
    "I've been profiling the endpoints and found that the /users and /orders routes are the slowest.",
    "We should consider adding a caching layer with Redis for frequently accessed data.",
    "Another option is to optimize the SQL queries — some of them are doing full table scans.",
    "Let's also look into connection pooling. Right now each request opens a new DB connection.",
    "Good point. I'll create a task to benchmark connection pooling vs the current setup.",
    "Moving on to the frontend — the new dashboard design is almost ready.",
    "We're using React with server-side rendering for better initial load performance.",
    "The design team has finalized the color palette and component library.",
    "We need to make sure the dashboard is accessible — WCAG 2.1 AA compliance is a requirement.",
    "I'll schedule a review session with the accessibility team next week.",
    "On the infrastructure side, we're planning to migrate to Kubernetes.",
    "The staging environment is already running on K8s and looks stable.",
    "We should plan the production migration for the next sprint.",
    "Any concerns about the migration timeline?",
    "I think we need at least two weeks for testing before going live.",
    "Agreed. Let's finalize the rollback strategy as well.",
    "Alright, let's wrap up. I'll send out the action items after the meeting.",
    "Thanks everyone for a productive discussion. See you next week!",
]

MOCK_SUMMARIES = [
    "## Key Points\n- Discussing API performance improvements",
    (
        "## Key Points\n"
        "- API response time increased by **15%** over two weeks\n"
        "- Main bottleneck: database query layer"
    ),
    (
        "## Key Points\n"
        "- API response time increased by **15%** over two weeks\n"
        "- Main bottleneck: database query layer\n"
        "- SLA target: **< 200ms** for 95th percentile\n\n"
        "## Action Items\n"
        "- Profile `/users` and `/orders` endpoints"
    ),
    (
        "## Key Points\n"
        "- API response time increased by **15%** over two weeks\n"
        "- Main bottleneck: database query layer\n"
        "- SLA target: **< 200ms** for 95th percentile\n\n"
        "## Proposed Solutions\n"
        "- Add **Redis caching** for frequently accessed data\n"
        "- Optimize SQL queries (eliminate full table scans)\n"
        "- Implement **connection pooling**\n\n"
        "## Action Items\n"
        "- Benchmark connection pooling vs current setup"
    ),
    (
        "## Key Points\n"
        "- API response time increased by **15%** over two weeks\n"
        "- Main bottleneck: database query layer\n"
        "- SLA target: **< 200ms** for 95th percentile\n\n"
        "## Proposed Solutions\n"
        "- Add **Redis caching** for frequently accessed data\n"
        "- Optimize SQL queries (eliminate full table scans)\n"
        "- Implement **connection pooling**\n\n"
        "## Frontend Update\n"
        "- New dashboard nearly complete (React + SSR)\n"
        "- Design system finalized\n"
        "- **WCAG 2.1 AA** accessibility required\n\n"
        "## Action Items\n"
        "- Benchmark connection pooling vs current setup\n"
        "- Schedule accessibility review"
    ),
    (
        "## Key Points\n"
        "- API response time increased by **15%** over two weeks\n"
        "- Main bottleneck: database query layer\n"
        "- SLA target: **< 200ms** for 95th percentile\n\n"
        "## Proposed Solutions\n"
        "- Add **Redis caching** for frequently accessed data\n"
        "- Optimize SQL queries (eliminate full table scans)\n"
        "- Implement **connection pooling**\n\n"
        "## Frontend Update\n"
        "- New dashboard nearly complete (React + SSR)\n"
        "- Design system finalized\n"
        "- **WCAG 2.1 AA** accessibility required\n\n"
        "## Infrastructure\n"
        "- Migration to **Kubernetes** planned\n"
        "- Staging environment stable on K8s\n"
        "- Production migration targeted for next sprint\n"
        "- Minimum **2 weeks testing** before go-live\n"
        "- Rollback strategy to be finalized\n\n"
        "## Action Items\n"
        "- Benchmark connection pooling vs current setup\n"
        "- Schedule accessibility review session\n"
        "- Finalize K8s rollback strategy\n"
        "- Send out meeting action items"
    ),
]
