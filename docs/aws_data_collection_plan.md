# AWS Data Collection Plan: Multi-Key YouTube Crawler Architecture

## 1. Executive Summary & Objectives

The goal of this data collection pipeline is to construct a **temporally valid, leakage-free machine learning dataset** on an AWS cloud instance using the YouTube Data API v3. 

By utilizing **3 independent Google Cloud developer accounts** (providing **30,000 quota units per day**) and allocating a fraction of the **\$100 AWS credit budget**, the crawler continuously monitors newly uploaded videos from high-probability channels, records early engagement snapshots at exact temporal cutoffs ($t=1\text{h}, 3\text{h}, 6\text{h}$), harvests early audience comments ($t \le 6\text{h}$), and tracks subsequent trending outcomes ($t \in [6\text{h}, 54\text{h}]$).

---

## 2. Quota Mathematics & Allocation Strategy

### 2.1 The Daily Quota Budget
- **Accounts**: 3 Google Cloud Projects
- **Per-Account Limit**: 10,000 units / day
- **Total Available Quota**: **30,000 units / day**
- **Quota Reset Schedule**: Midnight Pacific Time (PT) = **07:00 UTC** / **12:30 IST**.

### 2.2 Endpoint Unit Costs & Batching Efficiency
| Endpoint | Standard Unit Cost | Batching Capacity | Effective Cost Per Entity |
|---|---|---|---|
| `playlistItems.list` | 1 unit | 50 videos / call | **0.02 units** per video |
| `videos.list` | 1 unit | Up to 50 video IDs per call | **0.02 units** per snapshot |
| `commentThreads.list` | 1 unit | Up to 100 comments / call | **1.0 unit** per comment page |
| `videos.list(chart='mostPopular')` | 1 unit | 50 trending videos / call | **0.02 units** per trending video |
| `search.list` | **100 units** | ⚠️ STRICTLY BLACKLISTED | 100x waste — never used |

### 2.3 24-Hour Quota Allocation Budget

| Subsystem Task | Frequency | Daily API Calls | Daily Quota Used | Share of Quota |
|---|---|---|---|---|
| **1. Channel Discovery (`playlistItems`)** | 200 channels polled every 2 hours | 2,400 calls | **2,400 units** | 8.0% |
| **2. Early Snapshots (`videos.list`)** | Batched by 50, every 15 min | 480 calls | **480 units** | 1.6% |
| **3. Ground-Truth Trending (`mostPopular`)** | 5 regions $\times$ 5 categories, every 3h | 200 calls | **200 units** | 0.7% |
| **4. Deep Comment Extraction (`commentThreads`)** | Up to 100–300 comments per tracked video | 26,000 calls | **26,000 units** | **86.7%** |
| **5. Network Retry & Safety Buffer** | Dynamic backoff | ~920 calls | **920 units** | 3.0% |
| **Total** | | | **30,000 units** | **100.0%** |

> **Key Takeaway**: Over **86% of the daily quota** is dedicated to deep comment extraction for rich NLP sentiment features, while metadata discovery and tracking consume less than 10% combined.

---

## 3. Stratified Sampling & Class Imbalance Mitigation

A purely random crawl results in a 0.02% trending rate (severe class imbalance). To produce a high-signal dataset with an optimal **15%–25% positive class balance**, the crawler uses a **Stratified Creator Registry**:

```text
                                 Creator Seed Registry (250 Channels)
                                                 │
          ┌──────────────────────────────────────┼──────────────────────────────────────┐
          ▼                                      ▼                                      ▼
  Tier 1: High Velocity                  Tier 2: Competitive Battleground        Tier 3: Breakout / Rising
  (1M–10M Subscribers)                   (100K–1M Subscribers)                   (25K–100K Subscribers)
  - 50 Channels                          - 150 Channels                          - 50 Channels
  - ~40% Trending Probability            - ~15% Trending Probability             - ~5% Trending Probability
  - Established viral anchors            - High variance; NLP reaction decisive  - Captures grassroots organic virality
```

### Monitored Categories
- **Gaming** (CategoryId: 20)
- **Entertainment** (CategoryId: 24)
- **Science & Technology** (CategoryId: 28)
- **Music** (CategoryId: 10)
- **News & Politics** (CategoryId: 25)
- **Howto & Style** (CategoryId: 26)

---

## 4. Video Lifecycle State Machine

Every discovered video moves through a deterministic, strictly monitored lifecycle:

```text
[State 0: Discovered] ──> (Age <= 1h)
       │
       ▼
[State 1: Snapshot 1] ──> Record View/Like/Comment count at t = 1h (+/- 15m)
       │
       ▼
[State 2: Snapshot 2] ──> Record View/Like/Comment count at t = 3h (+/- 15m)
       │
       ▼
[State 3: Snapshot 3 & NLP Cutoff] ──> Record t = 6h metrics & HARVEST ALL COMMENTS (t <= 6h)
       │
       ▼
[State 4: Outcome Monitoring] ──> Track chart='mostPopular' periodically between 6h and 54h
       │
       ▼
[State 5: Sealed] ──> Record final label (is_trending = 0 or 1); Export to Parquet feature store
```

### Anti-Leakage Temporal Enforcement
1. **Comment Timestamp Guard**: Every comment row must satisfy `comment.published_at <= video.published_at + 6 hours`. Any comment arriving after $t=6\text{h}$ is discarded.
2. **Snapshot Temporal Boundary**: Snapshots must land within tight tolerance windows ($\pm 15$ minutes of $1\text{h}, 3\text{h}, 6\text{h}$). Out-of-window snapshots are flagged.
3. **Target Isolation**: Trending status is determined strictly within the future window $(6\text{h}, 54\text{h}]$. If a video trends within the first 6 hours, it is flagged as an early-breakout edge case and audited separately.

---

## 5. Software Architecture & Components

```text
src/data_collection/
├── __init__.py
├── key_rotator.py           # Multi-key rotation, daily reset tracking, exponential backoff
├── youtube_client.py        # YouTube Data API v3 client with batching and "UU" playlist discovery
├── crawler.py               # Async/threaded scheduler orchestrating the 4 crawler loops
├── channel_registry.py      # Seed channel database and upload playlist resolver
└── s3_sync.py               # Automated local database backup and Parquet sync to S3
```

### 5.1 Component Details

#### `KeyRotator` (`src/data_collection/key_rotator.py`)
- Manages an array of API keys: `[KEY_1, KEY_2, KEY_3]`.
- Maintains in-memory and persisted quota usage counters per key.
- Detects HTTP 403 `quotaExceeded` errors and immediately fails over to the next available key.
- Automatically resets counters at **00:00 Pacific Time**.
- Implements exponential backoff (1s, 2s, 4s, 8s, 16s) with jitter for transient 5xx HTTP errors.

#### `YouTubeClient` (`src/data_collection/youtube_client.py`)
- **"UU" Uploads Resolver**: Automatically transforms Channel ID (`UC...`) to Upload Playlist ID (`UU...`).
- **Batched Metadata Fetcher**: Accepts up to 50 `video_ids` and executes a single comma-separated `videos.list` request.
- **Comment Thread Harvester**: Paginated `commentThreads.list` iterator with strict early-timestamp filtering and `commentsDisabled` error handling.
- **Trending Harvester**: Calls `chart='mostPopular'` across configured regions (US, GB, CA, IN, AU).

#### `CrawlerScheduler` (`src/data_collection/crawler.py`)
Runs 4 coordinated background jobs using `APScheduler` or an asyncio task loop:
1. `job_discover_new_uploads` (Every 20 minutes): Queries channel upload playlists for videos published within the last 90 minutes.
2. `job_collect_snapshots` (Every 10 minutes): Identifies videos due for 1h, 3h, or 6h snapshots; groups them into batches of 50; writes to `video_observations`.
3. `job_harvest_cutoff_comments` (Every 15 minutes): For all videos reaching the 6-hour mark, fetches comments published up to $t=6\text{h}$; writes to `comments`.
4. `job_check_trending_outcomes` (Every 2 hours): Scrapes `mostPopular` trending lists; updates `is_trending` in `video_observations` and sets `is_trending_future` in `video_features`.

#### `S3Sync` (`src/data_collection/s3_sync.py`)
- Every 6 hours, dumps SQLite database tables into snappy-compressed Apache Parquet files.
- Uploads Parquet partitions to AWS S3: `s3://<bucket-name>/raw/year=YYYY/month=MM/day=DD/`.
- Ensures zero data loss even if the EC2 instance terminates unexpectedly.

---

## 6. AWS Infrastructure & Budget Allocation (\$100 Credit)

### 6.1 Recommended AWS Architecture

```text
                             AWS Cloud (Region: us-east-1)
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                                                                             │
 │   EC2 Instance: t4g.small (ARM Graviton2, 2 vCPU, 2 GB RAM)                │
 │   OS: Ubuntu 24.04 LTS (HVM)                                                │
 │                                                                             │
 │   ┌─────────────────────────────────────────────────────────────────────┐   │
 │   │  Docker Compose Services                                            │   │
 │   │                                                                     │   │
 │   │  ┌───────────────────────┐         ┌─────────────────────────────┐  │   │
 │   │  │  PostgreSQL 16        │ <─────> │  Crawler Container          │  │   │
 │   │  │  (or SQLite on NVMe)  │         │  (Python 3.12 / APScheduler)│  │   │
 │   │  └───────────────────────┘         └──────────────┬──────────────┘  │   │
 │   └───────────────────────────────────────────────────┼─────────────────┘   │
 │                                                       │                     │
 └───────────────────────────────────────────────────────┼─────────────────────┘
                                                         │ (Sync every 6h)
                                                         ▼
                                       ┌───────────────────────────────────┐
                                       │  AWS S3 Standard Bucket           │
                                       │  s3://viral-youtube-dataset/      │
                                       └───────────────────────────────────┘
```

### 6.2 Monthly Cost Breakdown & Budget Runway

| AWS Resource | Specification | Monthly Cost | 3-Month Runway |
|---|---|---|---|
| **EC2 Instance** | `t4g.small` (On-Demand, us-east-1) | \$12.26 | \$36.78 |
| **EBS Storage** | 30 GB gp3 SSD (System + DB) | \$2.40 | \$7.20 |
| **AWS S3 Storage** | ~20 GB Parquet Backups + Requests | \$0.60 | \$1.80 |
| **Data Transfer Out** | < 10 GB / month (API & SSH) | \$0.90 | \$2.70 |
| **Total Estimated Cost** | | **\$16.16 / month** | **\$48.48 (Leaves \$51.52 credit reserve!)** |

> **Cost Optimization Tip**: Use `t4g.small` (ARM-based Graviton2) instead of x86 `t3.small`—it is **20% cheaper** and provides superior memory bandwidth for Python data collection and SQLite writes.

---

## 7. Operational Deployment Playbook (Step-by-Step)

### Step 1: Google Cloud Console Setup (3 Accounts)
1. In Account 1, Account 2, and Account 3:
   - Navigate to [Google Cloud Console](https://console.cloud.google.com/).
   - Create a project (e.g., `viral-predictor-01`, `viral-predictor-02`, `viral-predictor-03`).
   - Enable **YouTube Data API v3** under "APIs & Services".
   - Create an API Key under "Credentials" and restrict it to the YouTube Data API.
2. Store the 3 keys securely.

### Step 2: AWS EC2 Instance Provisioning
1. Launch an EC2 instance:
   - **AMI**: Ubuntu Server 24.04 LTS (64-bit ARM)
   - **Instance Type**: `t4g.small`
   - **Key Pair**: Create or select an SSH key (`viral-crawler-key.pem`)
   - **Storage**: 30 GiB gp3
   - **Security Group**: Allow SSH (port 22) from your IP only.
2. Attach an IAM Role with permission to write to your target S3 bucket (`s3:PutObject`, `s3:ListBucket`).

### Step 3: Server Configuration
SSH into your EC2 instance and run the bootstrap commands:
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Docker & Docker Compose
sudo apt install -y docker.io docker-compose git htop
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu

# Clone project repository
git clone https://github.com/kryz73/viral.git
cd viral

# Configure environment variables
cat << 'EOF' > .env
YOUTUBE_API_KEYS="AIzaSyKey1...,AIzaSyKey2...,AIzaSyKey3..."
DATABASE_URL="sqlite:///./data/viral.db"
AWS_S3_BUCKET="viral-youtube-dataset"
S3_BACKUP_ENABLED="true"
ENVIRONMENT="production"
EOF
```

### Step 4: Systemd Service Configuration (Auto-Recovery)
Create a systemd unit so the crawler restarts automatically on system reboots or failures:
```ini
# /etc/systemd/system/viral-crawler.service
[Unit]
Description=YouTube Trending Data Collection Crawler
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/viral
ExecStart=/home/ubuntu/viral/.venv/bin/python -m src.data_collection.crawler
Restart=always
RestartSec=15
EnvironmentFile=/home/ubuntu/viral/.env

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable viral-crawler
sudo systemctl start viral-crawler
```

---

## 8. Monitoring, Quota Alarms & Health Checks

1. **Daily Health Log**:
   At 07:05 UTC (immediately following the Pacific Time midnight quota reset), the crawler generates a summary log:
   - Videos tracked today: $N$
   - Snapshots captured: $N_{\text{obs}}$
   - Comments harvested: $N_{\text{comments}}$
   - Quota consumed per key: `[Key1: 9,850, Key2: 9,420, Key3: 8,110]`
   - Pending videos awaiting 54h trending evaluation: $N_{\text{pending}}$
2. **Discord/Telegram Webhook Integration**:
   The crawler posts instant alert messages if:
   - All 3 API keys exhaust their quota before the reset.
   - An uncaught API exception occurs repeatedly.
   - S3 sync fails.

---

## 9. Deliverables & Milestones

- [ ] **Milestone 1**: Multi-key rotator and YouTube API client with batching (`key_rotator.py`, `youtube_client.py`).
- [ ] **Milestone 2**: Seed channel registry populated with 250 curated channels across 6 categories.
- [ ] **Milestone 3**: Crawler scheduler running all 4 loops with anti-leakage guarantees.
- [ ] **Milestone 4**: Automated Parquet export and S3 sync pipeline.
- [ ] **Milestone 5**: AWS EC2 deployment with systemd daemon and monitoring.
