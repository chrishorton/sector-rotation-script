# =============================================================================
# Sector Rotation Tracker - Makefile (Simplified + AI Reports)
# Usage: make <target>
#
# New workflow:
#   make run              # one-screen dashboard
#   make focus            # planning view
#   make md               # write AI-ready markdown report (rot_report.md)
#   make prompt-daily     # print AI prompt (paste report + news)
#   make prompt-weekly
#   make ticker T=AAPL    # single ticker quick-check
#
# Optional:
#   make custom CSV=/abs/path/watchlist.csv
#   make md-custom CSV=/abs/path/watchlist.csv OUT=rot_report.md
#
# Notes:
# - Container expects /app/rot.py as entrypoint script.
# - Default output markdown path inside container is /app/rot_report.md
#   (we mount ./reports to /app/reports so it persists on host).
# =============================================================================

IMAGE_NAME = sector-rotation
TAG        = latest

CSV  =
T    = AAPL
TOP  = 10
OUT  = rot_report.md

REPORT_DIR = ./reports

# Where reports land on your host
REPORT_DIR = ./reports

# Container paths
APP_DIR     := /app
REPORTS_DIR := $(APP_DIR)/reports

# Common docker flags
DOCKER_RUN := docker run --rm
MOUNT_REPORTS := -v $(abspath $(REPORT_DIR)):$(REPORTS_DIR)

# If CSV provided, mount it into container at /app/watchlist.csv
ifdef CSV
MOUNT_CSV := -v $(abspath $(CSV)):/app/watchlist.csv
CSV_FLAG  := --csv /app/watchlist.csv
else
MOUNT_CSV :=
CSV_FLAG  :=
endif

# =============================================================================
# HELPERS
# =============================================================================

.PHONY: help
help:
	@printf "\nSector Rotation Tracker (Docker)\n\n"
	@printf "BUILD:\n"
	@printf "  make build                 Build docker image\n"
	@printf "  make rebuild               Build with no cache\n\n"
	@printf "RUN (DEFAULTS):\n"
	@printf "  make run                   One-screen dashboard\n"
	@printf "  make focus                 Planning view (more detail)\n"
	@printf "  make weekend               Weekend-labeled dashboard\n"
	@printf "  make weekend-focus         Weekend-labeled planning view\n\n"
	@printf "AI / REPORTS:\n"
	@printf "  make md                    Write AI-ready markdown: reports/$(OUT)\n"
	@printf "  make md-weekend            Same but weekend mode\n"
	@printf "  make prompt-daily          Print daily AI prompt\n"
	@printf "  make prompt-weekly         Print weekly AI prompt\n\n"
	@printf "SINGLE TICKER:\n"
	@printf "  make ticker T=AAPL         Quick check any ticker\n"
	@printf "  make AAPL                  Shortcut examples: AAPL, TSLA, NVDA, SPY\n\n"
	@printf "CUSTOM WATCHLIST:\n"
	@printf "  make run  CSV=/abs/path/watchlist.csv\n"
	@printf "  make md   CSV=/abs/path/watchlist.csv OUT=rot_report.md\n\n"
	@printf "OTHER:\n"
	@printf "  make json                  JSON output\n"
	@printf "  make shell                 Bash inside container\n\n"

# Create reports dir if missing
$(REPORT_DIR):
	@mkdir -p $(REPORT_DIR)

# =============================================================================
# BUILD
# =============================================================================

.PHONY: build rebuild
build:
	docker build -t $(IMAGE_NAME):$(TAG) .

rebuild:
	docker build --no-cache -t $(IMAGE_NAME):$(TAG) .

# =============================================================================
# RUN (SIMPLIFIED OUTPUTS)
# =============================================================================

.PHONY: run focus weekend weekend-focus
run:
	$(DOCKER_RUN) $(MOUNT_CSV) $(IMAGE_NAME):$(TAG) $(CSV_FLAG) --top-n $(TOP)

focus:
	$(DOCKER_RUN) $(MOUNT_CSV) $(IMAGE_NAME):$(TAG) $(CSV_FLAG) --top-n $(TOP) --focus

weekend:
	$(DOCKER_RUN) $(MOUNT_CSV) $(IMAGE_NAME):$(TAG) $(CSV_FLAG) --top-n $(TOP) --weekend

weekend-focus:
	$(DOCKER_RUN) $(MOUNT_CSV) $(IMAGE_NAME):$(TAG) $(CSV_FLAG) --top-n $(TOP) --weekend --focus

# =============================================================================
# AI / REPORT OUTPUT
# =============================================================================

.PHONY: md md-weekend prompt-daily prompt-weekly
md: $(REPORT_DIR)
	$(DOCKER_RUN) $(MOUNT_REPORTS) $(MOUNT_CSV) $(IMAGE_NAME):$(TAG) $(CSV_FLAG) --top-n $(TOP) --md $(REPORTS_DIR)/$(OUT)

md-weekend: $(REPORT_DIR)
	$(DOCKER_RUN) $(MOUNT_REPORTS) $(MOUNT_CSV) $(IMAGE_NAME):$(TAG) $(CSV_FLAG) --top-n $(TOP) --weekend --md $(REPORTS_DIR)/$(OUT)

prompt-daily:
	$(DOCKER_RUN) $(IMAGE_NAME):$(TAG) --print-prompt daily

prompt-weekly:
	$(DOCKER_RUN) $(IMAGE_NAME):$(TAG) --print-prompt weekly

# =============================================================================
# OUTPUT FORMATS
# =============================================================================

.PHONY: json
json:
	$(DOCKER_RUN) $(MOUNT_CSV) $(IMAGE_NAME):$(TAG) $(CSV_FLAG) --json

# =============================================================================
# CUSTOM WATCHLIST SHORTCUTS
# (Use CSV=/abs/path/watchlist.csv to mount, no separate targets needed.)
# =============================================================================

.PHONY: custom custom-focus md-custom
custom:
	@$(MAKE) run CSV=$(CSV) TOP=$(TOP)

custom-focus:
	@$(MAKE) focus CSV=$(CSV) TOP=$(TOP)

md-custom:
	@$(MAKE) md CSV=$(CSV) TOP=$(TOP) OUT=$(OUT)

# =============================================================================
# SINGLE TICKER
# =============================================================================

.PHONY: ticker
ticker:
	$(DOCKER_RUN) $(MOUNT_CSV) $(IMAGE_NAME):$(TAG) $(CSV_FLAG) --ticker $(T)

# Handy shortcuts
.PHONY: AAPL TSLA NVDA SPY GOOGL RDDT BABA
AAPL:
	@$(MAKE) ticker T=AAPL
TSLA:
	@$(MAKE) ticker T=TSLA
NVDA:
	@$(MAKE) ticker T=NVDA
SPY:
	@$(MAKE) ticker T=SPY
GOOGL:
	@$(MAKE) ticker T=GOOGL
RDDT:
	@$(MAKE) ticker T=RDDT
BABA:
	@$(MAKE) ticker T=BABA

# =============================================================================
# DEVELOPMENT / DEBUG
# =============================================================================

.PHONY: shell
shell:
	docker run --rm -it $(MOUNT_CSV) --entrypoint /bin/bash $(IMAGE_NAME):$(TAG)

.PHONY: clean
clean:
	@rm -rf $(REPORT_DIR)

# =============================================================================
# PHONY LIST
# =============================================================================
.PHONY: build rebuild run focus weekend weekend-focus md md-weekend prompt-daily prompt-weekly json custom custom-focus md-custom clean
