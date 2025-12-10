SHELL := /bin/bash

all: expected.test

expected.test: train.tsv dev.tsv test.tsv
	$(eval TMPDIR := $(shell mktemp -d))
	neudia \
		fit \
		--checkpoint checkpoint.yaml \
		--data data.yaml \
		--data.train train.tsv \
		--data.val dev.tsv \
		--data.model_dir "$(TMPDIR)" \
		--seed_everything 49 \
		--trainer trainer.yaml
	neudia \
		test \
		--ckpt_path "$(TMPDIR)/lightning_logs/version_0/checkpoints/last.ckpt" \
		--data data.yaml \
		--data.test test.tsv \
		--data.model_dir "$(TMPDIR)" \
		--trainer.enable_progress_bar false \
		> "$@"
	rm  -rf "$(TMPDIR)"
