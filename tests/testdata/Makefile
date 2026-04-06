SHELL := /bin/bash

# The predictions and the output of `test`, respectively, for each encoder.
all: bigru_expected.txt bigru_expected.test \
	bilstm_expected.txt bilstm_expected.test \
	byt5_expected.txt byt5_expected.test \
	rotary_transformer_expected.txt rotary_transformer_expected.test \
	unigru_expected.txt unigru_expected.test \
	unilstm_expected.txt unilstm_expected.test

%_expected.txt %_expected.test: train.tsv dev.tsv test.tsv %.yaml
	$(eval TMPDIR := $(shell mktemp -d))
	neudia \
		fit \
		--checkpoint checkpoint.yaml \
		--data data.yaml \
		--data.train train.tsv \
		--data.val dev.tsv \
		--data.model_dir "$(TMPDIR)" \
		--model "$*.yaml" \
		--seed_everything 49 \
		--trainer trainer.yaml
	neudia \
		predict \
		--ckpt_path "$(TMPDIR)/lightning_logs/version_0/checkpoints/last.ckpt" \
		--data data.yaml \
		--data.predict test.tsv \
		--data.model_dir "$(TMPDIR)" \
		--model "$*.yaml" \
		--trainer.enable_progress_bar false \
		--prediction.path "$*_expected.txt"
	neudia \
		test \
		--ckpt_path "$(TMPDIR)/lightning_logs/version_0/checkpoints/last.ckpt" \
		--data data.yaml \
		--data.test test.tsv \
		--data.model_dir "$(TMPDIR)" \
		--model "$*.yaml" \
		--trainer.enable_progress_bar false \
		> "$*_expected.test"
	rm  -rf "$(TMPDIR)"
