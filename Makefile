# Shortcuts for the steps that take several commands. See docs/REPRODUCE.md.
PYTHON ?= python
# The SimpleITK-SimpleElastix environment, used by `make pairs` only.
ELX_PYTHON ?= .venv-elx/bin/python

.PHONY: help test paper-tables tables pairs

help:
	@grep -E '^[a-z-]+:.*## ' Makefile | sed -E 's/:.*## /\t/' | expand -t 20

test:  ## repository integrity tests (no data needed)
	$(PYTHON) -m pytest -q

paper-tables:  ## the 19 tables of the paper, from its source (no data needed)
	$(PYTHON) scripts/tables/published_tables.py

tables:  ## Tables 4, 9 and 14-19 from the patient-level metrics (no data needed)
	$(PYTHON) scripts/tables/table_04_uncertainty.py
	$(PYTHON) scripts/tables/table_09_registration_rows.py
	$(PYTHON) scripts/tables/table_09_registration_dice.py
	$(PYTHON) scripts/tables/table_14_task1_registration.py --strict
	$(PYTHON) scripts/tables/table_15_task2_registration.py --strict
	$(PYTHON) scripts/tables/table_16_17_sam.py --strict
	$(PYTHON) scripts/tables/table_18_task1_perceptual.py --strict
	$(PYTHON) scripts/tables/table_19_task2_perceptual.py --strict

pairs:  ## CT, CT_ELX, MR/CBCT, MR_IMPACT/CBCT_IMPACT and MASK for every downloaded case, Sim-CBCT included
	for task in 1 2; do \
	  for region in AB HN TH; do \
	    $(ELX_PYTHON) scripts/preprocessing/prepare_pairs.py --year 2025 --task $$task --region $$region || exit 1; done; \
	  for region in brain pelvis; do \
	    $(ELX_PYTHON) scripts/preprocessing/prepare_pairs.py --year 2023 --task $$task --region $$region || exit 1; done; \
	done
	$(ELX_PYTHON) scripts/preprocessing/prepare_external.py --sim-cbct data/raw/sim_cbct
