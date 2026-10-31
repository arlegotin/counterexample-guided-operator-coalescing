# Confirmation access v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement inline; one fresh whole-feature review after focused checks.

**Goal:** Add durable, freeze-bound one-attempt confirmation access while preserving exploratory denial.
**Architecture:** Closed metadata issuer in poc/data/confirmation.py; immutable context/opaque permit and opt-in reader dispatch in access.py. Existing ExposureLedger schema records text lineage; exclusive grant journal prevents repeat/reset.
**Tech Stack:** Python3.12 stdlib, existing canonical records/locks, pytest owned fixtures.
**Spec:** docs/superpowers/specs/2026-10-31-confirmation-access-v1.md

## Global Constraints

- Owned fixtures only; no historical or new real protected content, no model jobs.
- Default role denials and legacy permit constructor unchanged.
- Durable claim before I/O; failures consume access, no automatic retry.
- Canonical paths, strict closed schema, frozen metadata hashes and explicit caller context.
- Record complete executed source including untracked files; retain all failures.

## Review Focus

- Two separately issued grants race on one population: exposure claim must serialize and permit only one read.
- Missing/replaced grant or exposure journal after issuance: fail before any text I/O.
- Process interruption between intent and completion: never reset to unused.
- Different caller analysis/method/partial population: explicit context and full digest deny.
- Metadata changes during materialization: return no text and retain consumed failure.

### Task 1: Freeze issuer and protected reader

**Files:** create poc/data/confirmation.py, tests/test_confirmation_access.py; modify poc/data/access.py.
**Interfaces:** consume DataManifest.digest(), ExposureLedger.record(), canonical_json/read_rows/write_row; produce EvaluationContext, issue_evaluation_permit(authorization, freeze, manifest), opt-in open_records(..., evaluation=...).

- [ ] Write owned fixture success and all binding/durable consumption negative controls; run to see missing issuer fail.
- [ ] Implement strict issuer, immutable permit/context and confirmation reader dispatch; keep exploratory path.
- [ ] Run new controls and existing access/split tests; inspect every failure and run focused green.
- [ ] One fresh whole-feature review, one material RED→GREEN fix pass, no rereview.
- [ ] Freeze source/environment, full CPU/public qualification and independently inspect fixture-only receipts; record qualified implementation and remaining real-confirmation prerequisites.
