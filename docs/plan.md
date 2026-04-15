# Plan

## Overview

This document captures near-term staged work for `cubesim`. It is a working
plan for the current scaffold rather than a source of truth for current
behavior.

## Phase Plan

### Phase 1

Define and stabilize the initial package-root API.

- define the supported package-root exports
- merge the corresponding implementation work
- add tests that cover the declared public API

### Phase 2

Add config-driven instrument definitions.

- define the initial instrument configuration contract
- decide where config loading and validation live
- add tests for accepted and rejected config shapes

### Phase 3

Implement the minimal ETC compute path.

- define the minimal compute inputs and result shape
- implement the first supported compute flow
- add tests for compute preconditions and result behavior

### Phase 4

Establish artifact handling for atmospheric and instrument-specific inputs.

- define the initial artifact ownership boundaries
- validate artifact-loading expectations
- add tests for supported artifact inputs and failure cases

## Deliverables

- a stable initial public API
- config-driven instrument definitions
- a minimal ETC compute path
- initial artifact-handling rules

## Assumptions And Deferred Decisions

- Active implementation work may land from another branch before this plan is
  revised again.
- This plan does not change the current source-of-truth docs for implemented
  behavior.
- Broader ETC scope stays deferred until the minimal compute path and artifact
  contracts are stable.
