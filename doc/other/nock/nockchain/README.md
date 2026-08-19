# Nockchain

## Overview

A decentralized ledger using Nock for formal verification and smart contract execution.

## Purpose

To provide a secure, formally verified foundation for distributed Active Inference agents.

## 🔒 Security Implications for Passive Agents

Running Active Inference on-chain via Nock provides unique security properties:
- **Determinism**: Every belief update is provably deterministic and verifiable by all nodes.
- **Privacy**: Use Nock's formal structure to specify "private belief blocks" that are never revealed in plaintext but still contribute to valid state updates.
- **Sandboxing**: The Nock VM ensures that a malicious agent cannot execute arbitrary code beyond its specified GNN policy.

## Contents
- Nockchain specification
- Integration with Iroh P2P
- Smart contract examples
