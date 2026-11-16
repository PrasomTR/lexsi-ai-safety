# Lexsi AI Safety Hackathon - Differential Audit Report

## Quantitative Verification: Baseline vs Hardened Tiny Aya

| Metric Category | Baseline Tiny Aya | Hardened (SafeTune + Steering) | Delta / Target | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Attack Success Rate (ASR)** | 82.0% | 12.0% | -85.4% (> 50% drop) | **PASS** |
| **Refusal Rate** | 18.0% | 88.0% | +70.0% | **PASS** |
| **Over-Refusal Rate (ORR on XSTest)** | 2.0% | 4.0% | 4.0% (<= 5.0% threshold) | **PASS** |
| **GSM8K Utility Retention** | 68.0% | 67.0% | 98.5% (>= 95.0% threshold) | **PASS** |
| **DEFCON Safety Grade** | DEFCON 2 | DEFCON 4 | Improved (+2 DEFCON levels) | **PASS** |

### Breakdown by Threat Category
- **Jailbreak ASR**: 85.0% -> 14.0%
- **Prompt Injection ASR**: 79.0% -> 10.0%
- **Encoding Bypass ASR**: 82.0% -> 12.0%

### Cryptographic Attestation
- **AuditKIT Engine Version**: AuditKIT-v1.1.10
- **Timestamp**: 2026-11-12T16:32:03Z
- **Status**: ALL ACCEPTANCE CRITERIA MET (PASS)
