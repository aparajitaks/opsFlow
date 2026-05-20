import os
import sys
import time
import json
import math

# Ensure workspace root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.pipeline import rag_pipeline
from core.config import settings

# ANSI formatting utilities for a stunning visual report
GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
CYAN = "\033[1;36m"
MAGENTA = "\033[1;35m"
BOLD = "\033[1m"
RESET = "\033[0m"

IN_DOMAIN_QUERIES = [
    # 1. Lockout/tagout & Safety Procedures
    ("LOTO Steps", "What are the six sequential steps of the LOTO-E-401 lockout/tagout protocol?"),
    ("Arc Flash PPE", "What Level 3.1 PPE is required for Category 2 arc flash protection inside the 480V cabinet?"),
    ("Insulated Gloves", "What Class 0 insulated rubber glove test and voltage limits must be checked before use?"),
    ("E-Stop Locations", "Where are the red mushroom Emergency Stop (E-Stop) buttons located on the equipment and conveyor?"),
    
    # 2. Hydraulics & Pressure Systems
    ("Hydraulic Cavitation", "What are the causes and symptoms of hydraulic fluid pump cavitation (Hydrax-P50)?"),
    ("Bypass Leaks", "How do you verify internal bypass cylinder seal wear or a blown piston seal?"),
    ("Nitrogen Accumulator", "How do you diagnose a nitrogen bladder accumulator using the Schrader valve on the AccuForce-10L?"),
    ("PRV Symptoms", "What are the symptoms and verification steps for a faulty Pressure Relief Valve PRV-200?"),
    ("PRV Overhaul", "Explain the step-by-step overhaul, seat lapping, and calibration setting procedure for a PRV-200."),
    
    # 3. CNC Spindle & Linear Axes
    ("Spindle Overheating", "What are the causes of a Spindle-Sync-22kW CNC lathe spindle motor overheating when it hits 85C?"),
    ("V-Belt Tension", "How do you verify drive V-belt tension using a sonic tension meter and what is the nominal frequency?"),
    ("Magnetic Speed Sensor", "How do you adjust the non-contact magnetic speed sensor Mag-Pick-50 gap alignment?"),
    ("Ballscrew Backlash", "Describe the mechanical backlash double-nut preload adjustment procedure for the Ball-X-40 ballscrew."),
    
    # 4. Vibrations & Bearing Analysis
    ("Vibration BPFO Peak", "What vibration peak occurs for SKF-NU208 cylindrical outer race defects BPFO?"),
    ("Vibration BPFI Peak", "What vibration peak occurs for SKF-NU208 cylindrical inner race defects BPFI?"),
    ("Sensor Mounting", "How do you mount and torque a Viber-100 acceleration sensor on the active load zone?"),
    ("Bearing Replacement", "Describe the step-by-step rolling-element bearing SKF-NU208 replacement, puller, and induction heating checklist."),
    
    # 5. Tribology & Lubrication
    ("NLGI Greases", "What is the difference between NLGI Grade 2 and NLGI Grade 1 consistency greases and when is Grade 1 used?"),
    ("AES Spectroscopy", "What wear metal concentrations under Atomic Emission Spectroscopy indicate gear or shaft wear?"),
    ("Grease Gun Calibration", "Explain the grams-per-stroke calibration method for a manual grease gun."),
    ("Lube Auto Dosing", "What are the auto dosing cycle parameters and low-level interlocks for the Lube-Flow-10 centralized pump?"),
    
    # 6. Sensors & Telemetry
    ("4-20mA Telemetry", "What is the zero-point baseline and full-scale limit for a 4-20mA current loop?"),
    ("Pt100 RTD Specs", "What are the diagnostics for Pt100 RTD resistance curves at 0C and 100C?"),
    ("Ground Loops", "How do you identify and correct ground loop contamination in shielded signal cables?")
]

OUT_OF_DOMAIN_QUERIES = [
    ("Capital of France", "What is the capital of France?"),
    ("Cake Recipe", "How do you bake a chocolate cake?"),
    ("Dog Joke", "Tell me a joke about dogs.")
]

REFUSAL_MESSAGE = "I don't have enough information in my knowledge base to answer this question."

def run_rag_validation():
    print(f"\n{BOLD}{CYAN}=================================================================={RESET}")
    print(f"{BOLD}{CYAN}      ⚙️  opsFlow: AUTOMATED RAG PIPELINE COVERAGE VALIDATION     {RESET}")
    print(f"{BOLD}{CYAN}=================================================================={RESET}")
    
    # Initialize RAG Pipeline
    print(f"{YELLOW}[System] Initializing unified RAG pipeline and vector indexes...{RESET}")
    start_init = time.time()
    rag_pipeline.initialize_pipeline()
    init_duration = time.time() - start_init
    print(f"{GREEN}[System] Pipeline initialized successfully in {init_duration:.2f} seconds.{RESET}")
    
    in_domain_results = []
    out_of_domain_results = []
    
    # Loop over In-Domain Queries
    print(f"\n{BOLD}{MAGENTA}--- PHASE 1: EVALUATING IN-DOMAIN TECHNICAL QUERIES ({len(IN_DOMAIN_QUERIES)} TESTS) ---{RESET}")
    for idx, (label, query) in enumerate(IN_DOMAIN_QUERIES, 1):
        print(f"\n{BOLD}[Test {idx}/{len(IN_DOMAIN_QUERIES)}] {label}{RESET}")
        print(f" Query: {CYAN}\"{query}\"{RESET}")
        
        start_q = time.time()
        res = rag_pipeline.run_query(query)
        elapsed_ms = (time.time() - start_q) * 1000
        
        ans = res.get("answer", "")
        conf = res.get("confidence_score", 0.0)
        chunks = res.get("retrieved_chunks", [])
        faith = res.get("faithfulness", {})
        
        # Verify confidence is high (should be >= 0.30)
        conf_passed = conf >= 0.30
        # Verify answer is not the refusal message
        not_refused = REFUSAL_MESSAGE not in ans
        # Verify faithfulness
        faithful_passed = faith.get("faithful", False)
        
        status_symbol = f"{GREEN}✅ PASS{RESET}" if (conf_passed and not_refused and faithful_passed) else f"{RED}❌ FAIL{RESET}"
        
        print(f"  └─ Confidence Score : {GREEN if conf_passed else RED}{conf:.4f}{RESET} (Threshold >= 0.30)")
        print(f"  └─ Retrieved Chunks : {GREEN if chunks else YELLOW}{len(chunks)} chunks{RESET}")
        for c_idx, c in enumerate(chunks[:2]):
            print(f"     • [{c_idx+1}] Doc: {c['doc_name']} (CE Score: {c.get('score', 0.0):.2f})")
        print(f"  └─ Grounded Answer  : {GREEN if not_refused else RED}\"{ans[:100]}...\"{RESET}")
        print(f"  └─ Factual Audit    : {'✅ Faithful' if faithful_passed else '❌ Unfaithful'} (Verdict: \"{faith.get('verdict')}\")")
        print(f"  └─ Latency          : {CYAN}{elapsed_ms:.1f} ms{RESET}")
        print(f"  └─ Status           : {status_symbol}")
        
        in_domain_results.append({
            "label": label,
            "query": query,
            "confidence": conf,
            "chunks_count": len(chunks),
            "refused": not not_refused,
            "faithful": faithful_passed,
            "latency_ms": elapsed_ms,
            "passed": conf_passed and not_refused and faithful_passed
        })

    # Loop over Out-of-Domain Queries
    print(f"\n{BOLD}{MAGENTA}--- PHASE 2: EVALUATING OUT-OF-DOMAIN BLOCK/REFUSAL SAFETY ({len(OUT_OF_DOMAIN_QUERIES)} TESTS) ---{RESET}")
    for idx, (label, query) in enumerate(OUT_OF_DOMAIN_QUERIES, 1):
        print(f"\n{BOLD}[Safety Test {idx}/{len(OUT_OF_DOMAIN_QUERIES)}] {label}{RESET}")
        print(f" Query: {CYAN}\"{query}\"{RESET}")
        
        start_q = time.time()
        res = rag_pipeline.run_query(query)
        elapsed_ms = (time.time() - start_q) * 1000
        
        ans = res.get("answer", "")
        conf = res.get("confidence_score", 0.0)
        chunks = res.get("retrieved_chunks", [])
        
        # Verify confidence is low (should be < 0.30)
        conf_low = conf < 0.30
        # Verify response matches the required Refusal statement exactly
        refused_correctly = ans == REFUSAL_MESSAGE
        
        status_symbol = f"{GREEN}✅ PASS (REFUSED){RESET}" if (conf_low and refused_correctly) else f"{RED}❌ FAIL (NOT BLOCKED){RESET}"
        
        print(f"  └─ Confidence Score : {GREEN if conf_low else RED}{conf:.4f}{RESET} (Threshold < 0.30)")
        print(f"  └─ Grounded Answer  : {GREEN if refused_correctly else RED}\"{ans}\"{RESET}")
        print(f"  └─ Status           : {status_symbol}")
        
        out_of_domain_results.append({
            "label": label,
            "query": query,
            "confidence": conf,
            "chunks_count": len(chunks),
            "refused_correctly": refused_correctly,
            "passed": conf_low and refused_correctly
        })
        
    # --- PHASE 3: METRICS AND SUMMARY REPORT ---
    print(f"\n{BOLD}{CYAN}=================================================================={RESET}")
    print(f"{BOLD}{CYAN}                  RAG PIPELINE METRICS SUMMARY                    {RESET}")
    print(f"{BOLD}{CYAN}=================================================================={RESET}")
    
    total_in_domain = len(in_domain_results)
    passed_in_domain = sum(1 for r in in_domain_results if r["passed"])
    avg_in_conf = sum(r["confidence"] for r in in_domain_results) / total_in_domain
    avg_latency = sum(r["latency_ms"] for r in in_domain_results) / total_in_domain
    
    total_out_domain = len(out_of_domain_results)
    passed_out_domain = sum(1 for r in out_of_domain_results if r["passed"])
    avg_out_conf = sum(r["confidence"] for r in out_of_domain_results) / total_out_domain
    
    print(f"{BOLD}1. IN-DOMAIN TECHNICAL COVERAGE:{RESET}")
    print(f"   • Total Tested             : {total_in_domain}")
    print(f"   • Successfully Grounded    : {GREEN if passed_in_domain == total_in_domain else YELLOW}{passed_in_domain}/{total_in_domain}{RESET}")
    print(f"   • Technical Coverage Rate  : {GREEN if passed_in_domain == total_in_domain else YELLOW}{passed_in_domain/total_in_domain*100:.1f}%{RESET}")
    print(f"   • Avg Confidence Score     : {GREEN}{avg_in_conf:.4f}{RESET}")
    print(f"   • Avg Retrieval Latency    : {CYAN}{avg_latency:.1f} ms{RESET}")
    
    print(f"\n{BOLD}2. OUT-OF-DOMAIN SAFETY REJECTION FIREWALL:{RESET}")
    print(f"   • Total Tested             : {total_out_domain}")
    print(f"   • Successfully Refused     : {GREEN if passed_out_domain == total_out_domain else RED}{passed_out_domain}/{total_out_domain}{RESET}")
    print(f"   • Safety Refusal Rate      : {GREEN if passed_out_domain == total_out_domain else RED}{passed_out_domain/total_out_domain*100:.1f}%{RESET}")
    print(f"   • Avg Out-of-Domain Conf   : {GREEN}{avg_out_conf:.4f}{RESET}")
    
    all_passed = (passed_in_domain == total_in_domain) and (passed_out_domain == total_out_domain)
    
    print(f"\n{BOLD}3. OVERALL SYSTEM AUDIT STATUS:{RESET}")
    if all_passed:
        print(f"   🏆 {BOLD}{GREEN}ALL SYSTEM ASSIGNMENT REQUIREMENTS FULFILLED SUCCESSFULLY!{RESET}")
    else:
        print(f"   ⚠️  {BOLD}{RED}SOME RAG EVALUATIONS ENCOUNTERED ANOMALIES. INSPECT LOGS ABOVE.{RESET}")
        
    print(f"{BOLD}{CYAN}=================================================================={RESET}\n")

if __name__ == "__main__":
    run_rag_validation()
