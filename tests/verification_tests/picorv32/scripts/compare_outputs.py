#!/usr/bin/env python3
"""
Compare outputs between SystemC and Verilator implementations of PicoRV32
"""

import sys
import os
import subprocess
import argparse
import json
import re
from datetime import datetime

class OutputComparator:
    def __init__(self, systemc_exe, verilator_exe, program_hex):
        self.systemc_exe = systemc_exe
        self.verilator_exe = verilator_exe
        self.program_hex = program_hex
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'program': program_hex,
            'systemc': {},
            'verilator': {},
            'comparison': {}
        }
    
    def run_systemc(self, timeout=10000):
        """Run SystemC implementation"""
        print(f"Running SystemC: {self.systemc_exe}")
        
        cmd = [self.systemc_exe, '--hex', self.program_hex, '--timeout', str(timeout)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout/100)
            self.results['systemc'] = {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
            
            # Parse output for key metrics
            self.parse_systemc_output(result.stdout)
            
        except subprocess.TimeoutExpired:
            self.results['systemc']['timeout'] = True
            self.results['systemc']['success'] = False
            print("SystemC simulation timed out")
        except Exception as e:
            self.results['systemc']['error'] = str(e)
            self.results['systemc']['success'] = False
            print(f"SystemC simulation error: {e}")
        
        return self.results['systemc'].get('success', False)
    
    def run_verilator(self, timeout=10000):
        """Run Verilator implementation"""
        print(f"Running Verilator: {self.verilator_exe}")
        
        cmd = [self.verilator_exe, '--hex', self.program_hex, '--timeout', str(timeout)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout/100)
            self.results['verilator'] = {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
            
            # Parse output for key metrics
            self.parse_verilator_output(result.stdout)
            
        except subprocess.TimeoutExpired:
            self.results['verilator']['timeout'] = True
            self.results['verilator']['success'] = False
            print("Verilator simulation timed out")
        except Exception as e:
            self.results['verilator']['error'] = str(e)
            self.results['verilator']['success'] = False
            print(f"Verilator simulation error: {e}")
        
        return self.results['verilator'].get('success', False)
    
    def parse_systemc_output(self, output):
        """Parse SystemC output for metrics"""
        metrics = {}
        
        # Look for instruction count
        match = re.search(r'Instructions executed: (\d+)', output)
        if match:
            metrics['instructions'] = int(match.group(1))
        
        # Look for cycles
        match = re.search(r'cycle[s]?: (\d+)', output, re.IGNORECASE)
        if match:
            metrics['cycles'] = int(match.group(1))
        
        # Look for UART output
        uart_output = []
        for line in output.split('\n'):
            if 'UART:' in line:
                uart_output.append(line.split('UART:')[1].strip())
        if uart_output:
            metrics['uart_output'] = ''.join(uart_output)
        
        # Look for trap
        if 'trap' in output.lower():
            metrics['trap'] = True
        
        self.results['systemc']['metrics'] = metrics
    
    def parse_verilator_output(self, output):
        """Parse Verilator output for metrics"""
        metrics = {}
        
        # Look for instruction count
        match = re.search(r'Instructions: (\d+)', output)
        if match:
            metrics['instructions'] = int(match.group(1))
        
        # Look for cycles
        match = re.search(r'cycle[s]?: (\d+)', output, re.IGNORECASE)
        if match:
            metrics['cycles'] = int(match.group(1))
        
        # Look for UART characters (direct output)
        uart_lines = []
        for line in output.split('\n'):
            if not any(keyword in line for keyword in ['Instructions:', 'Reset', 'trap', 'cycle']):
                # Assume it's UART output
                uart_lines.append(line)
        if uart_lines:
            metrics['uart_output'] = '\n'.join(uart_lines)
        
        # Look for trap
        if 'trap' in output.lower():
            metrics['trap'] = True
        
        self.results['verilator']['metrics'] = metrics
    
    def compare_results(self):
        """Compare SystemC and Verilator results"""
        comparison = {}
        
        # Check if both simulations completed
        sc_success = self.results['systemc'].get('success', False)
        ver_success = self.results['verilator'].get('success', False)
        
        if not sc_success or not ver_success:
            comparison['status'] = 'INCOMPLETE'
            comparison['reason'] = 'One or both simulations failed'
            self.results['comparison'] = comparison
            return False
        
        # Compare metrics
        sc_metrics = self.results['systemc'].get('metrics', {})
        ver_metrics = self.results['verilator'].get('metrics', {})
        
        # Compare instruction counts
        if 'instructions' in sc_metrics and 'instructions' in ver_metrics:
            sc_inst = sc_metrics['instructions']
            ver_inst = ver_metrics['instructions']
            comparison['instruction_match'] = (sc_inst == ver_inst)
            comparison['instruction_diff'] = abs(sc_inst - ver_inst)
            if not comparison['instruction_match']:
                print(f"Instruction count mismatch: SystemC={sc_inst}, Verilator={ver_inst}")
        
        # Compare UART output
        if 'uart_output' in sc_metrics and 'uart_output' in ver_metrics:
            sc_uart = sc_metrics['uart_output']
            ver_uart = ver_metrics['uart_output']
            comparison['uart_match'] = (sc_uart == ver_uart)
            if not comparison['uart_match']:
                print(f"UART output mismatch:")
                print(f"  SystemC: {repr(sc_uart)}")
                print(f"  Verilator: {repr(ver_uart)}")
        
        # Compare trap status
        sc_trap = sc_metrics.get('trap', False)
        ver_trap = ver_metrics.get('trap', False)
        comparison['trap_match'] = (sc_trap == ver_trap)
        
        # Overall comparison result
        matches = [
            comparison.get('instruction_match', True),
            comparison.get('uart_match', True),
            comparison.get('trap_match', True)
        ]
        
        comparison['status'] = 'PASS' if all(matches) else 'FAIL'
        comparison['match_percentage'] = sum(matches) / len(matches) * 100
        
        self.results['comparison'] = comparison
        return comparison['status'] == 'PASS'
    
    def save_results(self, output_file):
        """Save comparison results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    def print_summary(self):
        """Print comparison summary"""
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        comp = self.results.get('comparison', {})
        status = comp.get('status', 'UNKNOWN')
        
        print(f"Status: {status}")
        
        if 'match_percentage' in comp:
            print(f"Match Rate: {comp['match_percentage']:.1f}%")
        
        if 'instruction_match' in comp:
            print(f"Instruction Count Match: {comp['instruction_match']}")
            if 'instruction_diff' in comp and comp['instruction_diff'] > 0:
                print(f"  Difference: {comp['instruction_diff']} instructions")
        
        if 'uart_match' in comp:
            print(f"UART Output Match: {comp['uart_match']}")
        
        if 'trap_match' in comp:
            print(f"Trap Status Match: {comp['trap_match']}")
        
        print("="*60)
        
        return status == 'PASS'

def main():
    parser = argparse.ArgumentParser(description='Compare PicoRV32 implementations')
    parser.add_argument('--systemc', required=True, help='Path to SystemC executable')
    parser.add_argument('--verilator', required=True, help='Path to Verilator executable')
    parser.add_argument('--program', required=True, help='Path to test program hex file')
    parser.add_argument('--timeout', type=int, default=10000, help='Simulation timeout in cycles')
    parser.add_argument('--output', default='comparison.json', help='Output JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Check if executables exist
    if not os.path.exists(args.systemc):
        print(f"Error: SystemC executable not found: {args.systemc}")
        return 1
    
    if not os.path.exists(args.verilator):
        print(f"Error: Verilator executable not found: {args.verilator}")
        return 1
    
    if not os.path.exists(args.program):
        print(f"Error: Program hex file not found: {args.program}")
        return 1
    
    # Create comparator and run tests
    comparator = OutputComparator(args.systemc, args.verilator, args.program)
    
    # Run both simulations
    sc_success = comparator.run_systemc(args.timeout)
    ver_success = comparator.run_verilator(args.timeout)
    
    # Compare results
    match = comparator.compare_results()
    
    # Save results
    comparator.save_results(args.output)
    
    # Print summary
    success = comparator.print_summary()
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
