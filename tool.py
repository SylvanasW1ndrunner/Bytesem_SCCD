"""
Purpose: Generates control flow graphs (CFG) from smart contract bytecode and creates function-basic block mappings.
Converts raw bytecode into structured graph representations for GNN analysis.
"""

import pandas as pd
from evm_cfg_builder.cfg import CFG
import json
import os
from typing import Dict, List, Set
from sklearn.model_selection import train_test_split

class ContractCFGGenerator:
    def __init__(self, bytecode_path: str):
        with open(bytecode_path, 'r') as f:
            self.bytecode = f.read().strip()
        self.cfg = CFG(self.bytecode)
        self.contract_name = os.path.basename(bytecode_path).replace('.txt', '')

    # TODO: Generate both DOT and JSON files for a contract
    def generate_files(self, output_dir: str = "output"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cfg_dir = os.path.join(output_dir, "cfg")
        function_dir = os.path.join(output_dir, "function")

        if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir)
        if not os.path.exists(function_dir):
            os.makedirs(function_dir)

        dot_file = os.path.join(cfg_dir, f"{self.contract_name}_cfg.dot")
        self.generate_dot_file(dot_file)

        json_file = os.path.join(function_dir, f"{self.contract_name}_function_mapping.json")
        self.generate_json_file(json_file)

        print(f"生成文件:")
        print(f"  DOT文件: {dot_file}")
        print(f"  JSON文件: {json_file}")
    
    # TODO: Get all functions that contain a specific basic block
    def _get_block_functions(self, bb):
        belonging_functions = []
        for function in self.cfg.functions:
            if bb in function.basic_blocks:
                belonging_functions.append(function)
        return belonging_functions
    
    # TODO: Generate DOT file representing the entire contract's control flow graph
    def generate_dot_file(self, output_file: str):
        dot_lines = []
        dot_lines.append(f'digraph "{self.contract_name}_CFG" {{')
        dot_lines.append('    rankdir=TB;')
        dot_lines.append('    node [shape=box, style=filled];')
        dot_lines.append('    edge [fontsize=10];')
        dot_lines.append('')

        dot_lines.append(f'    label="{self.contract_name} Control Flow Graph";')
        dot_lines.append('    labelloc=t;')
        dot_lines.append('    fontsize=16;')
        dot_lines.append('')

        all_basic_blocks = set()
        for function in self.cfg.functions:
            all_basic_blocks.update(function.basic_blocks)

        block_id_map = {}
        for i, bb in enumerate(sorted(all_basic_blocks, key=lambda x: x.start.pc)):
            block_id = f"block_{i}"
            block_id_map[bb] = block_id
            label = f"Block {i}"

            instructions_with_operands = []
            for ins in bb.instructions:
                if hasattr(ins, 'operand') and ins.operand is not None:
                    if isinstance(ins.operand, int):
                        operand_hex = f"0x{ins.operand:x}"
                    else:
                        operand_hex = str(ins.operand)
                    instructions_with_operands.append(f"{ins.name} {operand_hex}")
                else:
                    instructions_with_operands.append(ins.name)

            instr_text = "\\n".join(instructions_with_operands) if instructions_with_operands else "No Instructions"

            belonging_functions = self._get_block_functions(bb)
            if belonging_functions:
                function_names = []
                for func in belonging_functions:
                    if func.hash_id:
                        if isinstance(func.hash_id, int):
                            selector = f"0x{func.hash_id:x}"
                        elif isinstance(func.hash_id, str):
                            selector = func.hash_id if func.hash_id.startswith('0x') else f"0x{func.hash_id}"
                        else:
                            selector = str(func.hash_id)
                        function_names.append(f"{func.name}({selector})")
                    else:
                        function_names.append(func.name)
                function_text = "\\n".join(function_names)
            else:
                function_text = "No Function"

            node_color = self._get_node_color(bb)

            dot_lines.append(f'    {block_id} [')
            dot_lines.append(f'        label="Label: {label}\\nInstr: {instr_text}\\nFunction: {function_text}",')
            dot_lines.append(f'        fillcolor="{node_color}",')
            dot_lines.append('        fontsize=9,')
            dot_lines.append('        shape=box')
            dot_lines.append('    ];')

        dot_lines.append('')

        added_edges = set()

        for function in self.cfg.functions:
            for bb in function.basic_blocks:
                if bb not in block_id_map:
                    continue

                source_id = block_id_map[bb]

                for outgoing_bb in bb.outgoing_basic_blocks(function.key):
                    if outgoing_bb not in block_id_map:
                        continue

                    target_id = block_id_map[outgoing_bb]
                    edge_key = (source_id, target_id)

                    if edge_key not in added_edges:
                        added_edges.add(edge_key)

                        edge_label = self._get_edge_label(bb)
                        edge_style = self._get_edge_style(bb)

                        if edge_label:
                            dot_lines.append(f'    {source_id} -> {target_id} [label="{edge_label}", {edge_style}];')
                        else:
                            dot_lines.append(f'    {source_id} -> {target_id} [{edge_style}];')

        dot_lines.append('}')

        with open(output_file, 'w') as f:
            f.write('\n'.join(dot_lines))

    # TODO: Generate JSON file with function-basic block mapping information
    def generate_json_file(self, output_file: str):
        all_basic_blocks = set()
        for function in self.cfg.functions:
            all_basic_blocks.update(function.basic_blocks)

        block_id_map = {}
        block_info_map = {}

        for i, bb in enumerate(sorted(all_basic_blocks, key=lambda x: x.start.pc)):
            block_id = f"block_{i}"
            block_id_map[bb] = block_id

            block_info_map[block_id] = {
                "block_id": block_id,
                "block_number": i,
                "start_pc": f"0x{bb.start.pc:x}",
                "end_pc": f"0x{bb.end.pc:x}",
                "start_pc_decimal": bb.start.pc,
                "end_pc_decimal": bb.end.pc,
                "instructions": [
                    {
                        "name": ins.name,
                        "pc": ins.pc,
                        "operand": ins.operand if hasattr(ins, 'operand') else None
                    } for ins in bb.instructions
                ],
                "instruction_count": len(bb.instructions)
            }

        function_membership = {}

        for function in sorted(self.cfg.functions, key=lambda x: x.start_addr):
            function_selector = function.hash_id
            if function_selector:
                if isinstance(function_selector, int):
                    function_selector = f"0x{function_selector:x}"
                elif isinstance(function_selector, str) and not function_selector.startswith('0x'):
                    function_selector = f"0x{function_selector}"

            function_key = function.name if function.name else f"func_0x{function.start_addr:x}"

            function_blocks = []
            function_block_ids = []

            for bb in sorted(function.basic_blocks, key=lambda x: x.start.pc):
                if bb in block_id_map:
                    block_id = block_id_map[bb]
                    function_block_ids.append(block_id)

                    block_info = block_info_map[block_id].copy()
                    block_info["incoming_blocks"] = [
                        block_id_map[inc_bb]
                        for inc_bb in bb.incoming_basic_blocks(function.key)
                        if inc_bb in block_id_map
                    ]
                    block_info["outgoing_blocks"] = [
                        block_id_map[out_bb]
                        for out_bb in bb.outgoing_basic_blocks(function.key)
                        if out_bb in block_id_map
                    ]
                    function_blocks.append(block_info)

            function_info = {
                "function_name": function.name,
                "function_selector": function_selector,
                "start_address": f"0x{function.start_addr:x}",
                "start_address_decimal": function.start_addr,
                "attributes": list(function.attributes),
                "basic_blocks_count": len(function.basic_blocks),
                "basic_block_ids": function_block_ids,
                "basic_blocks_detail": function_blocks
            }

            function_membership[function_key] = function_info

        contract_cfg_data = {
            "contract_name": self.contract_name,
            "total_functions": len(self.cfg.functions),
            "total_basic_blocks": len(all_basic_blocks),
            "all_basic_blocks": block_info_map,
            "function_membership": function_membership,
            "metadata": {
                "generated_by": "evm_cfg_builder",
                "description": "Function-to-BasicBlock mapping for smart contract CFG analysis",
                "note": "basic_block_ids shows the simple mapping F = {func: [block_0, block_1, ...]}"
            }
        }

        with open(output_file, 'w') as f:
            json.dump(contract_cfg_data, f, indent=2, ensure_ascii=False)

    # TODO: Determine node color based on instruction types in basic block
    def _get_node_color(self, bb) -> str:
        instructions = [ins.name for ins in bb.instructions]

        if any(ins in ['RETURN', 'REVERT', 'STOP', 'SELFDESTRUCT'] for ins in instructions):
            return 'lightcoral'
        elif any(ins == 'JUMPDEST' for ins in instructions):
            return 'lightblue'
        elif any(ins in ['CALL', 'DELEGATECALL', 'STATICCALL', 'CALLCODE'] for ins in instructions):
            return 'lightyellow'
        elif any(ins in ['SSTORE', 'SLOAD'] for ins in instructions):
            return 'lightgreen'
        else:
            return 'lightgray'

    # TODO: Get edge label based on the last instruction in basic block
    def _get_edge_label(self, bb) -> str:
        if not bb.instructions:
            return ""

        last_instruction = bb.instructions[-1].name
        if last_instruction == 'JUMPI':
            return "conditional"
        elif last_instruction == 'JUMP':
            return "jump"
        elif last_instruction in ['RETURN', 'REVERT', 'STOP']:
            return "exit"
        else:
            return ""

    # TODO: Get edge style based on the last instruction in basic block
    def _get_edge_style(self, bb) -> str:
        if not bb.instructions:
            return 'color=black'

        last_instruction = bb.instructions[-1].name
        if last_instruction == 'JUMPI':
            return 'color=blue, style=dashed'
        elif last_instruction == 'JUMP':
            return 'color=red'
        elif last_instruction in ['RETURN', 'REVERT', 'STOP']:
            return 'color=gray, style=dotted'
        else:
            return 'color=black'

# TODO: Process a single contract bytecode file
def process_contract(bytecode_path: str, output_dir: str = "cfg_output"):
    generator = ContractCFGGenerator(bytecode_path)
    generator.generate_files(output_dir)

# TODO: Batch process multiple contract bytecode files
def batch_process_contracts(contracts_dir: str, output_dir: str = "cfg_output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(contracts_dir):
        if filename.endswith('.txt'):
            contract_path = os.path.join(contracts_dir, filename)
            print(f"\n处理合约: {filename}")
            try:
                process_contract(contract_path, output_dir)
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

# TODO: Split dataset into train/validation/test sets with stratified sampling
def data_process(csv_path, type1_csv, type2_csv, type3_csv):
    df = pd.read_csv(csv_path)

    train_list, val_list, test_list = [], [], []

    type0 = df[df['type'] == 0]
    type0_train, type0_remain = train_test_split(type0, test_size=0.2, random_state=42)
    type0_val, type0_test = train_test_split(type0_remain, test_size=0.75, random_state=42)

    type4 = df[df['type'] == 4]
    type4_train, type4_remain = train_test_split(type4, test_size=0.2, random_state=42)
    type4_val, type4_test = train_test_split(type4_remain, test_size=0.75, random_state=42)

    for t in [1, 2, 3]:
        type_t = df[df['type'] == t]
        type_t_val, type_t_test = train_test_split(type_t, test_size=0.8, random_state=42)
        val_list.append(type_t_val)
        test_list.append(type_t_test)

    train_list.extend([type0_train, type4_train])
    val_list.extend([type0_val, type4_val])
    test_list.extend([type0_test, type4_test])

    train_set = pd.concat(train_list).reset_index(drop=True)
    val_set = pd.concat(val_list).reset_index(drop=True)
    test_set = pd.concat(test_list).reset_index(drop=True)

    train_set.to_csv("train.csv", index=False)
    val_set.to_csv("val.csv", index=False)
    test_set.to_csv("test.csv", index=False)

    return train_set, val_set, test_set

# TODO: Extract Type 0 and Type 4 samples for encoder testing
def encodercsv_extract(csv_path):
    df = pd.read_csv(csv_path)

    type0 = df[df['type'] == 0.0]
    type4 = df[df['type'] == 4.0]
    print(len(type0), len(type4))

    df = pd.concat([type0, type4]).reset_index(drop=True)
    df.to_csv("encoder_test.csv", index=False)

encodercsv_extract("test.csv")