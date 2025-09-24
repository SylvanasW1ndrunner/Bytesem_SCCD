"""
Purpose: Processes control flow graphs to extract and abstract instruction sequences.
Generates training corpus for sequence-to-sequence autoencoder models.
"""

import pandas as pd
from matplotlib import pyplot as plt
import os
import re
import pydot
import networkx as nx
import fasttext
import os
import numpy as np
from typing import List, Tuple, Optional
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_model = None

cfg_path = "GNNdata/cfg"
instr_list = []

# TODO: Display control flow graph visualization
def show_cfg(cfg):
    graphs = pydot.graph_from_dot_file(cfg)
    graph = nx.drawing.nx_pydot.from_pydot(graphs[0])
    nx.draw(graph, with_labels=True)
    plt.show()

# TODO: Parse node label string to extract structured instruction information
def parse_node_label(label_str: str) -> dict:
    parsed_info = {
        'id': None,
        'function': 'unknown',
        'instructions': []
    }
    instr_pattern = re.compile(r"([A-Z0-9]+)(?:\s+(0x[0-9a-fA-F]+))?")
    in_instr_section = False
    for line in label_str.split('\\n'):
        line = line.strip()
        if line.startswith("Label: Block"):
            match = re.search(r"Block\s+(\d+)", line)
            if match:
                parsed_info['id'] = match.group(1)
            continue
        if line.startswith("Function:"):
            match = re.search(r"Function:\s+([\w_]+)", line)
            if match:
                parsed_info['function'] = match.group(1)
            continue
        if line.startswith("Instr:"):
            in_instr_section = True
            line = line.replace("Instr:", "").strip()
        if in_instr_section:
            match = instr_pattern.match(line)
            if match:
                name, operand = match.groups()
                instr_dict = {'name': name}
                if operand:
                    instr_dict['operand'] = operand
                parsed_info['instructions'].append(instr_dict)
    return parsed_info

# TODO: Check if operand value looks like a mask pattern
def _is_mask_value(operand_str: str) -> bool:
    try:
        val = int(operand_str, 16)
        bits = bin(val)[2:]
        if not bits: return False
        ones = bits.count('1')
        zeros = len(bits) - ones
        if ones / len(bits) > 0.8 or zeros / len(bits) > 0.8:
            return True
        if operand_str.endswith('ff' * 4) or operand_str.endswith('00' * 4):
            return True
    except (ValueError, TypeError):
        return False
    return False

# TODO: Abstract opcodes with semantic understanding for semantic encoder
def abstract_opcodes(instructions: list, is_in_dispatcher: bool) -> list[str]:
    tokens = []
    i = 0
    while i < len(instructions):
        instr = instructions[i]
        name = instr['name']

        if name.startswith('PUSH'):
            operand_str = instr.get('operand', '0x0')
            operand_int = int(operand_str, 16)
            next_instr = instructions[i + 1] if i + 1 < len(instructions) else None

            token_to_add = None
            push_size = int(name[4:])
            if name == 'PUSH32' and (
                    (next_instr and next_instr['name'].startswith('LOG')) or
                    re.fullmatch(r'0x[a-fA-F0-9]{64}', operand_str)
            ):
                print(f"operand_str:is {operand_str},is HASH")
                token_to_add = 'PUSH_HASH'
            elif name == 'PUSH32' and next_instr and next_instr['name'] in ['SLOAD', 'SSTORE']:
                token_to_add = 'PUSH_SLOT'
            elif _is_mask_value(operand_str) or (next_instr and next_instr['name'] in ['AND', 'OR']):
                token_to_add = 'PUSH_MASK'
            elif name == 'PUSH4' and is_in_dispatcher:
                token_to_add = 'PUSH_SELECTOR'
            elif name == 'PUSH20' and (next_instr and ['name'] in ['CALL', 'DELEGATECALL', 'STATICCALL', 'EQ']):
                token_to_add = 'PUSH_ADDRESS'
            elif name == 'PUSH1' and operand_int in [0x40, 0x60]:
                token_to_add = 'PUSH_MEM_POINTER'
            elif push_size <= 4 and 0 <= operand_int <= 1023:
                token_to_add = f"PUSH_VAL{operand_int}"
            else:
                token_to_add = 'PUSH_IMM'

            tokens.append(token_to_add)
            i += 1
            continue

        tokens.append(name)
        i += 1

    return tokens

# TODO: Abstract opcodes without semantic understanding for baseline encoder
def abstract_opcodes_baseline(instructions: list, **kwargs) -> list[str]:
    return [instr['name'] for instr in instructions]

# TODO: Extract and abstract instructions from all CFG files
def extract_instrs_and_abstract(cfg_path: str):
    all_processed_blocks = []
    if not os.path.exists(cfg_path):
        print(f"错误: 路径 '{cfg_path}' 不存在。请修改为正确的路径。")
        return

    print("开始处理...")
    for os_file in sorted(os.listdir(cfg_path)):
        if os_file.endswith('.dot'):
            file_path = os.path.join(cfg_path, os_file)
            print(f"--- 正在处理文件: {os_file} ---")

            try:
                graphs = pydot.graph_from_dot_file(file_path)
                graph = nx.drawing.nx_pydot.from_pydot(graphs[0])
            except Exception as e:
                print(f"    无法加载或解析dot文件: {e}")
                continue

            dispatcher_node_ids = {
                node_id for node_id, node_data in graph.nodes(data=True)
                if "Function: _dispatcher" in node_data.get('label', '')
            }

            for node_id, node_data in graph.nodes(data=True):
                label_str = node_data.get('label', '')
                if not label_str: continue

                parsed_info = parse_node_label(label_str)
                instructions = parsed_info['instructions']
                if not instructions: continue

                is_in_dispatcher = node_id in dispatcher_node_ids
                processed_tokens = abstract_opcodes_baseline(instructions)
                all_processed_blocks.append(" ".join(processed_tokens))

    print(f"\n--- 处理完成 ---\n总共生成了 {len(all_processed_blocks)} 个处理后的基本块（句子）。")

    print("\n结果预览 (前10条):")
    for sentence in all_processed_blocks[:10]:
        print(sentence)

    with open("fasttext_corpus_final.txt", "w") as f:
        for sentence in all_processed_blocks:
            f.write(sentence + "\n")
    print("\n所有结果已保存到 no_abstract_fasttext_corpus_final.txt 文件中。")

    return all_processed_blocks

# TODO: Process single DOT file and write processed version
def process_dot_file(dot_path, output_path):
    print(f"--- 正在处理文件: {dot_path} ---")
    try:
        pydot_graphs = pydot.graph_from_dot_file(dot_path)
        if not pydot_graphs:
            print(f"    错误: 文件 '{dot_path}' 不是一个有效的DOT文件或为空。")
            return
        pydot_graph = pydot_graphs[0]

        dispatcher_node_names = {
            node.get_name().strip('"') for node in pydot_graph.get_nodes()
            if node.get('label') and "Function: _dispatcher" in node.get('label')
        }
        print(f"    找到 {len(dispatcher_node_names)} 个 dispatcher 块。")

        print("    正在处理节点标签...")
        for node in tqdm(pydot_graph.get_nodes(), desc="Processing nodes"):
            node_name = node.get_name().strip('"')
            label_str = node.get('label')

            if not label_str: continue

            parsed_info = parse_node_label(label_str)
            instructions = parsed_info['instructions']

            if not instructions:
                node.set('label', f'"{node_name}"')
                continue

            is_in_dispatcher = node_name in dispatcher_node_names

            processed_tokens = abstract_opcodes_baseline(instructions)
            new_label = " ".join(processed_tokens)

            node.set('label', f'"{new_label}"')

            node.set('shape', 'box')
            node.set('fontsize', '10')

        pydot_graph.write_dot(output_path)
        print(f"    处理完成，结果已保存到: {output_path}")

    except Exception as e:
        print(f"    处理文件时发生严重错误: {e}")

extract_instrs_and_abstract(cfg_path)
lines = set(open('fasttext_corpus_final.txt', 'r', encoding='utf-8').readlines())
print(len(sorted(list(lines))))